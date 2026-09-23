"""Transcribe a video on-device with Qwen3-ASR + Qwen3-ForcedAligner (mlx-qwen3-asr).

Extracts mono 16kHz audio via ffmpeg, transcribes it locally, aligns every
word to the waveform, optionally diarizes with pyannote, and writes a
Scribe-shaped transcript to <edit_dir>/transcripts/<video_stem>.json:

    {"language_code": "zh", "text": "...", "words": [
        {"text": "大家", "start": 0.12, "end": 0.40, "type": "word", "speaker_id": "speaker_0"},
        {"text": " ", "start": 0.40, "end": 0.52, "type": "spacing", "speaker_id": "speaker_0"},
        ...]}

Nothing leaves the machine. Same ASR runtime and weights as jz-meeting-skills:
mlx-qwen3-asr with moona3k/mlx-qwen3-asr-1.7b-8bit, read from the LM Studio model
directory when present (else the Hugging Face cache). Requires Apple Silicon (MLX).

Cached: if the output file already exists, transcription is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language zh
    python helpers/transcribe.py <video_path> --num-speakers 2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
import warnings
import os
from pathlib import Path

import numpy as np


ASR_REPO = "moona3k/mlx-qwen3-asr-1.7b-8bit"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
SAMPLE_RATE = 16000
# How far past the cursor a token may sit in the ASR text before we stop trusting the match.
PUNCT_SEARCH_WINDOW = 32

# --language takes ISO 639 codes. Only languages the forced aligner supports are
# accepted: without alignment there are no word timestamps to cut on.
ISO_TO_QWEN = {
    "zh": "Chinese", "yue": "Cantonese", "en": "English", "ja": "Japanese",
    "ko": "Korean", "fr": "French", "de": "German", "es": "Spanish",
    "pt": "Portuguese", "it": "Italian", "ru": "Russian",
}
QWEN_TO_ISO = {v: k for k, v in ISO_TO_QWEN.items()}

_session = None
_aligner = None


def asr_model() -> str:
    """VIDEO_USE_ASR_MODEL, else the LM Studio copy shared with jz-meeting-skills, else the repo id."""
    override = os.environ.get("VIDEO_USE_ASR_MODEL")
    if override:
        return override
    local = Path.home() / ".lmstudio/models" / ASR_REPO
    return str(local) if (local / "weights.safetensors").is_file() else ASR_REPO


def load_models():
    """Load once per process; batch mode reuses the same weights for every file."""
    global _session, _aligner
    if _session is None:
        from mlx_qwen3_asr import ForcedAligner, Session
        _session = Session(model=asr_model())
        _aligner = ForcedAligner(ALIGNER_MODEL)
    return _session, _aligner


def count_audio_tracks(video_path: Path) -> int:
    """How many audio streams the container holds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def extract_audio(video_path: Path, dest: Path, audio_track: int = 0) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", f"0:a:{audio_track}",
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_wav(path: Path) -> np.ndarray:
    """16-bit mono PCM wav -> float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _is_kept(ch: str) -> bool:
    """Characters the aligner keeps in a token (mirrors its tokenizer's clean_token)."""
    return ch == "'" or unicodedata.category(ch)[0] in "LN"


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2A6DF


def attach_punctuation(text: str, items) -> list[dict]:
    """The aligner strips punctuation from its tokens. Walk the ASR text and hand each
    token back the punctuation that trails it — phrase packing and subtitle chunking
    both break on it."""
    units: list[dict] = []
    cursor = 0
    for it in items:
        core = it["text"]
        suffix = ""
        pos = text.find(core, cursor)
        if 0 <= pos <= cursor + PUNCT_SEARCH_WINDOW:
            end = j = pos + len(core)
            while j < len(text) and not text[j].isspace() and not _is_kept(text[j]):
                j += 1
            suffix = text[end:j]
            cursor = j
        units.append({"core": core, "text": core + suffix,
                      "start": it["start"], "end": it["end"]})
    return units


def group_cjk(units: list[dict]) -> list[dict]:
    """The aligner times CJK text per character. Merge runs of characters into lexical
    words (jieba) so a word is something a cut or a subtitle chunk can stand on."""
    with warnings.catch_warnings():  # jieba's regexes predate Python 3.12's escape warnings
        warnings.simplefilter("ignore", SyntaxWarning)
        import jieba
    jieba.setLogLevel(logging.WARNING)

    words: list[dict] = []
    run: list[dict] = []

    def flush() -> None:
        i = 0
        for piece in jieba.lcut("".join(u["core"] for u in run)):
            span = run[i:i + len(piece)]
            i += len(piece)
            words.append({"text": "".join(u["text"] for u in span),
                          "start": span[0]["start"], "end": span[-1]["end"]})
        run.clear()

    for u in units:
        if len(u["core"]) == 1 and _is_cjk(u["core"]):
            run.append(u)
            if u["text"] != u["core"]:  # trailing punctuation ends the run
                flush()
        else:
            if run:
                flush()
            words.append({"text": u["text"], "start": u["start"], "end": u["end"]})
    if run:
        flush()
    return words


def transcribe_audio(audio: np.ndarray, language: str | None) -> tuple[list[dict], str | None, str]:
    """ASR + forced alignment. The library chunks at low-energy points (≤30s) and aligns
    each chunk, so segments come back per aligner token with absolute seconds.

    Returns (words, iso_language, full_text).
    """
    session, aligner = load_models()
    result = session.transcribe(audio, language=ISO_TO_QWEN[language] if language else None,
                                return_timestamps=True, forced_aligner=aligner)
    text = result.text.strip()
    if not text:
        return [], language, ""
    lang = result.language if isinstance(result.language, str) else None
    if language is None and lang not in QWEN_TO_ISO:
        raise RuntimeError(
            f"ASR detected language {lang!r}, which the forced aligner does not support - "
            f"pass --language with one of {', '.join(ISO_TO_QWEN)}"
        )
    words = group_cjk(attach_punctuation(text, result.segments or []))
    for w in words:
        w["start"] = round(float(w["start"]), 3)
        w["end"] = round(float(w["end"]), 3)
    return words, language or QWEN_TO_ISO[lang], text


def diarize(audio: np.ndarray, num_speakers: int) -> list[tuple[float, float, str]]:
    """pyannote speaker turns, non-overlapping, sorted by start."""
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    result = pipeline(
        {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE},
        num_speakers=num_speakers,
    )
    annotation = result.exclusive_speaker_diarization
    return sorted((turn.start, turn.end, label)
                  for turn, _, label in annotation.itertracks(yield_label=True))


def assign_speakers(words: list[dict], turns: list[tuple[float, float, str]]) -> None:
    """Give each word the turn it overlaps most; a word in a gap takes the nearest turn.
    Labels are renamed speaker_0, speaker_1, ... in order of first appearance."""
    names: dict[str, str] = {}
    i = 0
    for w in words:
        while i < len(turns) and turns[i][1] <= w["start"]:
            i += 1
        best, best_score = None, -math.inf
        for s, e, label in turns[max(i - 1, 0):]:
            score = min(e, w["end"]) - max(s, w["start"])  # overlap, or minus the gap
            if score > best_score:
                best, best_score = label, score
            if s >= w["end"]:
                break
        w["speaker_id"] = names.setdefault(best, f"speaker_{len(names)}")


def to_scribe_words(words: list[dict]) -> list[dict]:
    """Interleave 'spacing' entries: downstream reads silence from their start/end."""
    out: list[dict] = []
    prev = None
    for w in words:
        if prev is not None:
            out.append({"text": " ", "start": prev["end"], "end": max(prev["end"], w["start"]),
                        "type": "spacing", "speaker_id": w["speaker_id"]})
        out.append({"text": w["text"], "start": w["start"], "end": w["end"],
                    "type": "word", "speaker_id": w["speaker_id"]})
        prev = w
    return out


def transcript_path(edit_dir: Path, video: Path, audio_track: int = 0) -> Path:
    """Where a video's transcript lands.

    The track belongs in the name, or a rerun with --audio-track hands back the transcript of
    the track it is meant to replace. Track 0 keeps the plain name, so transcripts made before
    the flag existed stay valid. Batch mode tests its cache with this too — one function, so
    the two cannot drift apart.
    """
    suffix = "" if audio_track == 0 else f".track{audio_track}"
    return edit_dir / "transcripts" / f"{video.stem}{suffix}.json"


def transcribe_one(
    video: Path,
    edit_dir: Path,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
    audio_track: int = 0,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    MLX inference must stay on the main thread — callers run this sequentially.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcript_path(edit_dir, video, audio_track)

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    n_tracks = count_audio_tracks(video)
    if n_tracks > 1 and verbose:
        print(f"  note: {video.name} has {n_tracks} audio tracks, using track "
              f"{audio_track + 1} (--audio-track to change)", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, wav, audio_track)
        audio = read_wav(wav)

    # A silent track transcribes to nothing; catch the wrong-track case up front.
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    peak_db = 20 * math.log10(peak) if peak > 0 else float("-inf")
    if peak_db < -60.0:
        raise RuntimeError(
            f"track {audio_track + 1} of {video.name} is silent "
            f"(peak {peak_db:.1f} dBFS) - not transcribing. "
            + (f"The file has {n_tracks} audio tracks; try --audio-track "
               + " or ".join(str(i) for i in range(n_tracks) if i != audio_track) + "."
               if n_tracks > 1 else "Check the source audio.")
        )

    if verbose:
        print(f"  transcribing {video.stem} ({len(audio) / SAMPLE_RATE:.1f}s) on-device", flush=True)
    words, iso, text = transcribe_audio(audio, language)

    if num_speakers and num_speakers > 1 and words:
        if verbose:
            print(f"  diarizing ({num_speakers} speakers)", flush=True)
        assign_speakers(words, diarize(audio, num_speakers))
    else:
        for w in words:
            w["speaker_id"] = "speaker_0"

    payload = {"language_code": iso, "text": text, "words": to_scribe_words(words)}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        print(f"    words: {len(words)}  language: {iso}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video on-device (Qwen3-ASR + ForcedAligner)")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        choices=sorted(ISO_TO_QWEN),
        default=None,
        help="ISO language code. Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Number of speakers. 2+ runs pyannote diarization; omit for a single speaker.",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe. OBS writes the game on track 0 "
             "and the mic on track 1; without this ffmpeg applies its default audio "
             "stream selection, which picks the track with the most channels.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        language=args.language,
        num_speakers=args.num_speakers,
        audio_track=args.audio_track,
    )


if __name__ == "__main__":
    main()
