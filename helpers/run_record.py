#!/usr/bin/env python3
"""Write a de-identified summary of one completed video editing task."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import subprocess
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

STORE = Path.home() / ".local/share/video-use/run-records"
FILLERS = {"um", "uh", "erm", "啊", "呃", "嗯", "那个", "就是"}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _probe(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,width,height,avg_frame_rate",
             "-of", "json", str(path)], capture_output=True, text=True, check=True,
        )
        metadata = json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    streams = metadata.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    duration = _number(metadata.get("format", {}).get("duration"))
    return {
        "duration_s": round(duration, 3) if duration is not None else None,
        "resolution": (f"{video['width']}x{video['height']}"
                       if video.get("width") and video.get("height") else None),
        "fps": video.get("avg_frame_rate") if video.get("avg_frame_rate") not in (None, "0/0") else None,
    }


def _transcripts(edit_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((edit_dir / "transcripts").glob("*.json")):
        data = _json(path)
        if isinstance(data, dict):
            records.append(data)
    return records


def _redact_note(note: str, transcripts: list[dict[str, Any]]) -> str:
    names: set[str] = set()
    for data in transcripts:
        for key in ("speaker_map", "speakers", "speaker_names"):
            value = data.get(key)
            if isinstance(value, dict):
                names.update(str(x) for x in value.values() if isinstance(x, str) and len(x) > 1)
            elif isinstance(value, list):
                names.update(str(x) for x in value if isinstance(x, str) and len(x) > 1)
    for name in sorted(names, key=len, reverse=True):
        note = re.sub(re.escape(name), "[REDACTED]", note, flags=re.IGNORECASE)
    note = re.sub(r"https?://\S+|www\.\S+", "[URL]", note, flags=re.IGNORECASE)
    note = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", note)
    note = re.sub(r"\b\+?\d[\d .()/-]{3,}\d\b", "[NUMBER]", note)
    return " ".join(note.split())


def _self_eval(edit_dir: Path) -> dict[str, Any]:
    passes = 0
    issues = 0
    issue_counts: Counter[str] = Counter()
    candidates = list(edit_dir.glob("*self*eval*.json")) + list(edit_dir.glob("self-eval*.json"))
    for path in set(candidates):
        data = _json(path)
        if not isinstance(data, (dict, list)):
            continue
        entries = data if isinstance(data, list) else data.get("passes", [data])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            passes += 1
            found = entry.get("issues", [])
            if isinstance(found, list):
                issues += len(found)
                for issue in found:
                    label = issue.get("type") if isinstance(issue, dict) else issue
                    if isinstance(label, str):
                        issue_counts[label] += 1
    return {"passes": passes, "issue_count": issues, "issues_by_type": dict(sorted(issue_counts.items()))}


def build_record(edit_dir: Path, note: str | None = None) -> dict[str, Any]:
    edit_dir = edit_dir.expanduser().resolve()
    today = dt.date.today()
    transcripts = _transcripts(edit_dir)
    edl = _json(edit_dir / "edl.json")
    edl = edl if isinstance(edl, dict) else {}
    sources = edl.get("sources", {})
    source_paths = list(sources.values()) if isinstance(sources, dict) else []
    source_stats = []
    for item in source_paths:
        if isinstance(item, str):
            source = Path(item)
            if not source.is_absolute():
                source = edit_dir / source
            stat = _probe(source)
            if stat:
                source_stats.append(stat)
    word_items = [w for data in transcripts for w in data.get("words", [])
                  if isinstance(w, dict) and w.get("type", "word") == "word"]
    speaker_ids = {w.get("speaker_id") for w in word_items if w.get("speaker_id")}
    ranges = edl.get("ranges", []) if isinstance(edl.get("ranges", []), list) else []
    kept = sum(max(0, (_number(r.get("end")) or 0) - (_number(r.get("start")) or 0))
               for r in ranges if isinstance(r, dict))
    pads = []
    for r in ranges:
        if not isinstance(r, dict):
            continue
        for key in ("padding_before", "padding_after", "pad_before", "pad_after"):
            val = _number(r.get(key))
            if val is not None:
                pads.append(val)
    if not pads:
        words_by_source = {}
        for path in sorted((edit_dir / "transcripts").glob("*.json")):
            data = _json(path)
            if isinstance(data, dict):
                words_by_source[path.stem] = data.get("words", [])
        for cut in ranges:
            if not isinstance(cut, dict):
                continue
            words = words_by_source.get(str(cut.get("source", "")), [])
            timed = [w for w in words if isinstance(w, dict) and w.get("type", "word") == "word"]
            start, end = _number(cut.get("start")), _number(cut.get("end"))
            inside = [w for w in timed if start is not None and end is not None
                      and (_number(w.get("start")) or 0) >= start
                      and (_number(w.get("end")) or 0) <= end]
            if inside and start is not None and end is not None:
                pads.extend((max(0.0, (_number(inside[0].get("start")) or start) - start),
                             max(0.0, end - (_number(inside[-1].get("end")) or end))))
    animations: Counter[str] = Counter()
    overlays = edl.get("overlays", [])
    for overlay in overlays if isinstance(overlays, list) else []:
        if isinstance(overlay, dict):
            animations[str(overlay.get("engine", "unknown"))] += 1
    renders = {}
    for name in ("final.mp4", "preview.mp4", "base.mp4"):
        stat = _probe(edit_dir / name)
        if stat:
            renders[name.removesuffix(".mp4")] = stat["duration_s"]
    final_info = _probe(edit_dir / "final.mp4")
    versions = list(edit_dir.glob("edl*.json"))
    timings = []
    for data in transcripts:
        value = data.get("asr_timing_s", data.get("transcription_duration_s"))
        if (num := _number(value)) is not None:
            timings.append(num)
    languages = sorted({str(d["language_code"]) for d in transcripts if d.get("language_code")})
    durations = [x["duration_s"] for x in source_stats if x["duration_s"] is not None]
    resolutions = sorted({x["resolution"] for x in source_stats if x["resolution"]})
    fps = sorted({x["fps"] for x in source_stats if x["fps"]})
    model = next((d.get("asr_model") for d in transcripts if d.get("asr_model")), None)
    if isinstance(model, str) and (model.startswith("/") or "\\" in model):
        model = "local"
    record: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "date": today.isoformat(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sources": {"count": len(source_paths) or len(transcripts),
                    "total_duration_s": round(sum(durations), 3),
                    "resolutions": resolutions, "fps": fps},
        "language": languages,
        "speakers_count": len(speaker_ids),
        "asr": {"model": model, "timings_s": timings or None},
        "words_count": len(word_items),
        "filler_count": sum(1 for w in word_items if str(w.get("text", "")).strip().lower() in FILLERS),
        "edl": {"ranges_count": len(ranges), "kept_duration_s": round(kept, 3),
                "cut_count": max(0, len(ranges) - 1),
                "mean_cut_padding_s": round(sum(pads) / len(pads), 3) if pads else None},
        "grade_preset": edl.get("grade"),
        "subtitles": bool(edl.get("subtitles")),
        "animations_by_engine": dict(sorted(animations.items())),
        "self_eval": _self_eval(edit_dir),
        "render_durations_s": renders,
        "final_duration_s": final_info["duration_s"] if final_info else None,
        "iterations": len(versions) if len(versions) > 1 else None,
    }
    if note:
        record["friction_note"] = _redact_note(note, transcripts)
    return record


def _state() -> dict[str, Any]:
    state = _json(STORE / "state.json")
    if not isinstance(state, dict):
        state = {}
    state.setdefault("last_optimization_at", None)
    state.setdefault("threshold", 5)
    return state


def write_record(edit_dir: Path, note: str | None = None) -> dict[str, Any]:
    record = build_record(edit_dir, note)
    month_dir = STORE / record["date"][:7]
    month_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    (edit_dir.expanduser().resolve() / "run-record.json").write_text(payload, encoding="utf-8")
    (month_dir / f"{record['id']}.json").write_text(payload, encoding="utf-8")
    state = _state()
    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    since = state.get("last_optimization_at")
    records = list(STORE.glob("????-??/*.json"))
    recent = []
    for path in records:
        data = _json(path)
        if not isinstance(data, dict):
            continue
        created = data.get("created_at")
        if not since or (created and created > since) or (not created and data.get("date", "") > str(since)[:10]):
            recent.append(data)
    threshold = int(state.get("threshold", 5))
    if len(recent) >= threshold:
        since_date = str(since)[:10] if since else min(str(item["date"]) for item in recent)
        print(f"OPTIMIZATION_DUE: {len(recent)} runs since {since_date}; run helpers/run_record.py --summarize --since {since_date}")
    return record


def summarize(since: str | None) -> dict[str, Any]:
    runs = []
    for path in sorted(STORE.glob("????-??/*.json")):
        item = _json(path)
        if isinstance(item, dict) and item.get("date") and (not since or item["date"] >= since):
            runs.append(item)
    numeric = ("sources.count", "sources.total_duration_s", "words_count", "filler_count",
               "edl.kept_duration_s", "edl.cut_count", "final_duration_s")
    stats = {}
    for dotted in numeric:
        vals = []
        for item in runs:
            value: Any = item
            for part in dotted.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if (n := _number(value)) is not None:
                vals.append(n)
        if vals:
            stats[dotted] = {"count": len(vals), "mean": round(sum(vals) / len(vals), 3)}
    common: Counter[str] = Counter()
    for item in runs:
        ev = item.get("self_eval", {})
        if isinstance(ev, dict) and isinstance(ev.get("issues_by_type"), dict):
            common.update({str(k): int(v) for k, v in ev["issues_by_type"].items() if isinstance(v, int)})
    return {"runs_count": len(runs), "since": since, "means": stats,
            "common_issues": dict(common.most_common())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edit_dir", nargs="?", type=Path)
    parser.add_argument("--note")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--since")
    parser.add_argument("--mark-optimized", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.since), ensure_ascii=False, indent=2))
    elif args.mark_optimized:
        state = _state()
        state["last_optimization_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        if args.note:
            state["last_optimization_note"] = _redact_note(args.note, [])
        STORE.mkdir(parents=True, exist_ok=True)
        (STORE / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(state, ensure_ascii=False, indent=2))
    elif args.edit_dir:
        print(json.dumps(write_record(args.edit_dir, args.note), ensure_ascii=False, indent=2))
    else:
        parser.error("edit_dir is required unless --summarize or --mark-optimized is used")


if __name__ == "__main__":
    main()
