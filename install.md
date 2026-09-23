---
name: video-use-install
description: Install video-use into the current agent (Claude Code, Codex, Hermes, Openclaw, etc.) and wire up ffmpeg + on-device transcription so the user can start editing immediately.
---

# video-use install

Use this file only for first-time install or reconnect. For daily editing, read `SKILL.md`. Always read `helpers/` — that's where the scripts live.

## What you're doing

You're setting up a conversation-driven video editor for the user. After install, the user drops raw footage into any folder, runs their agent (`claude`, `codex`, etc.) there, and says "edit these into a launch video." You do the rest by reading `SKILL.md`.

Three things must exist on this machine:

1. The `video-use` repo cloned somewhere stable.
2. `ffmpeg` on `$PATH`, built with libass for subtitle burning (plus optional `yt-dlp` for online sources).
3. The repo's `.venv` with the on-device ASR stack (Qwen3-ASR + Qwen3-ForcedAligner on MLX — Apple Silicon only).

And one thing must be true about the current agent:

4. It can discover `SKILL.md` — either via a global skills directory (`~/.claude/skills/`, `~/.codex/skills/`) or via a `CLAUDE.md` / system-prompt import.

## Install prompt contract

- Do everything yourself. Only ask the user for confirmation before `brew install`.
- Prefer a stable clone path like `~/Developer/video-use` (not `/tmp`, not `~/Downloads`).
- The skill references helpers by bare name (`transcribe.py`, `render.py`). That works because SKILL.md and `helpers/` ship together — keep them as siblings when you register the skill.
- After install, verify by running one real command against one real file. Don't declare success on file-existence checks alone.

## Steps

### 1. Clone to a stable path

```bash
test -d ~/Developer/video-use || git clone https://github.com/browser-use/video-use ~/Developer/video-use
cd ~/Developer/video-use
```

If the repo is already there, `git pull --ff-only` and continue.

### 2. Install Python deps

```bash
uv sync
```

`pyproject.toml` pulls `mlx-audio` (ASR + forced alignment), `jieba` (CJK word grouping), `pyannote.audio` (diarization), `librosa`, `matplotlib`, `pillow`, `numpy`. No console scripts — helpers are invoked as `.venv/bin/python helpers/<name>.py`.

### 3. Install ffmpeg (+ optional yt-dlp)

`ffmpeg` and `ffprobe` are hard requirements. `yt-dlp` is only needed if the user wants to pull sources from URLs. Animation engines such as HyperFrames, Remotion, and Manim are installed lazily the first time a project actually needs them.

```bash
# macOS — Homebrew's plain `ffmpeg` has no libass, so subtitle burning fails. Use ffmpeg-full:
ffmpeg -hide_banner -filters 2>&1 | grep -q " subtitles " || { brew install ffmpeg-full && brew unlink ffmpeg 2>/dev/null; brew link --force --overwrite ffmpeg-full; }
command -v yt-dlp >/dev/null || brew install yt-dlp     # optional

# Debian / Ubuntu
# sudo apt-get update && sudo apt-get install -y ffmpeg
# pip install yt-dlp

# Arch
# sudo pacman -S ffmpeg yt-dlp
```

If `brew` / `apt` / `pacman` requires a sudo prompt, tell the user the exact command and wait. Do not invent a password.

### 4. Register the skill with the current agent

Figure out which agent you are running under, and register once. A symlink of the whole repo directory is the right shape — helpers/ needs to sit next to SKILL.md.

- **Claude Code** (`~/.claude/` present):

    ```bash
    mkdir -p ~/.claude/skills
    ln -sfn ~/Developer/video-use ~/.claude/skills/video-use
    ```

- **Codex** (`$CODEX_HOME` set, or `~/.codex/` present):

    ```bash
    mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
    ln -sfn ~/Developer/video-use "${CODEX_HOME:-$HOME/.codex}/skills/video-use"
    ```

- **Hermes / Openclaw / another agent with a skills directory**: symlink `~/Developer/video-use` into that agent's skills directory under the name `video-use`. If the agent has no skills directory, add a line to its system prompt / config pointing at `~/Developer/video-use/SKILL.md` (e.g. an `@~/Developer/video-use/SKILL.md` import in a `CLAUDE.md`-equivalent).

If you can't tell which agent you're in, ask the user once: "which agent am I running under — Claude Code, Codex, or something else?" Then pick the right target.

### 5. Prefetch the transcription models

Transcription runs entirely on-device; nothing is uploaded and there is no API key. The first run downloads the models (~3 GB) into the Hugging Face cache. Fetch them now so the user's first clip doesn't stall:

```bash
~/Developer/video-use/.venv/bin/python -c "from mlx_audio.stt.utils import load_model; [load_model(m) for m in ('mlx-community/Qwen3-ASR-1.7B-8bit', 'mlx-community/Qwen3-ForcedAligner-0.6B-8bit')]"
```

Diarization (`--num-speakers 2+`) additionally loads `pyannote/speaker-diarization-community-1` on first use.

### 6. Verify end-to-end

Run one real thing. Prefer the lightest verification that still proves the pipeline is wired up:

```bash
~/Developer/video-use/.venv/bin/python ~/Developer/video-use/helpers/timeline_view.py --help >/dev/null && echo "helpers OK"
ffmpeg -hide_banner -filters | grep -q " subtitles " && echo "libass OK"
ffprobe -version | head -1
```

Full transcription test is optional at install time. Better to wait until the user hands you their first clip.

### 7. Hand off

Tell the user, in one short message:

- Where the skill is installed (`~/Developer/video-use`).
- That they should `cd` into their footage folder and start their agent there (e.g. `claude`).
- That a good first message is: *"edit these into a launch video"* or *"inventory these takes and propose a strategy."*
- That all outputs land in `<videos_dir>/edit/` — the repo stays clean.

## Keeping the skill current

- `cd ~/Developer/video-use && git pull --rebase upstream main && uv sync && git push --force-with-lease origin local-asr` pulls upstream, replays the local commits, and publishes to the fork (`origin` = Loveacup/video-use, `upstream` = browser-use/video-use; `setup.sh` adds `upstream` and enables `rerere`). The symlink auto-picks it up on the next run.
- This install carries local commits (on-device ASR on branch `local-asr`), so `--ff-only` would fail. If upstream touched `helpers/transcribe.py`, the rebase conflicts there — keep the local backend and port any upstream fixes into it.

## Cold-start reminders

- Symlink the **whole directory**, not just `SKILL.md`. The helpers need to sit next to it.
- Run helpers with `~/Developer/video-use/.venv/bin/python`, not the system `python3` — only the venv has the ASR stack.
- `ffmpeg` from static builds works fine. Any modern (≥ 4.x) build is enough.
- `yt-dlp` is optional. Don't block install on it; install lazily the first time a user asks to pull from a URL.
- Node.js/npm are only needed for HyperFrames or Remotion slots. HyperFrames currently requires Node.js 22+.
- HyperFrames, Remotion, and Manim are optional animation engines. Don't install or prefer one globally during setup; pick the engine per animation slot in `SKILL.md`. HyperFrames can run through `npx --yes hyperframes ...` in the slot directory. Remotion can be scaffolded with `npx create-video@latest` or installed inside the slot before rendering.
- Transcription is local but GPU-heavy; don't run it as part of install verification unless the user explicitly asks.
- If the user is on Linux without a package manager Claude recognizes, print the manual `ffmpeg` install URL and wait rather than guessing.
