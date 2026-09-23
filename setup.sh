#!/usr/bin/env bash
# One-shot setup for this fork (on-device ASR). Idempotent: safe to re-run.
#   git clone https://github.com/Loveacup/video-use ~/Developer/video-use && ~/Developer/video-use/setup.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
UPSTREAM="https://github.com/browser-use/video-use"
step() { printf '\n==> %s\n' "$*"; }

step "Platform"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo "needs an Apple Silicon Mac (MLX)"; exit 1; }
command -v brew >/dev/null || { echo "install Homebrew first: https://brew.sh"; exit 1; }

step "uv"
command -v uv >/dev/null || brew install uv

step "Python deps"
(cd "$REPO" && uv sync)

step "ffmpeg with libass (subtitle burning)"
has_subtitles() { ffmpeg -hide_banner -filters 2>&1 | grep " subtitles " >/dev/null; }  # not grep -q: early exit SIGPIPEs ffmpeg under pipefail
if ! has_subtitles; then
  brew install ffmpeg-full
  brew unlink ffmpeg 2>/dev/null || true
  brew link --force --overwrite ffmpeg-full
  hash -r
fi
has_subtitles || { echo "ffmpeg still lacks the subtitles filter"; exit 1; }

step "Register skill"
for dir in "$HOME/.claude/skills" "${CODEX_HOME:-$HOME/.codex}/skills"; do
  [[ "$dir" == "$HOME/.claude/skills" || -d "$(dirname "$dir")" ]] || continue
  mkdir -p "$dir" && ln -sfn "$REPO" "$dir/video-use" && echo "  $dir/video-use -> $REPO"
done

step "Upstream remote"
git -C "$REPO" remote get-url upstream >/dev/null 2>&1 || git -C "$REPO" remote add upstream "$UPSTREAM"
git -C "$REPO" config rerere.enabled true

step "Prefetch models (~3 GB first time)"
"$REPO/.venv/bin/python" -c "from mlx_audio.stt.utils import load_model; [load_model(m) for m in ('mlx-community/Qwen3-ASR-1.7B-8bit', 'mlx-community/Qwen3-ForcedAligner-0.6B-8bit')]"

step "Self-check: transcribe + subtitle render on a synthetic clip"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
# Chinese voices are optional downloads on macOS; fall back to any English voice.
voice="$(say -v '?' | awk '$2 ~ /^zh_/ {print $1; exit}')"
if [[ -n "$voice" ]]; then lang=zh; line="大家好，今天我们来测试一下视频剪辑。"
else voice="$(say -v '?' | awk '$2 == "en_US" {print $1; exit}')"; lang=en; line="Hello, today we are testing video editing."; fi
say ${voice:+-v "$voice"} -o "$tmp/a.aiff" "$line"
ffmpeg -loglevel error -f lavfi -i color=c=black:s=640x360:r=30 -i "$tmp/a.aiff" -shortest \
  -c:v libx264 -pix_fmt yuv420p -c:a aac "$tmp/take.mp4"
PY="$REPO/.venv/bin/python"
"$PY" "$REPO/helpers/transcribe.py" "$tmp/take.mp4" --language "$lang" 2>&1 | grep -v Fetching
"$PY" "$REPO/helpers/pack_transcripts.py" --edit-dir "$tmp/edit" >/dev/null
end="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['words'][-1]['end'])" "$tmp/edit/transcripts/take.json")"
printf '{"version":1,"sources":{"take":"%s"},"ranges":[{"source":"take","start":0,"end":%s}]}' "$tmp/take.mp4" "$end" > "$tmp/edit/edl.json"
"$PY" "$REPO/helpers/render.py" "$tmp/edit/edl.json" -o "$tmp/edit/final.mp4" --build-subtitles >/dev/null
sed -n '6,7p' "$tmp/edit/takes_packed.md"
[[ -s "$tmp/edit/final.mp4" ]] && echo "  render with subtitles OK"

step "Done. Start a new agent session; upgrade with:"
echo "  cd $REPO && git pull --rebase upstream main && uv sync && git push --force-with-lease origin local-asr"
