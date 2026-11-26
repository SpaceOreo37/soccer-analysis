#!/usr/bin/env bash
# One-time setup for the Phase 0 baseline (Roboflow's soccer example).
#   1. clone roboflow/sports into third_party/ (gitignored) at the same commit pyproject.toml pins
#   2. create/refresh the .venv from uv.lock
#   3. run their setup.sh to download the 3 models + 5 sample videos (from Google Drive via gdown)
# Re-running is safe: existing clones/downloads are kept.
# Windows: run from Git Bash; on the AMD PC install AMD's PyTorch wheels afterwards (see CLAUDE.md).
set -euo pipefail
cd "$(dirname "$0")/.."

SPORTS_COMMIT=21a7d2b2ce77b047fac93527aed6dd08c69d85f3   # keep in sync with pyproject.toml
DEST=third_party/sports

if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/roboflow/sports.git "$DEST"
fi
git -C "$DEST" fetch --quiet --depth 1 origin "$SPORTS_COMMIT" 2>/dev/null || true
git -C "$DEST" checkout --quiet "$SPORTS_COMMIT"

uv sync

DATA="$DEST/examples/soccer/data"
if [ -f "$DATA/football-player-detection.pt" ] && [ -f "$DATA/2e57b9_0.mp4" ]; then
  echo "Models and sample videos already downloaded in $DATA"
else
  PATH="$PWD/.venv/bin:$PWD/.venv/Scripts:$PATH" bash "$DEST/examples/soccer/setup.sh"
fi

git config core.hooksPath .githooks   # enable the commit guard (blocks videos/weights)
echo "Baseline ready. Try: uv run python scripts/run_baseline.py --help"
