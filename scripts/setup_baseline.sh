#!/usr/bin/env bash
# One-time setup for the Phase 0 YOLO learning baseline (Roboflow's soccer example).
#   1. clone roboflow/sports into third_party/sports (gitignored) at the pinned commit
#   2. build the SEPARATE baseline environment in third_party/baseline-venv (gitignored)
#      from baseline/pyproject.toml + baseline/uv.lock (supervision <0.28, ultralytics)
#   3. run their setup.sh to download the 3 models + 5 sample videos (from Google Drive via gdown)
# The main project environment (.venv, for src/) is separate: create it with plain `uv sync`.
# Re-running is safe: existing clones/downloads are kept.
# Windows: run from Git Bash; on the AMD PC install AMD's PyTorch wheels afterwards (see CLAUDE.md).
set -euo pipefail
cd "$(dirname "$0")/.."

SPORTS_COMMIT=21a7d2b2ce77b047fac93527aed6dd08c69d85f3   # keep in sync with baseline/pyproject.toml
DEST=third_party/sports
BASELINE_VENV=third_party/baseline-venv   # relative on purpose: works in Git Bash on Windows too

if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/roboflow/sports.git "$DEST"
fi
git -C "$DEST" fetch --quiet --depth 1 origin "$SPORTS_COMMIT" 2>/dev/null || true
git -C "$DEST" checkout --quiet "$SPORTS_COMMIT"

# --locked: install exactly what baseline/uv.lock says (fails if the lock is out of date;
# after editing baseline/pyproject.toml run: uv lock --project baseline)
UV_PROJECT_ENVIRONMENT="$BASELINE_VENV" uv sync --project baseline --locked

DATA="$DEST/examples/soccer/data"
if [ -f "$DATA/football-player-detection.pt" ] && [ -f "$DATA/2e57b9_0.mp4" ]; then
  echo "Models and sample videos already downloaded in $DATA"
else
  # their setup.sh calls `gdown`, which is installed in the baseline environment
  PATH="$PWD/$BASELINE_VENV/bin:$PWD/$BASELINE_VENV/Scripts:$PATH" bash "$DEST/examples/soccer/setup.sh"
fi

git config core.hooksPath .githooks   # enable the commit guards (media/weights, ultralytics in src/)
echo "Baseline ready. Try: python3 scripts/run_baseline.py --help"
