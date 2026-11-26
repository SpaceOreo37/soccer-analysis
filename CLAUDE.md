# CLAUDE.md — soccer-analysis

## What this project is
An open-source, AI-powered soccer match analysis web app for **amateur, youth, and high school
teams**. Coaches upload match video, mainly **Veo follow-cam MP4 exports** (a single camera that
pans and zooms like a TV broadcast, so the visible part of the pitch changes every frame), and
get back:
- match stats,
- a 2D radar/minimap of player positions,
- eventually per-player coaching tips, each linked to the video clips that support it.

Long-term stack: Python computer-vision pipeline → job queue → worker on the owner's own GPU
machine → Next.js frontend on Vercel.

Repo: https://github.com/SpaceOreo37/soccer-analysis (branch `main`).

**Current phase: Phase 0 (setup + working baseline), complete.** See the roadmap below and
`docs/devlog.md` for history.

## How to work with the owner
- The owner is a high school student who wants to **learn**, not just get results. Briefly
  explain what each step does and why, especially environments, git, and how the CV pipeline
  works.
- If something fails, troubleshoot it yourself first. If truly stuck, explain the problem in
  plain language and offer options.
- **No license has been chosen.** Don't add a LICENSE file or pick one. The research and
  options are in `docs/licensing.md`; the owner decides.
- Add a dated entry to `docs/devlog.md` for each work session: what was done, what worked,
  what broke, and numbers.

## Machines
| Machine | Role | PyTorch device | Notes |
|---|---|---|---|
| MacBook Air M2, 24 GB, **fanless** | Development | `mps` | Only process **short clips** (≤30 s). It slows down when it gets hot (thermal throttling). |
| Windows PC, AMD RX 7900 XTX | Full matches, future GPU worker | `cuda` with AMD's ROCm build (verify with `torch.cuda.is_available()`) | Uses AMD's PyTorch for Windows, which needs **Python 3.12**. Not set up yet (Phase 1). |

Python is pinned to **3.12** (`.python-version`, `requires-python`) on both machines so they match.

## Environment
- Managed by **uv**. `pyproject.toml` declares dependencies, and `uv.lock` pins exact versions.
  Commit both.
- The virtual environment lives in `.venv/` (gitignored). Create or refresh it with `uv sync`.
- Activate it: `source .venv/bin/activate` (macOS) or `.venv\Scripts\activate` (Windows). Or skip
  activating and prefix commands with `uv run`, e.g. `uv run python scripts/run_baseline.py ...`.
- Add a package: `uv add <pkg>` (or `uv add --group baseline <pkg>`). Don't pip-install into
  `.venv` by hand.
- Dependency groups: the main `dependencies` list is for our own code (empty so far). The
  **`baseline`** group holds Roboflow's example stack (`sports`, `ultralytics`, `gdown`,
  pinned `supervision`). It's installed by default (`[tool.uv] default-groups`). Ultralytics is
  **AGPL-3.0**, so don't import it from `src/` until the owner makes the license decision.
- Windows/AMD caveat: `uv.lock` resolves `torch` from PyPI. On the AMD PC, AMD's ROCm PyTorch
  wheels must replace it (probably via a platform-specific uv index/source). Work this out in
  Phase 1.

## Layout
```
src/soccer_analysis/   our own pipeline package (Phase 1+; empty so far)
scripts/               runnable entry points
  setup_baseline.sh    clone roboflow/sports (pinned) into third_party/, uv sync, download models+videos
  run_baseline.py      run their soccer example headless, time it, save H.264 video + timings
  trim_clip.sh         cut a short clip with ffmpeg (frame-accurate re-encode)
docs/                  devlog.md, licensing.md
data/                  local inputs (gitignored), e.g. data/clips/2e57b9_0_20s.mp4
outputs/               pipeline outputs (gitignored): videos + timings.jsonl
third_party/sports/    roboflow/sports clone (gitignored, never committed, never edited)
.githooks/pre-commit   refuses commits with videos, weights, datasets or files >20 MB
```

## Hard rules: never commit media or models
- Videos (`*.mp4`, `*.mov`, …), model weights (`*.pt`, `*.onnx`, `*.safetensors`, `*.engine`, …),
  datasets/archives, `data/`, `outputs/` and `third_party/` are all in `.gitignore`.
- A pre-commit hook in `.githooks/` blocks them even with `git add -f`. Each fresh clone must
  enable it once: `git config core.hooksPath .githooks` (`setup_baseline.sh` does this).
- Never copy Roboflow's repo, models, or sample videos into tracked files.

## Running the Phase 0 baseline
```bash
scripts/setup_baseline.sh                                  # once per machine
scripts/trim_clip.sh third_party/sports/examples/soccer/data/2e57b9_0.mp4 0 20 data/clips/2e57b9_0_20s.mp4
uv run python scripts/run_baseline.py --mode PLAYER_DETECTION --device mps --source data/clips/2e57b9_0_20s.mp4
uv run python scripts/run_baseline.py --mode RADAR            --device mps --source data/clips/2e57b9_0_20s.mp4
```
- Modes: `PITCH_DETECTION`, `PLAYER_DETECTION`, `BALL_DETECTION`, `PLAYER_TRACKING`,
  `TEAM_CLASSIFICATION`, `RADAR`. Use `--max-frames N` for quick smoke tests.
- Output goes to `outputs/<clip>-<mode>.mp4`, and a timing record is appended to
  `outputs/timings.jsonl`.
- Use `PYTHONWARNINGS=ignore` to silence supervision's harmless `FutureWarning` spam.
- Why a wrapper instead of their `main.py`: their `main()` calls `cv2.imshow` on every frame
  (opens a preview window, can hang with no screen), prints no timings, and writes the `mp4v`
  codec, which QuickTime won't play. The wrapper imports their `run_*` generators unchanged.

### How the baseline pipeline works (Roboflow `sports` soccer example)
1. **Player detection**: a YOLOv8 model (`football-player-detection.pt`, input 1280 px)
   finds ball/goalkeeper/player/referee boxes in each frame.
2. **Tracking**: ByteTrack links boxes across frames into IDs
   (`minimum_consecutive_frames=3`).
3. **Team classification**: crops of players from every 60th frame → SigLIP image embeddings
   → UMAP (reduce to 3-D) → KMeans (2 clusters = 2 teams). Goalkeepers join the team whose
   centroid is nearest.
4. **Pitch keypoints**: a YOLOv8-pose model finds 32 known pitch landmarks (corners, box
   corners, centre circle…).
5. **Homography** (`ViewTransformer`, `cv2.findHomography`): the visible keypoints are matched
   to their real pitch positions, giving a transform from image pixels to pitch coordinates.
   Each player's foot point (box bottom-centre) is mapped onto a 2D pitch drawing = **radar**.
   This is computed per frame, which is what Phase 3 needs for Veo follow-cam footage.
6. **Ball detection** (separate mode): a separate YOLO model on 640×640 tiles
   (`InferenceSlicer`), because the ball is tiny.

### Known gotchas
- **supervision is pinned `>=0.26,<0.28`.** In 0.28+, `Detections.merge()` of all-empty
  detections returns `tracker_id=None`, which crashes `RADAR`/`TEAM_CLASSIFICATION` on frame 1
  (ByteTrack confirms nothing until frame 3). In 0.31+, `sv.ByteTrack` is removed entirely
  (moved to Roboflow's `trackers` package).
- `sports` is pinned to commit `21a7d2b` in both `pyproject.toml` and `scripts/setup_baseline.sh`.
  Keep them in sync.
- Ultralytics respects `.to('mps')` on the model, so `--device mps` does reach the GPU.
- The first `RADAR`/`TEAM_CLASSIFICATION` run downloads SigLIP from Hugging Face (~800 MB,
  cached in `~/.cache/huggingface`).

### Baseline speed (M2 Air, MPS, 1080p, 20 s / 500-frame clip)
See `docs/devlog.md` (2026-11-25) for details. **Sustained** (500-frame runs):
player detection **1.69 fps**, radar **0.53 fps**. **Burst** (first ~15–20 s, cool chip):
≈4 fps and ≈1.4 fps. The fanless M2 thermal-throttles after ~15–20 s of GPU load, so
**short smoke tests overstate speed by 2–3×**. Benchmark with ≥300 frames. A 90-minute match
(135k frames) would take ~22 h (detection) or ~70 h (radar) on the Mac, so full matches
belong on the Windows/AMD PC.
- Ultralytics' `NMS time limit … exceeded` warnings during throttling are harmless at batch
  size 1. The timer includes the queued GPU work (MPS is asynchronous), and the frame's
  results are kept before the limit is checked.

## Roadmap (don't start a phase until the owner says so)
- **Phase 0 — Setup + baseline (done 2026-11-25).** uv env, Roboflow soccer example running on
  MPS, timing, licensing research, project structure.
- **Phase 1:** Test the pipeline on **SoccerTrack v2** (amateur, full-pitch, labeled data on
  Hugging Face). Write an **evaluation script** that compares our output to the ground-truth
  labels. Set up the **Windows/AMD PC** and time a full match there.
- **Phase 2:** Restructure the pipeline into **stages that each save their output to a file**
  (detections → tracks → pitch coordinates → stats). Add team classification, a possession
  estimate, and heatmaps.
- **Phase 3:** Handle **Veo follow-cam** footage by mapping each frame onto the pitch
  separately. Build a tool where the coach **confirms player identities** (the system guesses
  jersey numbers, the coach corrects them).
- **Phase 4:** Detect **events** (passes, turnovers, shots), find patterns for each player, and
  generate **LLM coaching tips**. **Every tip must link to the clips that support it.**
- **Phase 5:** **Web app**: uploads, a job queue, a GPU worker, and a results page (Next.js on
  Vercel).
- **Phase 6:** **Pilot** with 1–2 real coaches.
