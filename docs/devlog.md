# Devlog

## 2026-11-25: Phase 0: setup and baseline pipeline

### What we set up
- **Tools:** git 2.50, Homebrew 7.0, ffmpeg 9.0 and `gh` were already installed (and `gh` was
  already logged in). Installed **uv 0.13** with Homebrew. The system Python is Apple's 3.9,
  which we leave alone. uv downloaded its own **Python 3.12.13**.
- **Repo:** cloned `SpaceOreo37/soccer-analysis` into the (empty) project folder, keeping the
  existing README and Python `.gitignore`.
- **Environment:** `pyproject.toml` + `uv.lock`, a `.venv` built with `uv sync`. PyTorch 2.15
  with MPS confirmed working. The baseline stack (`sports` pinned to commit `21a7d2b`,
  `ultralytics` 8.4, `gdown`, `supervision` 0.27) lives in a separate `baseline` dependency
  group because Ultralytics is AGPL-3.0.
- **Roboflow `sports`:** cloned into `third_party/sports/` (gitignored). Their `setup.sh`
  downloaded 3 models (~135 MB each) and 5 sample Bundesliga clips (30 s, 1080p, 25 fps each).
  SigLIP (~775 MB) downloads from Hugging Face on first radar run.
- **Our scripts:** `scripts/run_baseline.py` (headless wrapper + timing + H.264 output),
  `scripts/trim_clip.sh` (ffmpeg trim), `scripts/setup_baseline.sh` (one-command setup).
- **Protection against committing media/models:** extended `.gitignore`, plus a pre-commit hook
  (`.githooks/pre-commit`) that refuses videos, weights, datasets and anything >20 MB, even
  with `git add -f`. Tested with a fake `test.pt`, and it was blocked.
- **Docs:** `CLAUDE.md`, `docs/licensing.md`, this devlog.

### What worked
- All six modes of the example run on `--device mps` (pitch, player, ball, tracking, team
  classification, radar).
- Ultralytics really does use the GPU: `.to('mps')` is respected (checked `predictor.device`).
- Output videos re-encoded to H.264 so QuickTime plays them.

### What broke (and fixes)
1. **`AttributeError: module 'supervision' has no attribute 'ByteTrack'`.** uv installed the
   newest supervision (0.32.2). ByteTrack was removed in 0.31 (it moved to Roboflow's
   separate `trackers` package).
2. **After pinning `<0.31`: `TypeError: 'NoneType' object is not iterable` on
   `detections.tracker_id`.** ByteTrack needs 3 consecutive frames before it confirms a track,
   so frames 0–1 have no tracks. From supervision **0.28** on, `Detections.merge()` of
   all-empty detections returns `tracker_id=None` (up to 0.27 it was an empty array). Found
   the exact version by running a tiny test script against 0.23 → 0.30.9 with
   `uv run --with supervision==X`.
   **Fix:** pin `supervision>=0.26,<0.28` (installed 0.27.0.post2). No changes to their code.
3. **Their `main.py` calls `cv2.imshow` every frame.** Avoided by importing their `run_*`
   generators from our wrapper instead.
4. **Speed mystery:** the full runs were 2–3× slower than 15-frame smoke tests. Diagnosed as
   **thermal throttling** of the fanless M2. Speed is ~4.1 fps for the first ~15–20 s, then
   drops to ~0.5 and settles around 1.4–1.8 fps. GPU memory, process memory and swap stayed
   flat (no leak, 47% RAM free). Starting at frame 100 instead of 0 still showed the fast
   burst then a drop, so it's time-under-load, not video content. Ultralytics prints
   `NMS time limit 2.050s exceeded` while throttled. That's harmless at batch size 1: the
   timer includes the asynchronous GPU work, and results are saved before the check.
   (Temperature itself wasn't read; that needs `sudo powermetrics`.)

### Timing numbers
Clip: `data/clips/2e57b9_0_20s.mp4`, the first 20 s of their `2e57b9_0.mp4`
(1920×1080, 25 fps, **500 frames**). MacBook Air M2, MPS, Python 3.12.13, torch 2.15.0.
Runs were back-to-back (player detection first).

| Mode | Total time | Setup | Sustained fps | Slower than real time |
|---|---|---|---|---|
| PLAYER_DETECTION | 297.4 s (~5 min) | 1.3 s | **1.69** | 14.9× |
| RADAR | 955.6 s (~16 min) | 21.2 s | **0.53** | 47.8× |

"Setup" = model loading. For radar it also includes collecting player crops from every 60th
frame and fitting the team classifier (SigLIP → UMAP → KMeans).

Burst speeds (15-frame smoke tests, cool chip, loop fps): pitch 11.1, player detection 3.9,
tracking 3.3, radar 1.4, ball 1.0, team classification 0.45.

**What this means:** a 90-min match = 135,000 frames, which would be ~22 h for detection
alone or ~70 h for radar on this Mac. The Mac is for short clips only. Full matches go to
the RX 7900 XTX.

Speed-up ideas for later (not tried): run the detector on every 2nd–3rd frame and let the
tracker fill gaps, a smaller input than 1280 px, half precision, CoreML/ONNX export, and
running the pitch model less often when the camera is still.

### Outputs
- `outputs/2e57b9_0_20s-player-detection.mp4`
- `outputs/2e57b9_0_20s-radar.mp4`
- `outputs/timings.jsonl` (machine-readable timing records)

### Open decisions for the owner
- **Project license** (see `docs/licensing.md`). The real question is whether to keep
  Ultralytics YOLO (AGPL-3.0) or plan to replace it with Apache-licensed RF-DETR + `trackers`.
