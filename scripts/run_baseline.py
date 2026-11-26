"""Run Roboflow's soccer example (third_party/sports/examples/soccer/main.py) headless and time it.

Why a wrapper instead of running their main.py directly?
  * Their main() calls cv2.imshow() on every frame, which opens a preview window. That is
    slow and breaks when there is no screen (SSH, background jobs, the future GPU worker).
  * We want timing numbers (setup time vs. per-frame speed), which they don't print.
  * Their sink writes the 'mp4v' codec, which QuickTime often refuses to play. We re-encode
    to H.264 with ffmpeg so the output opens anywhere.
We import their code unchanged, so third_party/ stays a pristine copy of their repo.

Example:
    uv run python scripts/run_baseline.py --mode PLAYER_DETECTION \
        --source data/clips/2e57b9_0_20s.mp4 --device mps
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType

import supervision as sv
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "third_party" / "sports" / "examples" / "soccer"
OUTPUTS_DIR = REPO_ROOT / "outputs"
MODES = [
    "PITCH_DETECTION",
    "PLAYER_DETECTION",
    "BALL_DETECTION",
    "PLAYER_TRACKING",
    "TEAM_CLASSIFICATION",
    "RADAR",
]


def load_example() -> ModuleType:
    """Import their main.py as a module (it isn't installed as a package)."""
    path = EXAMPLE_DIR / "main.py"
    if not path.exists():
        sys.exit(f"Missing {path}. Run scripts/setup_baseline.sh first.")
    spec = importlib.util.spec_from_file_location("roboflow_soccer_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 so QuickTime/browsers can play it. Returns False if ffmpeg is missing."""
    if shutil.which("ffmpeg") is None:
        return False
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast", str(dst)],
        check=True,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="input video")
    parser.add_argument("--mode", choices=MODES, default="PLAYER_DETECTION")
    parser.add_argument("--device", default="mps", help="mps (Mac GPU), cuda (NVIDIA/AMD ROCm), or cpu")
    parser.add_argument("--target", type=Path, help="output video (default: outputs/<clip>-<mode>.mp4)")
    parser.add_argument("--max-frames", type=int, help="stop after N frames (quick smoke tests)")
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target or OUTPUTS_DIR / f"{source.stem}-{args.mode.lower().replace('_', '-')}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_target = target.with_suffix(".mp4v.mp4")  # temporary file in their codec

    example = load_example()
    run_mode = getattr(example, f"run_{args.mode.lower()}")
    video_info = sv.VideoInfo.from_video_path(str(source))
    total = min(video_info.total_frames, args.max_frames or video_info.total_frames)

    # Their run_* functions are *generators*: nothing happens until we ask for the first frame.
    # So "setup" = time until frame 1 comes out (model loading, plus for RADAR/TEAM_CLASSIFICATION
    # the crop-collection pass and fitting the team classifier). "loop" = every frame after that.
    t_start = time.perf_counter()
    t_first = None
    n = 0
    frames = run_mode(source_video_path=str(source), device=args.device)
    with sv.VideoSink(str(raw_target), video_info) as sink:
        for frame in tqdm(frames, total=total, desc=args.mode, unit="frame"):
            if t_first is None:
                t_first = time.perf_counter()
            sink.write_frame(frame)
            n += 1
            if n >= total:
                break
    t_end = time.perf_counter()

    encoded = to_h264(raw_target, target)
    if encoded:
        raw_target.unlink()
    else:
        raw_target.rename(target)

    total_s = t_end - t_start
    setup_s = (t_first or t_end) - t_start
    loop_fps = (n - 1) / (t_end - t_first) if n > 1 else float("nan")
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "device": args.device,
        "source": str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source),
        "output": str(target.relative_to(REPO_ROOT)) if target.is_relative_to(REPO_ROOT) else str(target),
        "frames": n,
        "video_seconds": round(n / video_info.fps, 2),
        "resolution": f"{video_info.width}x{video_info.height}",
        "total_seconds": round(total_s, 1),
        "setup_seconds": round(setup_s, 1),
        "overall_fps": round(n / total_s, 2),
        "loop_fps": round(loop_fps, 2),
        "slower_than_realtime_x": round(video_info.fps / (n / total_s), 1),
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    with open(OUTPUTS_DIR / "timings.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"\nSaved {target}")
    print(f"{n} frames ({record['video_seconds']} s of video) in {total_s:.1f} s "
          f"| setup {setup_s:.1f} s | loop {loop_fps:.2f} fps | overall {record['overall_fps']} fps "
          f"| {record['slower_than_realtime_x']}x slower than real time")


if __name__ == "__main__":
    main()
