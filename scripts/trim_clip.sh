#!/usr/bin/env bash
# Cut a short clip out of a longer video with ffmpeg.
#
#   scripts/trim_clip.sh INPUT START DURATION OUTPUT
#   scripts/trim_clip.sh match.mp4 00:12:30 20 data/clips/match_20s.mp4
#
# We re-encode (libx264) instead of using "-c copy": stream copy can only cut at keyframes,
# so the clip could start a few seconds early or late. Re-encoding cuts on the exact frame.
# -an drops audio (the pipeline doesn't use it). crf 18 = visually near-lossless.
set -euo pipefail
if [ $# -ne 4 ]; then
  echo "usage: $0 INPUT START DURATION OUTPUT" >&2; exit 1
fi
mkdir -p "$(dirname "$4")"
ffmpeg -y -loglevel error -ss "$2" -i "$1" -t "$3" -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p -an "$4"
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,r_frame_rate,nb_read_frames -of default=nw=1 "$4"
