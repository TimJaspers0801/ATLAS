#!/usr/bin/env python3
"""
Generate per-video surgical frame index files from E:/atlas120k_full_videos.

The full_videos dataset contains all frames at 15 fps with non-surgical frames
manually removed. The resulting gaps encode which frames should be skipped
during processing:
  - Frames before start_frame  → non-surgical prefix
  - Frames in excluded_ranges  → non-surgical mid-video segments
  - Frames after end_frame     → non-surgical suffix

One JSON is written per video to:
  <output_dir>/<procedure>/<youtube_id>.json

These files are stored in the repository under data/surgical_index/ and used
by process_atlas120k.py to filter frames when extracting the dataset.

Usage (run once by the dataset authors):
  python download/generate_surgical_index.py \\
      --full_videos_dir E:/atlas120k_full_videos \\
      --output_dir      data/surgical_index
"""

import argparse
import json
import re
from pathlib import Path

FRAME_RE = re.compile(r"^frame_(\d+)\.jpg$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate surgical frame index files from atlas120k_full_videos."
    )
    parser.add_argument(
        "--full_videos_dir",
        required=True,
        help="Root of atlas120k_full_videos (<procedure>/<youtube_id>/frame_*.jpg)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/surgical_index",
        help="Output directory for per-video JSON index files (default: data/surgical_index)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing index files",
    )
    return parser.parse_args()


def build_index(proc_name: str, youtube_id: str, video_dir: Path) -> dict:
    # Frames may be directly in the video dir or inside an images/ subdirectory
    search_dirs = [video_dir]
    if (video_dir / "images").exists():
        search_dirs = [video_dir / "images"]

    frames = sorted(
        int(m.group(1))
        for search_dir in search_dirs
        for f in search_dir.iterdir()
        if (m := FRAME_RE.match(f.name))
    )
    if not frames:
        return None

    start_frame = frames[0]
    end_frame = frames[-1]

    # Collect mid-video excluded ranges (gaps between consecutive surgical frames)
    excluded_ranges = []
    for i in range(len(frames) - 1):
        if frames[i + 1] - frames[i] > 1:
            excluded_ranges.append([frames[i] + 1, frames[i + 1] - 1])

    return {
        "youtube_id": youtube_id,
        "procedure": proc_name,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "excluded_ranges": excluded_ranges,
        "total_surgical_frames": len(frames),
    }


def main():
    args = parse_args()
    full_dir = Path(args.full_videos_dir)
    out_dir = Path(args.output_dir)

    if not full_dir.exists():
        raise FileNotFoundError(f"full_videos_dir not found: {full_dir}")

    created = skipped = 0

    for proc_dir in sorted(full_dir.iterdir()):
        if not proc_dir.is_dir():
            continue
        procedure = proc_dir.name
        proc_out = out_dir / procedure
        proc_out.mkdir(parents=True, exist_ok=True)

        for vid_dir in sorted(proc_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            youtube_id = vid_dir.name
            out_file = proc_out / f"{youtube_id}.json"

            if out_file.exists() and not args.overwrite:
                skipped += 1
                continue

            index = build_index(procedure, youtube_id, vid_dir)
            if index is None:
                print(f"[WARN] No frames found: {procedure}/{youtube_id}")
                continue

            out_file.write_text(json.dumps(index, indent=2))

            gap_info = f", {len(index['excluded_ranges'])} excluded range(s)" if index["excluded_ranges"] else ""
            print(
                f"[OK] {procedure}/{youtube_id}: "
                f"frames {index['start_frame']}..{index['end_frame']} "
                f"({index['total_surgical_frames']} surgical{gap_info})"
            )
            created += 1

    print(f"\n[INFO] Done — {created} created, {skipped} skipped")


if __name__ == "__main__":
    main()
