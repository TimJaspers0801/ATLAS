#!/usr/bin/env python3
"""
Generate per-video clip index files for the ATLAS-120k dataset.

Scans the dataset directory (either the full local dataset at E:/atlas120k or
the HuggingFace annotation download) and writes one clip_index.json per video.
Each JSON lists the exact frame numbers belonging to every clip, derived from
the filenames present in the images/ or machine_masks/ subdirectories.

The resulting clip_index.json files are stored alongside the mask data:
  <dataset_root>/<split>/<procedure>/<video_folder>/clip_index.json

These index files are included in the HuggingFace release so that end users
do not need to run this script themselves.

Usage (run once by the dataset authors):
  python download/generate_clip_index.py --dataset_dir E:/atlas120k
"""

import argparse
import json
import re
from pathlib import Path


# Subdirectories to search for frame files (in priority order)
FRAME_SUBDIRS = ["machine_masks", "images", "masks"]
FRAME_PATTERN = re.compile(r"^frame_(\d+)\.(jpg|png)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate clip_index.json files for ATLAS-120k."
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Root of the ATLAS-120k dataset (contains train/, val/, test/)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing clip_index.json files",
    )
    return parser.parse_args()


def extract_youtube_id(folder_name: str) -> str:
    """
    Extract the bare YouTube video ID from an ATLAS-120k video folder name.

    Handles all naming patterns found in the dataset:
      4FpUGXO9mzg              → 4FpUGXO9mzg
      J5bg8KTYrw0_ROBOT        → J5bg8KTYrw0
      watch#v=1ud3syYKD3A_ROBOT → 1ud3syYKD3A
      6X7BRo4hNt8#si=xxx_ROBOT → 6X7BRo4hNt8
    """
    name = folder_name

    # Strip _ROBOT suffix
    if name.endswith("_ROBOT"):
        name = name[:-6]

    # Strip watch#v= prefix (malformed URL stored as folder name)
    if name.startswith("watch#v="):
        name = name[8:]

    # Strip #si= fragment parameter
    if "#si=" in name:
        name = name[: name.index("#si=")]

    # YouTube IDs are exactly 11 characters
    return name[:11]


def frame_digits(sample_stem: str) -> int:
    """Return the zero-pad width used for frame filenames in this video."""
    digits = sample_stem.replace("frame_", "")
    return len(digits)


def collect_clip_frames(clip_dir: Path) -> tuple[list[int], int] | tuple[None, None]:
    """
    Return the sorted list of frame numbers and the digit width for a clip.
    Searches FRAME_SUBDIRS in order and uses the first non-empty one found.
    """
    for subdir_name in FRAME_SUBDIRS:
        subdir = clip_dir / subdir_name
        if not subdir.exists():
            continue
        matches = []
        for f in subdir.iterdir():
            m = FRAME_PATTERN.match(f.name)
            if m:
                matches.append((int(m.group(1)), f.stem))
        if matches:
            matches.sort()
            frame_nums = [n for n, _ in matches]
            digits = frame_digits(matches[0][1])
            return frame_nums, digits
    return None, None


def process_video(video_dir: Path, procedure: str, split: str, overwrite: bool) -> bool:
    out_file = video_dir / "clip_index.json"
    if out_file.exists() and not overwrite:
        return False  # already done

    folder_name = video_dir.name
    youtube_id = extract_youtube_id(folder_name)
    is_robot = "_ROBOT" in folder_name

    clips_data: dict[str, list[int]] = {}
    video_frame_digits = None

    for clip_dir in sorted(video_dir.iterdir()):
        if not clip_dir.is_dir() or not clip_dir.name.startswith("clip_"):
            continue
        frame_nums, digits = collect_clip_frames(clip_dir)
        if frame_nums is None:
            continue
        clips_data[clip_dir.name] = frame_nums
        if video_frame_digits is None:
            video_frame_digits = digits

    if not clips_data:
        return False

    index = {
        "folder_name": folder_name,
        "youtube_id": youtube_id,
        "procedure": procedure,
        "split": split,
        "is_robot": is_robot,
        "frame_digits": video_frame_digits or 6,
        "clips": clips_data,
    }

    out_file.write_text(json.dumps(index, indent=2))
    n_clips = len(clips_data)
    n_frames = sum(len(v) for v in clips_data.values())
    print(f"[OK] {split}/{procedure}/{folder_name}: {n_clips} clips, {n_frames} frames")
    return True


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    created = skipped = 0
    for split in args.splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"[WARN] Split directory not found: {split_dir}")
            continue
        for proc_dir in sorted(split_dir.iterdir()):
            if not proc_dir.is_dir():
                continue
            procedure = proc_dir.name
            for video_dir in sorted(proc_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                if process_video(video_dir, procedure, split, args.overwrite):
                    created += 1
                else:
                    skipped += 1

    print(f"\n[INFO] Done — {created} created, {skipped} skipped (use --overwrite to regenerate)")


if __name__ == "__main__":
    main()
