#!/usr/bin/env python3
"""
Process raw YouTube videos into the ATLAS-120k dataset format.

Given:
  - Raw videos downloaded by download_videos.py  (raw_data/<procedure>/<video_id>.mp4)
  - Annotation download from HuggingFace         (<atlas_dir>/<split>/<procedure>/<video>/)
    which contains machine_masks/ and clip_index.json per clip/video

This script:
  1. Converts each raw video to 15 fps (streamed through ffmpeg; no temp files).
  2. Reads the per-video clip_index.json to know exactly which frames are needed.
  3. Saves those frames as <atlas_dir>/<split>/<procedure>/<video>/<clip>/images/frame_XXXXXX.jpg

Usage:
  python download/process_atlas120k.py \\
      --raw_data_dir raw_data \\
      --atlas_dir    /path/to/hf_download \\
      --workers      4

Requirements: ffmpeg must be installed and on PATH.
"""

import argparse
import json
import re
import subprocess
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


TARGET_FPS = 15
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ATLAS-120k clip frames from raw YouTube videos."
    )
    parser.add_argument(
        "--raw_data_dir",
        default="raw_data",
        help="Root directory of raw videos: <raw_data_dir>/<procedure>/<youtube_id>.<ext>",
    )
    parser.add_argument(
        "--atlas_dir",
        required=True,
        help="Root of the HuggingFace annotation download (contains train/, val/, test/)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test)",
    )
    parser.add_argument(
        "--procedures",
        nargs="+",
        default=None,
        help="Limit to specific procedures",
    )
    parser.add_argument(
        "--workers",
        default=2,
        type=int,
        help="Parallel video workers (default: 2; each worker uses ffmpeg + CPU)",
    )
    parser.add_argument(
        "--quality",
        default=2,
        type=int,
        help="JPEG quality for saved frames (ffmpeg -q:v, 1=best; default: 2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract frames even if images/ folder already exists for a clip",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Video ID extraction (mirrors generate_clip_index.py)
# ---------------------------------------------------------------------------

def extract_youtube_id(folder_name: str) -> str:
    name = folder_name
    if name.endswith("_ROBOT"):
        name = name[:-6]
    if name.startswith("watch#v="):
        name = name[8:]
    if "#si=" in name:
        name = name[: name.index("#si=")]
    return name[:11]


# ---------------------------------------------------------------------------
# Raw video lookup
# ---------------------------------------------------------------------------

def find_raw_video(raw_data_dir: Path, procedure: str, youtube_id: str) -> Path | None:
    proc_dir = raw_data_dir / procedure
    if not proc_dir.exists():
        return None
    for ext in VIDEO_EXTENSIONS:
        candidate = proc_dir / f"{youtube_id}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Video metadata via ffprobe
# ---------------------------------------------------------------------------

def get_video_info(video_path: Path) -> tuple[int, int, float]:
    """Return (width, height, fps) using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    parts = out.split(",")
    w, h = int(parts[0]), int(parts[1])
    num, den = parts[2].split("/")
    fps = float(num) / float(den)
    return w, h, fps


# ---------------------------------------------------------------------------
# Core per-video processing
# ---------------------------------------------------------------------------

def process_video(task: dict) -> dict:
    """
    Stream a raw video through ffmpeg at TARGET_FPS and save the frames
    listed in the clip index to the atlas_dir images/ folders.

    Returns a result dict with keys: folder_name, ok, clips_done, frames_written, error.
    """
    folder_name = task["folder_name"]
    video_path = Path(task["video_path"])
    clips = task["clips"]          # {clip_id: [frame_nums]}
    frame_digits = task["frame_digits"]
    overwrite = task["overwrite"]
    quality = task["quality"]

    result = {"folder_name": folder_name, "ok": False, "clips_done": 0, "frames_written": 0, "error": None}

    # Build the complete set of needed frames and map them to output paths
    needed: dict[int, list[Path]] = {}  # frame_num → [output_paths]
    for clip_id, frame_nums in clips.items():
        images_dir = Path(task["clip_dirs"][clip_id]) / "images"

        if images_dir.exists() and not overwrite:
            existing = sum(1 for _ in images_dir.glob("frame_*.jpg"))
            if existing == len(frame_nums):
                result["clips_done"] += 1
                continue  # already done

        images_dir.mkdir(parents=True, exist_ok=True)
        for fn in frame_nums:
            out_path = images_dir / f"frame_{fn:0{frame_digits}d}.jpg"
            needed.setdefault(fn, []).append(out_path)

    if not needed:
        result["ok"] = True
        return result

    # Get video dimensions
    try:
        w, h, src_fps = get_video_info(video_path)
    except Exception as exc:
        result["error"] = f"ffprobe failed: {exc}"
        return result

    frame_size = w * h * 3  # BGR raw bytes per frame

    # Pipe all frames at TARGET_FPS through ffmpeg
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={TARGET_FPS}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-loglevel", "error",
        "pipe:1",
    ]

    max_needed_frame = max(needed.keys())

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        current_idx = 0

        while current_idx <= max_needed_frame:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break  # end of video

            if current_idx in needed:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(1, 100 - quality * 10)]
                for out_path in needed[current_idx]:
                    cv2.imwrite(str(out_path), frame, encode_params)
                    result["frames_written"] += 1

            current_idx += 1

        proc.stdout.close()
        proc.wait()

        if proc.returncode not in (0, None) and result["frames_written"] == 0:
            stderr = proc.stderr.read().decode()
            result["error"] = f"ffmpeg exited {proc.returncode}: {stderr[:300]}"
            return result

    except Exception as exc:
        result["error"] = str(exc)
        return result

    result["clips_done"] += len({cn for cn, fns in clips.items() if any(
        (Path(task["clip_dirs"][cn]) / "images" / f"frame_{fn:0{frame_digits}d}.jpg").exists()
        for fn in fns
    )})
    result["ok"] = True
    return result


# ---------------------------------------------------------------------------
# Task collection
# ---------------------------------------------------------------------------

def collect_tasks(atlas_dir: Path, raw_data_dir: Path, splits: list[str],
                  procedures: list[str] | None, overwrite: bool, quality: int) -> list[dict]:
    tasks = []
    missing_videos = []

    for split in splits:
        split_dir = atlas_dir / split
        if not split_dir.exists():
            continue
        for proc_dir in sorted(split_dir.iterdir()):
            if not proc_dir.is_dir():
                continue
            procedure = proc_dir.name
            if procedures and procedure not in procedures:
                continue

            for video_dir in sorted(proc_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                index_file = video_dir / "clip_index.json"
                if not index_file.exists():
                    print(f"[WARN] No clip_index.json: {video_dir} — run generate_clip_index.py first")
                    continue

                index = json.loads(index_file.read_text())
                youtube_id = index["youtube_id"]
                video_path = find_raw_video(raw_data_dir, procedure, youtube_id)

                if video_path is None:
                    missing_videos.append(f"{procedure}/{youtube_id}")
                    continue

                clip_dirs = {
                    clip_id: str(video_dir / clip_id)
                    for clip_id in index["clips"]
                }

                tasks.append({
                    "folder_name": video_dir.name,
                    "video_path": str(video_path),
                    "clips": index["clips"],
                    "clip_dirs": clip_dirs,
                    "frame_digits": index["frame_digits"],
                    "overwrite": overwrite,
                    "quality": quality,
                })

    if missing_videos:
        print(f"\n[WARN] {len(missing_videos)} raw videos not found in {raw_data_dir}:")
        for v in missing_videos[:10]:
            print(f"  {v}")
        if len(missing_videos) > 10:
            print(f"  ... and {len(missing_videos) - 10} more")

    return tasks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    atlas_dir = Path(args.atlas_dir)
    raw_data_dir = Path(args.raw_data_dir)

    tasks = collect_tasks(
        atlas_dir, raw_data_dir, args.splits, args.procedures, args.overwrite, args.quality
    )

    if not tasks:
        print("[INFO] No videos to process.")
        return

    print(f"[INFO] Processing {len(tasks)} videos with {args.workers} worker(s)...\n")

    total_frames = 0
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_video, t): t["folder_name"] for t in tasks}
        for future in as_completed(futures):
            r = future.result()
            if r["ok"]:
                total_frames += r["frames_written"]
                print(
                    f"[OK] {r['folder_name']}: "
                    f"{r['clips_done']} clips, {r['frames_written']} frames written"
                )
            else:
                failures.append((r["folder_name"], r["error"]))
                print(f"[FAIL] {r['folder_name']}: {r['error']}")

    print(f"\n[INFO] Done — {total_frames} frames written, {len(failures)} failures")
    if failures:
        for name, err in failures:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
