#!/usr/bin/env python3
"""
Process raw YouTube videos into the ATLAS-120k dataset format.

Given:
  - Raw videos downloaded by download_videos.py  (raw_data/<procedure>/<youtube_id>.mp4)
  - Annotation download from HuggingFace         (<atlas_dir>/<split>/<procedure>/<video>/)
    which contains machine_masks/ and clip_index.json per clip/video
  - Surgical frame indices                        (data/surgical_index/<procedure>/<youtube_id>.json)
    which list the valid surgical frame range and any mid-video excluded segments

This script:
  1. Converts each raw video to 15 fps (streamed through ffmpeg; no temp files).
  2. Applies the surgical frame filter: frames before the surgical start, after the
     surgical end, or inside excluded mid-video ranges are skipped.
  3. Optionally saves ALL surgical frames to a separate directory (--save_surgical_frames),
     reproducing the atlas120k_full_videos layout (<procedure>/<youtube_id>/frame_XXXXXX.jpg).
  4. Reads the per-video clip_index.json to know exactly which frames are needed per clip,
     and saves those frames as <atlas_dir>/<split>/<procedure>/<video>/<clip>/images/frame_XXXXXX.jpg.

Usage:
  python download/process_atlas120k.py \\
      --raw_data_dir      raw_data \\
      --atlas_dir         atlas120k \\
      --surgical_index_dir data/surgical_index \\
      --workers           4

  # Also save all surgical frames (reproduces atlas120k_full_videos):
  python download/process_atlas120k.py \\
      --raw_data_dir           raw_data \\
      --atlas_dir              atlas120k \\
      --surgical_index_dir     data/surgical_index \\
      --save_surgical_frames \\
      --surgical_frames_dir    atlas120k_full_videos \\
      --workers                4

Requirements: ffmpeg must be installed and on PATH.
"""

import argparse
import json
import subprocess
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
        "--surgical_index_dir",
        default="data/surgical_index",
        help="Directory of per-video surgical frame index JSONs (default: data/surgical_index)",
    )
    parser.add_argument(
        "--save_surgical_frames",
        action="store_true",
        help="Also save all surgical frames (reproduces atlas120k_full_videos layout)",
    )
    parser.add_argument(
        "--surgical_frames_dir",
        default="atlas120k_full_videos",
        help="Output directory for surgical frames when --save_surgical_frames is set",
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
        help="Parallel video workers (default: 2)",
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
# Helpers
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


def find_raw_video(raw_data_dir: Path, procedure: str, youtube_id: str) -> Path | None:
    proc_dir = raw_data_dir / procedure
    if not proc_dir.exists():
        return None
    for ext in VIDEO_EXTENSIONS:
        candidate = proc_dir / f"{youtube_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def get_video_info(video_path: Path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if w == 0 or h == 0:
        raise RuntimeError(f"Could not read dimensions from: {video_path}")
    return w, h, fps


def load_surgical_index(surgical_index_dir: Path, procedure: str, youtube_id: str) -> dict | None:
    index_file = surgical_index_dir / procedure / f"{youtube_id}.json"
    if index_file.exists():
        return json.loads(index_file.read_text())
    return None


def build_excluded_set(surgical_index: dict) -> tuple[int, int, set[int]]:
    """
    Return (start_frame, end_frame, excluded_set) from a surgical index.
    excluded_set contains all frame numbers that should be skipped.
    """
    start = surgical_index["start_frame"]
    end = surgical_index["end_frame"]
    excluded = set()
    for from_f, to_f in surgical_index.get("excluded_ranges", []):
        excluded.update(range(from_f, to_f + 1))
    return start, end, excluded


# ---------------------------------------------------------------------------
# Core per-video processing
# ---------------------------------------------------------------------------

def process_video(task: dict) -> dict:
    folder_name      = task["folder_name"]
    video_path       = Path(task["video_path"])
    clips            = task["clips"]           # {clip_id: [frame_nums]}
    clip_dirs        = task["clip_dirs"]       # {clip_id: str path}
    frame_digits     = task["frame_digits"]
    overwrite        = task["overwrite"]
    quality          = task["quality"]
    surgical_index   = task["surgical_index"]  # dict or None
    save_surgical    = task["save_surgical"]
    surgical_out_dir = task["surgical_out_dir"]  # str or None

    result = {
        "folder_name": folder_name,
        "ok": False,
        "clips_done": 0,
        "frames_written": 0,
        "surgical_written": 0,
        "error": None,
    }

    # ── Surgical filter ──────────────────────────────────────────────────────
    if surgical_index is not None:
        surg_start, surg_end, surg_excluded = build_excluded_set(surgical_index)
    else:
        surg_start, surg_end, surg_excluded = 0, None, set()

    # ── Build clip frame lookup ───────────────────────────────────────────────
    needed: dict[int, list[Path]] = {}  # frame_num → [output_paths]
    for clip_id, frame_nums in clips.items():
        images_dir = Path(clip_dirs[clip_id]) / "images"
        if images_dir.exists() and not overwrite:
            existing = sum(1 for _ in images_dir.glob("frame_*.jpg"))
            if existing == len(frame_nums):
                result["clips_done"] += 1
                continue
        images_dir.mkdir(parents=True, exist_ok=True)
        for fn in frame_nums:
            out_path = images_dir / f"frame_{fn:0{frame_digits}d}.jpg"
            needed.setdefault(fn, []).append(out_path)

    # ── Determine stream end point ────────────────────────────────────────────
    max_clip_frame = max(needed.keys()) if needed else -1

    if save_surgical and surg_end is not None:
        surgical_out = Path(surgical_out_dir)
        surgical_out.mkdir(parents=True, exist_ok=True)
        stream_end = max(max_clip_frame, surg_end)
    elif save_surgical:
        surgical_out = Path(surgical_out_dir)
        surgical_out.mkdir(parents=True, exist_ok=True)
        stream_end = max_clip_frame
    else:
        surgical_out = None
        stream_end = max_clip_frame

    if stream_end < 0 and not save_surgical:
        # All clips already done, nothing to write
        result["ok"] = True
        return result

    # ── Get video dimensions ─────────────────────────────────────────────────
    try:
        w, h, _ = get_video_info(video_path)
    except Exception as exc:
        result["error"] = f"ffprobe failed: {exc}"
        return result

    frame_size = w * h * 3
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(1, 100 - quality * 10)]

    # ── Stream at TARGET_FPS and write frames ────────────────────────────────
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={TARGET_FPS}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-loglevel", "error",
        "pipe:1",
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        current_idx = 0

        while current_idx <= stream_end:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            # Skip non-surgical prefix and excluded ranges
            is_surgical = (
                current_idx >= surg_start
                and current_idx not in surg_excluded
                and (surg_end is None or current_idx <= surg_end)
            )

            if is_surgical:
                if current_idx in needed or (save_surgical and surgical_out is not None):
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))

                    # Save surgical frame if requested
                    if save_surgical and surgical_out is not None:
                        surg_path = surgical_out / f"frame_{current_idx:0{frame_digits}d}.jpg"
                        cv2.imwrite(str(surg_path), frame, encode_params)
                        result["surgical_written"] += 1

                    # Save clip frames
                    for out_path in needed.get(current_idx, []):
                        cv2.imwrite(str(out_path), frame, encode_params)
                        result["frames_written"] += 1

            current_idx += 1

        proc.stdout.close()
        proc.wait()

        if proc.returncode not in (0, None) and result["frames_written"] == 0 and result["surgical_written"] == 0:
            stderr = proc.stderr.read().decode()
            result["error"] = f"ffmpeg exited {proc.returncode}: {stderr[:300]}"
            return result

    except Exception as exc:
        result["error"] = str(exc)
        return result

    result["clips_done"] += len(clips) - (result["clips_done"])
    result["ok"] = True
    return result


# ---------------------------------------------------------------------------
# Task collection
# ---------------------------------------------------------------------------

def collect_tasks(
    atlas_dir: Path,
    raw_data_dir: Path,
    surgical_index_dir: Path,
    splits: list[str],
    procedures: list[str] | None,
    overwrite: bool,
    quality: int,
    save_surgical: bool,
    surgical_frames_dir: str,
) -> list[dict]:
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

                surgical_index = load_surgical_index(surgical_index_dir, procedure, youtube_id)

                surg_out_dir = None
                if save_surgical:
                    surg_out_dir = str(Path(surgical_frames_dir) / procedure / youtube_id)

                tasks.append({
                    "folder_name": video_dir.name,
                    "video_path": str(video_path),
                    "clips": index["clips"],
                    "clip_dirs": {cid: str(video_dir / cid) for cid in index["clips"]},
                    "frame_digits": index["frame_digits"],
                    "overwrite": overwrite,
                    "quality": quality,
                    "surgical_index": surgical_index,
                    "save_surgical": save_surgical,
                    "surgical_out_dir": surg_out_dir,
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
    atlas_dir          = Path(args.atlas_dir)
    raw_data_dir       = Path(args.raw_data_dir)
    surgical_index_dir = Path(args.surgical_index_dir)

    if not surgical_index_dir.exists():
        print(f"[WARN] surgical_index_dir not found ({surgical_index_dir}) — no surgical filtering applied")

    tasks = collect_tasks(
        atlas_dir, raw_data_dir, surgical_index_dir,
        args.splits, args.procedures, args.overwrite, args.quality,
        args.save_surgical_frames, args.surgical_frames_dir,
    )

    if not tasks:
        print("[INFO] No videos to process.")
        return

    n_with_index = sum(1 for t in tasks if t["surgical_index"] is not None)
    print(f"[INFO] Processing {len(tasks)} videos ({n_with_index} with surgical index) using {args.workers} worker(s)...\n")

    total_clip_frames = 0
    total_surg_frames = 0
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_video, t): t["folder_name"] for t in tasks}
        for future in as_completed(futures):
            r = future.result()
            if r["ok"]:
                total_clip_frames += r["frames_written"]
                total_surg_frames += r["surgical_written"]
                parts = [f"{r['clips_done']} clips", f"{r['frames_written']} clip frames"]
                if r["surgical_written"]:
                    parts.append(f"{r['surgical_written']} surgical frames")
                print(f"[OK] {r['folder_name']}: {', '.join(parts)}")
            else:
                failures.append((r["folder_name"], r["error"]))
                print(f"[FAIL] {r['folder_name']}: {r['error']}")

    print(f"\n[INFO] Done — {total_clip_frames} clip frames written, {total_surg_frames} surgical frames written, {len(failures)} failures")
    if failures:
        for name, err in failures:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
