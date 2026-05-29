#!/usr/bin/env python3
"""
Print a full statistics report for the ATLAS-120k dataset.

Reads clip_index.json files to count videos, clips, and frames per split
and per procedure. Optionally cross-checks against dataset_info.json.

Usage:
  python download/dataset_stats.py --atlas_dir E:/atlas120k
  python download/dataset_stats.py --atlas_dir E:/atlas120k --info dataset_info.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Print ATLAS-120k dataset statistics.")
    parser.add_argument(
        "--atlas_dir",
        required=True,
        help="Root of the ATLAS-120k dataset (contains train/, val/, test/)",
    )
    parser.add_argument(
        "--info",
        default=None,
        help="Optional path to dataset_info.json to cross-check totals against",
    )
    return parser.parse_args()


STANDARD_SPLITS = {"train", "val", "test"}


def collect_stats(atlas_dir: Path) -> dict:
    """
    Returns a nested dict:
      stats[split][procedure] = {videos, clips, frames}
    Only processes the standard splits (train, val, test).
    Unexpected directories are reported but not counted.
    """
    stats = defaultdict(lambda: defaultdict(lambda: {"videos": 0, "clips": 0, "frames": 0}))

    for split_dir in sorted(atlas_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        if split not in STANDARD_SPLITS:
            print(f"[WARN] Unexpected directory at dataset root (skipped): {split_dir}")
            continue
        for proc_dir in sorted(split_dir.iterdir()):
            if not proc_dir.is_dir():
                continue
            procedure = proc_dir.name
            for vid_dir in sorted(proc_dir.iterdir()):
                if not vid_dir.is_dir():
                    continue
                idx_file = vid_dir / "clip_index.json"
                if not idx_file.exists():
                    continue
                idx = json.loads(idx_file.read_text())
                stats[split][procedure]["videos"] += 1
                for frame_list in idx["clips"].values():
                    stats[split][procedure]["clips"] += 1
                    stats[split][procedure]["frames"] += len(frame_list)

    return stats


def sum_split(split_data: dict) -> dict:
    return {
        "videos": sum(p["videos"] for p in split_data.values()),
        "clips":  sum(p["clips"]  for p in split_data.values()),
        "frames": sum(p["frames"] for p in split_data.values()),
    }


def print_report(stats: dict):
    splits = sorted(stats.keys())
    all_procedures = sorted({p for s in stats.values() for p in s})

    # ── Header ───────────────────────────────────────────────────────────────
    col_w = max(len(p) for p in all_procedures) + 2
    split_w = 30  # width per split block

    print("=" * (col_w + len(splits) * split_w))
    print("ATLAS-120k Dataset Statistics")
    print("=" * (col_w + len(splits) * split_w))

    # ── Per-split, per-procedure table ───────────────────────────────────────
    header = f"{'Procedure':<{col_w}}"
    for s in splits:
        label = f"[{s}] videos/clips/frames"
        header += f"{label:>{split_w}}"
    print(header)
    print("-" * (col_w + len(splits) * split_w))

    for proc in all_procedures:
        row = f"{proc:<{col_w}}"
        for s in splits:
            d = stats[s].get(proc, {"videos": 0, "clips": 0, "frames": 0})
            cell = f"{d['videos']:>3} / {d['clips']:>4} / {d['frames']:>7,}"
            row += f"{cell:>{split_w}}"
        print(row)

    print("-" * (col_w + len(splits) * split_w))

    # ── Per-split totals ─────────────────────────────────────────────────────
    row = f"{'TOTAL':<{col_w}}"
    grand = {"videos": 0, "clips": 0, "frames": 0}
    for s in splits:
        t = sum_split(stats[s])
        cell = f"{t['videos']:>3} / {t['clips']:>4} / {t['frames']:>7,}"
        row += f"{cell:>{split_w}}"
        for k in grand:
            grand[k] += t[k]
    print(row)
    print("=" * (col_w + len(splits) * split_w))
    print(
        f"\nGrand total: {grand['videos']} videos | "
        f"{grand['clips']} clips | "
        f"{grand['frames']:,} frames | "
        f"{len(all_procedures)} procedures"
    )

    return stats, grand


def cross_check(stats: dict, grand: dict, info_path: Path):
    info = json.loads(info_path.read_text())
    print(f"\n{'-'*50}")
    print(f"Cross-check against {info_path.name}  (v{info['version']})")
    print(f"{'-'*50}")

    ok = True
    splits = sorted(stats.keys())

    for s in splits:
        expected = info["splits"].get(s, {})
        actual = sum_split(stats[s])
        for key in ["videos", "clips", "frames"]:
            exp_val = expected.get(key)
            act_val = actual[key]
            if exp_val is not None and exp_val != act_val:
                print(f"  MISMATCH  {s}.{key}: expected {exp_val:,}, got {act_val:,}")
                ok = False

    for key in ["videos", "clips", "frames"]:
        exp_val = info["total"].get(key)
        act_val = grand[key]
        if exp_val is not None and exp_val != act_val:
            print(f"  MISMATCH  total.{key}: expected {exp_val:,}, got {act_val:,}")
            ok = False

    if ok:
        print("  All totals match dataset_info.json.")


def main():
    args = parse_args()
    atlas_dir = Path(args.atlas_dir)

    if not atlas_dir.exists():
        raise FileNotFoundError(f"atlas_dir not found: {atlas_dir}")

    stats = collect_stats(atlas_dir)
    _, grand = print_report(stats)

    if args.info:
        cross_check(stats, grand, Path(args.info))


if __name__ == "__main__":
    main()
