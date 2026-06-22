#!/usr/bin/env python3
"""
Download ATLAS-120k YouTube videos organized by surgical procedure.

YouTube links are read from .txt files in --links_dir (one URL per line,
one file per procedure). Downloaded videos are saved to:
  <output_dir>/<procedure>/<video_id>.<ext>

Adapted from:
  https://github.com/aperezr20/SurgLaVi
  https://github.com/visurg-ai/LEMON
"""

import argparse
import multiprocessing
import os
from pathlib import Path

import yt_dlp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ATLAS-120k videos from YouTube."
    )
    parser.add_argument(
        "--links_dir",
        default="data/youtube_links",
        help="Directory containing per-procedure .txt files with YouTube URLs",
    )
    parser.add_argument(
        "--output_dir",
        default="raw_data",
        help="Root output directory; videos saved to <output_dir>/<procedure>/<video_id>.<ext>",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Path to a cookies.txt file for authenticated downloads",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        metavar="BROWSER",
        help="Read cookies directly from a logged-in browser: chrome, firefox, edge, safari",
    )
    parser.add_argument(
        "--player-client",
        default=None,
        help=(
            "Comma-separated YouTube player client(s) for yt-dlp to try, e.g. "
            "'default,tv,web_safari'. Use this if you see 'Only images are "
            "available' errors."
        ),
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        help="Number of parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--procedures",
        nargs="+",
        default=None,
        help="Limit download to these procedures (default: all found in links_dir)",
    )
    return parser.parse_args()


def read_links(txt_path: Path) -> list[str]:
    urls = []
    with open(txt_path, encoding="utf-8-sig") as f:  # utf-8-sig strips BOM if present
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def video_id_from_url(url: str) -> str:
    """Extract the 11-character YouTube video ID from a watch URL."""
    for part in url.split("v=")[1:]:
        return part.split("&")[0][:11]
    return url.rstrip("/").split("/")[-1][:11]


def download_video(
    url: str,
    output_dir: str,
    cookies: str | None,
    cookies_from_browser: str | None,
    player_client: str | None = None,
) -> tuple[str, bool, str | None]:
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "retries": 3,
        "quiet": True,
        "no_warnings": False,
        # Note: yt-dlp auto-detects an installed JS runtime (deno/node/bun) on
        # PATH to solve YouTube's n-challenge. Do NOT set "js_runtimes" here, as
        # pinning it disables auto-detection and breaks the solver.
    }
    if cookies:
        ydl_opts["cookiefile"] = cookies
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if player_client:
        ydl_opts["extractor_args"] = {
            "youtube": {"player_client": player_client.split(",")}
        }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return url, True, None
    except Exception as exc:
        return url, False, str(exc)


def _worker(args):
    # (url, output_dir, cookies, cookies_from_browser, player_client)
    return download_video(*args)


def main():
    args = parse_args()
    links_dir = Path(args.links_dir)
    output_dir = Path(args.output_dir)

    if not links_dir.exists():
        raise FileNotFoundError(f"links_dir not found: {links_dir}")

    txt_files = sorted(links_dir.glob("*.txt"))
    if args.procedures:
        txt_files = [f for f in txt_files if f.stem in args.procedures]

    if not txt_files:
        print("[INFO] No .txt files found. Nothing to download.")
        return

    tasks = []
    for txt_file in txt_files:
        procedure = txt_file.stem
        urls = read_links(txt_file)
        proc_out = output_dir / procedure
        proc_out.mkdir(parents=True, exist_ok=True)

        existing_ids = {f.stem for f in proc_out.iterdir() if f.is_file()}
        for url in urls:
            vid_id = video_id_from_url(url)
            if vid_id in existing_ids:
                print(f"[SKIP] {procedure}/{vid_id} already downloaded")
            else:
                tasks.append((
                    url,
                    str(proc_out),
                    args.cookies,
                    args.cookies_from_browser,
                    args.player_client,
                ))

    print(f"[INFO] {len(tasks)} videos to download across {len(txt_files)} procedure(s)")
    if not tasks:
        return

    with multiprocessing.Pool(processes=args.workers) as pool:
        results = pool.map(_worker, tasks)

    failures = [(url, err) for url, ok, err in results if not ok]
    successes = sum(1 for _, ok, _ in results if ok)
    print(f"\n[INFO] Done — {successes} succeeded, {len(failures)} failed")
    if failures:
        print("[WARN] Failed downloads:")
        for url, err in failures:
            print(f"  {url}: {err}")


if __name__ == "__main__":
    main()
