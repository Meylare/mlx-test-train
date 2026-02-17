import argparse
import json
import subprocess
import time
from pathlib import Path


def collect_urls(dataset_path: Path):
    data = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    seen = set()
    urls = []
    for row in data:
        author_record = row.get("author_record") or {}
        videos = author_record.get("videos") or []
        for v in videos:
            u = (v.get("webVideoUrl") or "").strip()
            if u and u not in seen:
                seen.add(u)
                urls.append(u)
    return urls


def count_archive_entries(path: Path) -> int:
    if not path.exists():
        return 0
    return len([x for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()])


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset_50vid_of_prof/authors_shopping_pass07_full_53.json")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--all", action="store_true", help="Download all URLs from dataset")
    p.add_argument("--outdir", default="dataset_50vid_of_prof/downloads_shopping")
    p.add_argument("--format", default="worst", help="yt-dlp format selector, default=worst")
    p.add_argument("--sleep-interval", type=float, default=2.0)
    p.add_argument("--max-sleep-interval", type=float, default=5.0)
    p.add_argument("--concurrency", type=int, default=1)
    args = p.parse_args()

    dataset = Path(args.dataset)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    urls = collect_urls(dataset)
    if not urls:
        raise SystemExit("No video URLs found in dataset")

    if args.all or args.count <= 0:
        target = urls
    else:
        target = urls[: args.count]

    if not target:
        raise SystemExit("Target URL list is empty")

    urls_file = outdir / f"urls_first_{len(target)}.txt"
    archive_file = outdir / "downloaded_archive.txt"
    urls_file.write_text("\n".join(target) + "\n", encoding="utf-8")
    before_count = count_archive_entries(archive_file)

    cmd = [
        "yt-dlp",
        "-a",
        str(urls_file),
        "-f",
        args.format,
        "--merge-output-format",
        "mp4",
        "-N",
        str(max(1, args.concurrency)),
        "--sleep-interval",
        str(args.sleep_interval),
        "--max-sleep-interval",
        str(args.max_sleep_interval),
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "--download-archive",
        str(archive_file),
        "-o",
        str(outdir / "%(uploader)s" / "%(id)s.%(ext)s"),
        "--no-warnings",
    ]

    print(f"Dataset URLs total: {len(urls)}")
    print(f"Target in this run: {len(target)}")
    print("Command:", " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - t0

    after_count = count_archive_entries(archive_file)
    downloaded_now = max(0, after_count - before_count)
    remaining = max(0, len(urls) - after_count)

    print(f"Exit code: {proc.returncode}")
    print(f"Elapsed seconds: {elapsed:.1f}")
    print(f"Archive before run: {before_count}")
    print(f"Archive after run:  {after_count}")
    print(f"Downloaded this run: {downloaded_now}")

    if downloaded_now > 0:
        sec_per_item = elapsed / downloaded_now
        print(f"Avg seconds per item (this run): {sec_per_item:.1f}")
        eta_total = sec_per_item * len(urls)
        eta_remaining = sec_per_item * remaining
        print(f"ETA for all {len(urls)} videos at current speed: {fmt_duration(eta_total)}")
        print(f"ETA for remaining {remaining} videos: {fmt_duration(eta_remaining)}")


if __name__ == "__main__":
    main()
