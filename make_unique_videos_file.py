"""Read tiktok_videos_account*.jsonl, keep one record per video ID, write a single unique-videos file."""
import json
import os
import re

DATASET_DIR = "dataset_50vid_of_prof"
OUTPUT_FILENAME = os.environ.get("MAKE_UNIQUE_OUTPUT_FILE", "tiktok_videos_unique.jsonl")
# Pattern: tiktok_videos_accountN_YYYY-MM-DD_HH-MM-SS.jsonl
TIMESTAMP_PATTERN = re.compile(r"tiktok_videos_account\d+_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.jsonl$")


def get_latest_batch_files(dataset_dir):
    """List tiktok_videos_account*.jsonl, group by timestamp, return files for the latest batch (most files wins; tie = latest timestamp)."""
    all_files = [
        f for f in os.listdir(dataset_dir)
        if f.startswith("tiktok_videos_account") and f.endswith(".jsonl")
    ]
    by_ts = {}
    for fn in all_files:
        m = TIMESTAMP_PATTERN.match(fn)
        if m:
            ts = m.group(1)
            by_ts.setdefault(ts, []).append(fn)
    if not by_ts:
        return []
    # Choose batch: most account files; if tie, latest timestamp (lexicographic order is fine for YYYY-MM-DD_HH-MM-SS)
    best_ts = max(by_ts.keys(), key=lambda t: (len(by_ts[t]), t))
    return sorted(by_ts[best_ts])


def main():
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATASET_DIR)
    files = get_latest_batch_files(dataset_dir)
    if not files:
        print(f"No tiktok_videos_account*_YYYY-MM-DD_HH-MM-SS.jsonl files in {dataset_dir}.")
        return
    print(f"Using latest batch: {len(files)} files.")

    seen_ids = set()
    unique_rows = []
    total_read = 0

    for fn in files:
        path = os.path.join(dataset_dir, fn)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_read += 1
                try:
                    row = json.loads(line)
                    vid = row.get("id")
                    if not vid:
                        vid = (row.get("webVideoUrl") or "").strip()
                    if vid and vid not in seen_ids:
                        seen_ids.add(vid)
                        unique_rows.append(row)
                except json.JSONDecodeError:
                    pass

    out_path = os.path.join(dataset_dir, OUTPUT_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in unique_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Read {total_read} rows from {len(files)} files.")
    print(f"Unique videos: {len(unique_rows)}")
    print(f"Written to: {out_path}")

if __name__ == "__main__":
    main()
