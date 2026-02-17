"""Check if video IDs are unique across tiktok_videos_account*.jsonl files."""
import json
import os
from collections import defaultdict

DATASET_DIR = "dataset_50vid_of_prof"
# Check the 3 files from batch 2026-02-15_11-37-53
PATTERN = "tiktok_videos_account*_2026-02-15_11-37-53.jsonl"

def main():
    files = sorted(f for f in os.listdir(DATASET_DIR) if f.startswith("tiktok_videos_account") and f.endswith(".jsonl"))
    # Only the 3 files from batch 2026-02-15_11-37-53
    files = [f for f in files if "2026-02-15_11-37-53" in f]
    if not files:
        print("No tiktok_videos_account*.jsonl files found.")
        return

    # file -> set of video ids; id -> list of (file, line_no)
    ids_by_file = {}
    id_occurrences = defaultdict(list)
    total_lines = 0

    for fn in files:
        path = os.path.join(DATASET_DIR, fn)
        ids = set()
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    row = json.loads(line)
                    vid = row.get("id") or row.get("webVideoUrl", "").strip()
                    if vid:
                        ids.add(vid)
                        id_occurrences[vid].append((fn, line_no))
                except json.JSONDecodeError:
                    pass
        ids_by_file[fn] = ids

    all_ids = set(id_occurrences.keys())
    unique_count = len(all_ids)
    duplicates = {vid: locs for vid, locs in id_occurrences.items() if len(locs) > 1}

    print("Files checked:", files)
    print()
    for fn in files:
        print(f"  {fn}: {len(ids_by_file[fn])} videos")
    print()
    print(f"Total lines:    {total_lines}")
    print(f"Unique video IDs: {unique_count}")
    print(f"Duplicates (same ID in >1 place): {len(duplicates)}")
    if duplicates:
        print()
        print("Sample duplicate IDs (first 10):")
        for i, (vid, locs) in enumerate(list(duplicates.items())[:10]):
            print(f"  {vid} -> {locs}")
    else:
        print()
        print("All 3k videos are unique across the 3 files.")

if __name__ == "__main__":
    main()
