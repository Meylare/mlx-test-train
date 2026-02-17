"""
Filter authors who:
  1. Have 30+ videos in the dataset
  2. Have median video views (playCount)
  3. Have at least one video with views >= (median * MULTIPLIER)

Output: JSON with structure author[ { metadata, videos: [ ... ] }, ... ]
"""
import json
import os
import statistics
from collections import defaultdict

INPUT_FILE = os.environ.get(
    "FILTER_AUTHORS_INPUT_FILE",
    os.path.join("dataset_50vid_of_prof", "tiktok_videos_unique.jsonl"),
)
OUTPUT_FILE = os.environ.get(
    "FILTER_AUTHORS_OUTPUT_FILE",
    os.path.join("dataset_50vid_of_prof", "authors_viral_filtered.json"),
)

MIN_VIDEOS = 30
MULTIPLIER = 4.5   # video must have >= median * 4.5 views (use 4.0–5.0)


def get_author_key(row):
    am = row.get("authorMeta") or {}
    return (am.get("id"), am.get("name") or "")


def author_metadata_from_row(row):
    """Extract author metadata from a video row (same for all videos by this author)."""
    am = row.get("authorMeta") or {}
    return {
        "id": am.get("id"),
        "name": am.get("name"),
        "profileUrl": am.get("profileUrl"),
        "nickName": am.get("nickName"),
        "verified": am.get("verified"),
        "signature": am.get("signature"),
        "avatar": am.get("avatar"),
        "originalAvatarUrl": am.get("originalAvatarUrl"),
        "privateAccount": am.get("privateAccount"),
        "fans": am.get("fans"),
        "heart": am.get("heart"),
        "video": am.get("video"),
        "following": am.get("following"),
        "friends": am.get("friends"),
        "digg": am.get("digg"),
    }


def video_without_author_meta(row):
    """Return video object without nesting authorMeta (author is at parent level)."""
    out = dict(row)
    out.pop("authorMeta", None)
    out.pop("input", None)
    out.pop("fromProfileSection", None)
    return out


def main():
    if not os.path.isfile(INPUT_FILE):
        print(f"Input not found: {INPUT_FILE}")
        return

    # Group by author
    by_author = defaultdict(list)
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = get_author_key(row)
                if key[0] or key[1]:
                    by_author[key].append(row)
            except json.JSONDecodeError:
                pass

    # Filter: 30+ videos, median views, at least one video >= median * MULTIPLIER
    results = []
    for (author_id, author_name), videos in by_author.items():
        if len(videos) < MIN_VIDEOS:
            continue
        plays = []
        for v in videos:
            p = v.get("playCount")
            if p is not None and isinstance(p, (int, float)):
                plays.append(int(p))
        if not plays:
            continue
        median_views = statistics.median(plays)
        threshold = median_views * MULTIPLIER
        has_viral = any(p >= threshold for p in plays)
        if not has_viral:
            continue

        meta = author_metadata_from_row(videos[0])
        meta["videoCountInDataset"] = len(videos)
        meta["medianPlayCount"] = median_views
        meta["viralThresholdUsed"] = threshold

        video_list = [video_without_author_meta(v) for v in videos]
        results.append({
            "author": meta,
            "videos": video_list,
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_videos = sum(len(r["videos"]) for r in results)
    print(f"Authors with 30+ videos: {sum(1 for v in by_author.values() if len(v) >= MIN_VIDEOS)}")
    print(f"Authors passing viral filter (>=1 video with views >= {MULTIPLIER}x median): {len(results)}")
    print(f"Total videos in output: {total_videos}")
    print(f"Written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
