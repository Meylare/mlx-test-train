"""
Remove from unique_authors.txt any usernames that already appear in the dataset,
so only authors we have not yet parsed remain in the file.
"""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_50vid_of_prof")
AUTHORS_FILE = os.path.join(SCRIPT_DIR, "unique_authors.txt")


def _normalize(s):
    if not s:
        return ""
    return (s.strip() or "").lstrip("@")


def collect_parsed_authors(dataset_dir):
    """Collect author usernames that appear in our selected dataset (already parsed)."""
    seen = set()
    if not os.path.isdir(dataset_dir):
        return seen
    # tiktok_videos_unique.jsonl (selected videos)
    uniq_path = os.path.join(dataset_dir, "tiktok_videos_unique.jsonl")
    if os.path.isfile(uniq_path):
        with open(uniq_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    am = row.get("authorMeta") or {}
                    name = am.get("name") or ""
                    if name:
                        seen.add(_normalize(name))
                except json.JSONDecodeError:
                    pass
    # authors_viral_filtered.json
    af_path = os.path.join(dataset_dir, "authors_viral_filtered.json")
    if os.path.isfile(af_path):
        with open(af_path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            author = item.get("author") or {}
            name = author.get("name") or ""
            if name:
                seen.add(_normalize(name))
    return seen


def main():
    parsed = collect_parsed_authors(DATASET_DIR)
    print(f"Found {len(parsed)} authors already in dataset.")

    with open(AUTHORS_FILE, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    kept = []
    removed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _normalize(stripped) in parsed:
            removed += 1
            continue
        kept.append(stripped)

    with open(AUTHORS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(kept))
        if kept:
            f.write("\n")

    print(f"Removed {removed} already-parsed usernames from {AUTHORS_FILE}.")
    print(f"Left {len(kept)} usernames (not yet parsed).")


if __name__ == "__main__":
    main()
