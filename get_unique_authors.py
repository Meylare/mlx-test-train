#!/usr/bin/env python3
"""
Reads 2 dataset JSONL files from the dataset folder and outputs unique author usernames.
Author username is taken from authorMeta.name in each record.
"""

import json
import argparse
from pathlib import Path


def get_authors_from_file(path: Path) -> set[str]:
    """Extract unique author usernames from a JSONL file."""
    usernames = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                author = obj.get("authorMeta", {}).get("name")
                if author:
                    usernames.add(author)
            except (json.JSONDecodeError, TypeError):
                continue
    return usernames


def main():
    dataset_dir = Path(__file__).resolve().parent / "dataset"
    parser = argparse.ArgumentParser(description="Get unique author usernames from 2 dataset JSONL files.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Paths to 2 JSONL files (default: first 2 dataset_*.jsonl in dataset folder)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Write usernames to file (one per line) instead of stdout",
    )
    args = parser.parse_args()

    if args.files:
        if len(args.files) != 2:
            parser.error("Exactly 2 files required when specifying paths.")
        paths = [Path(p) for p in args.files]
    else:
        jsonl_files = sorted(dataset_dir.glob("dataset_*.jsonl"))
        if len(jsonl_files) < 2:
            print("Need at least 2 dataset_*.jsonl files in the dataset folder.", file=__import__("sys").stderr)
            raise SystemExit(1)
        paths = jsonl_files[:2]

    for p in paths:
        if not p.exists():
            print(f"File not found: {p}", file=__import__("sys").stderr)
            raise SystemExit(1)

    all_usernames: set[str] = set()
    for p in paths:
        all_usernames |= get_authors_from_file(p)

    sorted_usernames = sorted(all_usernames)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text("\n".join(sorted_usernames) + "\n", encoding="utf-8")
        print(f"Wrote {len(sorted_usernames)} unique usernames to {out_path}")
    else:
        for u in sorted_usernames:
            print(u)


if __name__ == "__main__":
    main()
