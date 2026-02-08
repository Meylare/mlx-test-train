#!/usr/bin/env python3
import argparse
import json
import sys

from src.inference.db import init_db, upsert_video_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest metadata into SQLite.")
    parser.add_argument("--db-path", default="data/viral.db", help="Path to SQLite DB.")
    parser.add_argument("--input-jsonl", required=True, help="Path to JSONL file.")
    parser.add_argument("--mode", default="upsert", choices=["upsert"], help="Ingest mode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    init_db(args.db_path)

    count = 0
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

            video_id = obj.get("video_id")
            gcs_uri = obj.get("gcs_uri")
            if not video_id or not gcs_uri:
                raise ValueError(f"Missing video_id or gcs_uri on line {line_no}")

            metadata = obj.get("metadata")
            metadata_json = obj.get("metadata_json", metadata)

            row = {
                "video_id": video_id,
                "gcs_uri": gcs_uri,
                "transcript": obj.get("transcript"),
                "system_prompt": obj.get("system_prompt"),
                "metadata_json": metadata_json,
            }
            upsert_video_row(args.db_path, row)
            count += 1

    print(f"Ingested {count} rows into {args.db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
