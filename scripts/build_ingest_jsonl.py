#!/usr/bin/env python3
import argparse
import json
from typing import Dict, Iterable, List, Optional

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge parquet transcript with JSONL metadata.")
    parser.add_argument("--parquet", required=True, help="Path to parquet with transcript.")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL with metadata.")
    parser.add_argument("--out", required=True, help="Output JSONL for SQLite ingest.")
    parser.add_argument("--video-id-key", default="id", help="Video ID key in JSONL.")
    parser.add_argument("--video-path-key", default="video_path", help="Video path key in JSONL.")
    parser.add_argument(
        "--transcript-mode",
        choices=["join", "raw"],
        default="join",
        help="join = concatenate segment texts; raw = store JSON list as string.",
    )
    return parser.parse_args()


def _load_transcripts(parquet_path: str, mode: str) -> Dict[str, str]:
    pf = pq.ParquetFile(parquet_path)
    transcripts: Dict[str, str] = {}

    for batch in pf.iter_batches(columns=["video_id", "text"]):
        rows = batch.to_pylist()
        for row in rows:
            vid = row.get("video_id")
            if not vid:
                continue
            text = row.get("text")
            if text is None:
                continue
            if mode == "raw":
                transcripts[vid] = text
            else:
                # text is a JSON list of segments: [{"start":..,"end":..,"text":"..."}]
                try:
                    segments = json.loads(text)
                    if isinstance(segments, list):
                        joined = " ".join(
                            s.get("text", "") for s in segments if isinstance(s, dict)
                        ).strip()
                        transcripts[vid] = joined
                    else:
                        transcripts[vid] = str(text)
                except Exception:
                    transcripts[vid] = str(text)

    return transcripts


def main() -> int:
    args = parse_args()

    transcripts = _load_transcripts(args.parquet, args.transcript_mode)

    count = 0
    with open(args.jsonl, "r", encoding="utf-8") as f_in, open(
        args.out, "w", encoding="utf-8"
    ) as f_out:
        for line_no, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            video_id = obj.get(args.video_id_key)
            if not video_id:
                continue

            transcript = transcripts.get(video_id)
            gcs_uri = obj.get(args.video_path_key)

            out = {
                "video_id": video_id,
                "gcs_uri": gcs_uri,
                "transcript": transcript,
                "system_prompt": obj.get("system_prompt"),
                "metadata": {
                    "aux_caption": obj.get("aux_caption"),
                    "targets": obj.get("targets"),
                },
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
