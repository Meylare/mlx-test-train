from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge precompute shard outputs.")
    parser.add_argument(
        "--pt-pattern",
        default="dataset_50vid_of_prof/pvp_precomputed_rows.shard*.pt",
        help="Glob pattern for shard .pt files.",
    )
    parser.add_argument(
        "--skipped-pattern",
        default="dataset_50vid_of_prof/pvp_precompute_skipped.shard*.json",
        help="Glob pattern for shard skipped json files.",
    )
    parser.add_argument(
        "--summary-pattern",
        default="dataset_50vid_of_prof/pvp_precompute_summary.shard*.json",
        help="Glob pattern for shard summary json files.",
    )
    parser.add_argument(
        "--hf-pattern",
        default="/kaggle/working/pvp_precomputed_hf.shard*",
        help="Glob pattern for shard HF dataset dirs.",
    )
    parser.add_argument(
        "--output-pt",
        default="dataset_50vid_of_prof/pvp_precomputed_rows.pt",
        help="Merged .pt output path.",
    )
    parser.add_argument(
        "--output-skipped-json",
        default="dataset_50vid_of_prof/pvp_precompute_skipped.json",
        help="Merged skipped json output path.",
    )
    parser.add_argument(
        "--output-summary-json",
        default="dataset_50vid_of_prof/pvp_precompute_summary.json",
        help="Merged summary json output path.",
    )
    parser.add_argument(
        "--output-hf-dir",
        default="/kaggle/working/pvp_precomputed_hf",
        help="Merged HF dataset directory output.",
    )
    args = parser.parse_args()

    pt_files = sorted(Path(p) for p in glob.glob(args.pt_pattern))
    if not pt_files:
        raise FileNotFoundError(f"No shard pt files found by pattern: {args.pt_pattern}")

    merged_rows: List[Dict[str, Any]] = []
    for path in pt_files:
        rows = torch.load(path, map_location="cpu")
        if not isinstance(rows, list):
            raise ValueError(f"Shard file {path.as_posix()} does not contain list rows.")
        merged_rows.extend(rows)

    output_pt = Path(args.output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_rows, output_pt)

    skipped_files = sorted(Path(p) for p in glob.glob(args.skipped_pattern))
    merged_skipped: List[Dict[str, Any]] = []
    for path in skipped_files:
        merged_skipped.extend(_load_json_list(path))
    output_skipped = Path(args.output_skipped_json)
    output_skipped.parent.mkdir(parents=True, exist_ok=True)
    output_skipped.write_text(json.dumps(merged_skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_files = sorted(Path(p) for p in glob.glob(args.summary_pattern))
    shard_summaries: List[Dict[str, Any]] = []
    for path in summary_files:
        data = _load_json_list(path)
        if data:
            shard_summaries.append(data[0])

    merged_summary = {
        "rows_total": len(merged_rows),
        "pt_shards": [p.as_posix() for p in pt_files],
        "skipped_total": len(merged_skipped),
        "skipped_shards": [p.as_posix() for p in skipped_files],
        "summary_shards": [p.as_posix() for p in summary_files],
        "shard_summaries": shard_summaries,
        "output_pt": output_pt.as_posix(),
        "output_skipped_json": output_skipped.as_posix(),
    }
    output_summary = Path(args.output_summary_json)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(merged_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    hf_dirs = sorted(Path(p) for p in glob.glob(args.hf_pattern))
    if hf_dirs:
        from datasets import concatenate_datasets, load_from_disk

        datasets_list = [load_from_disk(str(p)) for p in hf_dirs]
        merged_hf = datasets_list[0] if len(datasets_list) == 1 else concatenate_datasets(datasets_list)
        output_hf = Path(args.output_hf_dir)
        output_hf.mkdir(parents=True, exist_ok=True)
        merged_hf.save_to_disk(str(output_hf))
        merged_summary["output_hf_dir"] = output_hf.as_posix()
        output_summary.write_text(json.dumps(merged_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Merged HF dataset rows: {len(merged_hf)} -> {output_hf.as_posix()}")

    print(json.dumps(merged_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
