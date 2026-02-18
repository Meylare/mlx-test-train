from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any, Dict, List

import torch


def _infer_kind(paths: List[Path]) -> str:
    if not paths:
        raise ValueError("No inputs were provided.")
    first = paths[0]
    if first.is_file() and first.suffix == ".pt":
        return "pt"
    if first.is_dir():
        return "hf"
    raise ValueError(f"Could not infer dataset kind from: {first.as_posix()}")


def _merge_pt(paths: List[Path], output: Path) -> Dict[str, Any]:
    merged_rows: List[Dict[str, Any]] = []
    for path in paths:
        rows = torch.load(path, map_location="cpu")
        if not isinstance(rows, list):
            raise ValueError(f"PT shard must contain list rows: {path.as_posix()}")
        merged_rows.extend(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_rows, output)
    return {
        "kind": "pt",
        "rows_total": len(merged_rows),
        "inputs": [p.as_posix() for p in paths],
        "output": output.as_posix(),
    }


def _merge_hf(paths: List[Path], output: Path) -> Dict[str, Any]:
    from datasets import concatenate_datasets, load_from_disk

    datasets = [load_from_disk(str(path)) for path in paths]
    merged = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(str(output))
    return {
        "kind": "hf",
        "rows_total": len(merged),
        "inputs": [p.as_posix() for p in paths],
        "output": output.as_posix(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge precomputed dataset shards.")
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Input glob, for example: dataset_50vid_of_prof/pvp_precomputed_hf.part*",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset path. For PT use .pt, for HF use directory path.",
    )
    parser.add_argument(
        "--kind",
        choices=["auto", "pt", "hf"],
        default="auto",
        help="Dataset kind. auto infers from the first matched input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = sorted(Path(p) for p in glob.glob(args.input_glob))
    if not inputs:
        raise FileNotFoundError(f"No inputs found for glob: {args.input_glob}")

    kind = args.kind if args.kind != "auto" else _infer_kind(inputs)
    output = Path(args.output)

    if kind == "pt":
        report = _merge_pt(inputs, output)
    else:
        report = _merge_hf(inputs, output)

    print(report)


if __name__ == "__main__":
    main()
