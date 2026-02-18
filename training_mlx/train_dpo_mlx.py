from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .mlx_train_config import MLXTrainConfig, ensure_free_disk, resolve_dataset_paths


class PVPMLXPrefixAdapter:
    """Lightweight prefix adapter representation for MLX pipeline metadata/artifacts."""

    def __init__(
        self,
        feature_dim: int,
        num_style_tokens: int = 16,
        hidden_mult: int = 4,
        seed: int = 42,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.num_style_tokens = int(num_style_tokens)
        hidden = max(1, int(self.feature_dim * hidden_mult))
        rng = np.random.default_rng(seed)
        self.style_proj_w = rng.normal(0.0, 0.02, size=(self.feature_dim, hidden)).astype(np.float32)
        self.style_proj_b = np.zeros((hidden,), dtype=np.float32)
        self.target_proj_w = rng.normal(0.0, 0.02, size=(self.feature_dim, hidden)).astype(np.float32)
        self.target_proj_b = np.zeros((hidden,), dtype=np.float32)
        self.out_proj_w = rng.normal(0.0, 0.02, size=(hidden, self.feature_dim)).astype(np.float32)
        self.out_proj_b = np.zeros((self.feature_dim,), dtype=np.float32)

    def encode_prefix(self, style_feats: np.ndarray, target_feats: np.ndarray) -> np.ndarray:
        style_vec = style_feats.reshape(-1, self.feature_dim).mean(axis=0)
        target_vec = target_feats.reshape(-1, self.feature_dim).mean(axis=0)
        hidden = (
            style_vec @ self.style_proj_w
            + self.style_proj_b
            + target_vec @ self.target_proj_w
            + self.target_proj_b
        )
        hidden = np.tanh(hidden)
        prefix = hidden @ self.out_proj_w + self.out_proj_b
        # Duplicate to a small token block to mimic prefix-token injection shape.
        return np.tile(prefix[None, :], (self.num_style_tokens, 1))

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            style_proj_w=self.style_proj_w,
            style_proj_b=self.style_proj_b,
            target_proj_w=self.target_proj_w,
            target_proj_b=self.target_proj_b,
            out_proj_w=self.out_proj_w,
            out_proj_b=self.out_proj_b,
            feature_dim=np.array([self.feature_dim], dtype=np.int32),
            num_style_tokens=np.array([self.num_style_tokens], dtype=np.int32),
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import mlx.core as mx  # type: ignore

        mx.random.seed(seed)
    except Exception:
        pass


def _expand_dataset_paths(dataset_path: str) -> List[Path]:
    paths = resolve_dataset_paths(dataset_path)
    if paths:
        return paths
    # One more compatibility fallback.
    part_matches = sorted(glob.glob(f"{dataset_path}.part*"))
    return [Path(p) for p in part_matches]


def _load_dataset_single(path: Path):
    from datasets import Dataset, load_from_disk

    if path.suffix == ".pt":
        rows = torch.load(path, map_location="cpu")
        if not isinstance(rows, list):
            raise ValueError(f"PT dataset must contain list rows: {path.as_posix()}")
        return Dataset.from_list(rows)
    return load_from_disk(str(path))


def _load_dataset(path_expr: str):
    from datasets import concatenate_datasets

    resolved = _expand_dataset_paths(path_expr)
    if not resolved:
        raise FileNotFoundError(f"Dataset path does not exist: {path_expr}")
    datasets_list = [_load_dataset_single(path) for path in resolved]
    if len(datasets_list) == 1:
        return datasets_list[0], resolved
    return concatenate_datasets(datasets_list), resolved


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _infer_feature_dim(value: Any) -> int:
    arr = _to_numpy(value)
    if arr.ndim < 1:
        raise ValueError(f"Feature tensor must have at least 1 dim, got shape={arr.shape}")
    return int(arr.shape[-1])


def _row_feature_metrics(row: Dict[str, Any]) -> Dict[str, float]:
    style = _to_numpy(row["style_vision_features"]).astype(np.float32, copy=False)
    chosen = _to_numpy(row["chosen_target_vision_features"]).astype(np.float32, copy=False)
    rejected = _to_numpy(row["rejected_target_vision_features"]).astype(np.float32, copy=False)

    chosen_flat = chosen.reshape(-1)
    rejected_flat = rejected.reshape(-1)

    chosen_norm = float(np.linalg.norm(chosen_flat))
    rejected_norm = float(np.linalg.norm(rejected_flat))
    denom = max(chosen_norm * rejected_norm, 1e-8)
    cosine = float(np.dot(chosen_flat, rejected_flat) / denom)

    return {
        "style_mean": float(style.mean()),
        "style_std": float(style.std()),
        "style_norm": float(np.linalg.norm(style.reshape(-1))),
        "chosen_norm": chosen_norm,
        "rejected_norm": rejected_norm,
        "target_norm_gap": float(chosen_norm - rejected_norm),
        "chosen_rejected_cosine": cosine,
    }


def _augment_prompt(prompt: str, metrics: Dict[str, float]) -> str:
    suffix = (
        "[PVP_FEATURES] "
        f"style_mean={metrics['style_mean']:.6f} "
        f"style_std={metrics['style_std']:.6f} "
        f"style_norm={metrics['style_norm']:.6f} "
        f"chosen_norm={metrics['chosen_norm']:.6f} "
        f"rejected_norm={metrics['rejected_norm']:.6f} "
        f"target_norm_gap={metrics['target_norm_gap']:.6f} "
        f"chosen_rejected_cosine={metrics['chosen_rejected_cosine']:.6f}"
    )
    return f"{prompt}\n{suffix}"


def _validate_row_schema(row: Dict[str, Any], index: int) -> Tuple[int, int, int]:
    required = {
        "prompt",
        "chosen",
        "rejected",
        "style_vision_features",
        "chosen_target_vision_features",
        "rejected_target_vision_features",
    }
    missing = sorted(required - set(row.keys()))
    if missing:
        raise ValueError(f"Row {index} is missing required keys: {missing}")

    style_dim = _infer_feature_dim(row["style_vision_features"])
    chosen_dim = _infer_feature_dim(row["chosen_target_vision_features"])
    rejected_dim = _infer_feature_dim(row["rejected_target_vision_features"])
    if chosen_dim != rejected_dim:
        raise ValueError(
            f"Row {index} target feature dims mismatch: chosen={chosen_dim} rejected={rejected_dim}"
        )
    if style_dim != chosen_dim:
        raise ValueError(
            f"Row {index} style/target feature dims mismatch: style={style_dim} target={chosen_dim}"
        )
    return style_dim, chosen_dim, rejected_dim


def _prepare_dpo_jsonl(
    dataset: Any,
    output_jsonl: Path,
    max_rows: int = 0,
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    feature_dim: Optional[int] = None

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(dataset):
            if max_rows > 0 and rows_written >= max_rows:
                break
            _style_dim, chosen_dim, _rejected_dim = _validate_row_schema(row, idx)
            if feature_dim is None:
                feature_dim = chosen_dim
            elif feature_dim != chosen_dim:
                raise ValueError(
                    f"Inconsistent feature dim at row {idx}: expected {feature_dim}, got {chosen_dim}"
                )

            metrics = _row_feature_metrics(row)
            prompt = _augment_prompt(str(row["prompt"]), metrics)
            record = {
                "prompt": prompt,
                "chosen": str(row["chosen"]),
                "rejected": str(row["rejected"]),
                "author_name": str(row.get("author_name") or ""),
                "pair_index": int(row.get("pair_index") or 0),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1

    if rows_written == 0:
        raise ValueError("Prepared dataset is empty after conversion.")
    return {
        "rows_written": rows_written,
        "feature_dim": int(feature_dim or 0),
        "output_jsonl": output_jsonl.as_posix(),
    }


def _save_adapter_manifest(
    cfg: MLXTrainConfig,
    adapter_dir: Path,
    prepared_report: Dict[str, Any],
    backend_status: Dict[str, Any],
) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter = PVPMLXPrefixAdapter(
        feature_dim=int(prepared_report.get("feature_dim") or 1),
        num_style_tokens=16,
        hidden_mult=4,
        seed=cfg.seed,
    )
    adapter.save_npz(adapter_dir / "pvp_prefix_adapter_stub.npz")

    manifest = {
        "model_id": cfg.model_id,
        "dataset_path": cfg.dataset_path,
        "output_dir": cfg.output_dir,
        "feature_dim": prepared_report.get("feature_dim"),
        "rows_written": prepared_report.get("rows_written"),
        "reference_free": cfg.reference_free,
        "beta": cfg.beta,
        "num_train_epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "save_steps": cfg.save_steps,
        "logging_steps": cfg.logging_steps,
        "lora": backend_status.get("lora") or _effective_lora_settings(cfg),
        "trainer_command": backend_status.get("command"),
        "trainer_command_str": backend_status.get("command_str"),
        "backend": backend_status,
        "adapter_weights": (adapter_dir / "pvp_prefix_adapter_stub.npz").as_posix(),
        "timestamp_unix": int(time.time()),
    }
    manifest_path = adapter_dir / "adapter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _format_command_template(tokens: Sequence[str], mapping: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for token in tokens:
        out.append(token.format(**mapping))
    return out


def _run_streaming_command(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n$ {shlex.join(cmd)}\n")
        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()
        logf.write(f"[exit_code={proc.returncode}]\n")
        return int(proc.returncode)


def _effective_lora_settings(cfg: MLXTrainConfig) -> Dict[str, Any]:
    return {
        "use_lora": bool(cfg.use_lora),
        "lora_rank": int(cfg.lora_rank),
        "lora_alpha": int(cfg.lora_alpha),
        "lora_dropout": float(cfg.lora_dropout),
        "lora_target_modules": list(cfg.lora_target_modules),
    }


def _lora_flag_variants(cfg: MLXTrainConfig) -> List[Tuple[str, List[str]]]:
    if not cfg.use_lora:
        return [("disabled", [])]

    rank = str(cfg.lora_rank)
    alpha = str(cfg.lora_alpha)
    dropout = str(cfg.lora_dropout)
    modules = list(cfg.lora_target_modules)

    return [
        (
            "lora_rank_alpha_modules",
            [
                "--lora",
                "--lora-rank",
                rank,
                "--lora-alpha",
                alpha,
                "--lora-dropout",
                dropout,
                "--lora-target-modules",
                *modules,
            ],
        ),
        (
            "use_lora_short_rank_modules",
            [
                "--use-lora",
                "--lora-r",
                rank,
                "--lora-alpha",
                alpha,
                "--lora-dropout",
                dropout,
                "--lora-modules",
                *modules,
            ],
        ),
        (
            "lora_rank_alpha_targets",
            [
                "--lora",
                "--rank",
                rank,
                "--alpha",
                alpha,
                "--dropout",
                dropout,
                "--target-modules",
                *modules,
            ],
        ),
    ]


def _default_backend_commands(cfg: MLXTrainConfig, train_jsonl: Path) -> List[Dict[str, Any]]:
    model = cfg.model_id
    output = cfg.output_dir
    epochs = str(cfg.num_train_epochs)
    lr = str(cfg.learning_rate)
    bs = str(cfg.per_device_train_batch_size)
    ga = str(cfg.gradient_accumulation_steps)
    beta = str(cfg.beta)
    max_len = str(cfg.max_length)
    max_prompt = str(cfg.max_prompt_length)
    save_steps = str(cfg.save_steps)
    logging_steps = str(cfg.logging_steps)
    seed = str(cfg.seed)
    base_args_full = [
        "--model",
        model,
        "--output-dir",
        output,
        "--learning-rate",
        lr,
        "--epochs",
        epochs,
        "--batch-size",
        bs,
        "--grad-accum-steps",
        ga,
        "--beta",
        beta,
        "--max-length",
        max_len,
        "--max-prompt-length",
        max_prompt,
        "--save-steps",
        save_steps,
        "--logging-steps",
        logging_steps,
        "--seed",
        seed,
    ]
    if cfg.reference_free:
        base_args_full.extend(["--reference-free"])

    train_entrypoints: List[Tuple[str, List[str]]] = [
        (
            "mlx_lm_dpo.train_train_file",
            [sys.executable, "-m", "mlx_lm_dpo.train", "--train-file", train_jsonl.as_posix()],
        ),
        (
            "mlx_lm_dpo_train_file",
            [sys.executable, "-m", "mlx_lm_dpo", "--train-file", train_jsonl.as_posix()],
        ),
        (
            "mlx_lm_dpo.train_dataset",
            [sys.executable, "-m", "mlx_lm_dpo.train", "--dataset", train_jsonl.as_posix()],
        ),
    ]
    commands: List[Dict[str, Any]] = []
    for backend_variant, prefix in train_entrypoints:
        for lora_cli_variant, lora_flags in _lora_flag_variants(cfg):
            commands.append(
                {
                    "command": [*prefix, *base_args_full, *lora_flags],
                    "backend_variant": backend_variant,
                    "lora_cli_variant": lora_cli_variant,
                }
            )
    return commands


def _run_mlx_dpo(cfg: MLXTrainConfig, train_jsonl: Path, output_dir: Path) -> Dict[str, Any]:
    log_path = output_dir / "train_dpo_mlx.log"
    lora_settings = _effective_lora_settings(cfg)
    mapping = {
        "model_id": cfg.model_id,
        "train_file": train_jsonl.as_posix(),
        "output_dir": output_dir.as_posix(),
        "learning_rate": str(cfg.learning_rate),
        "num_train_epochs": str(cfg.num_train_epochs),
        "batch_size": str(cfg.per_device_train_batch_size),
        "grad_accum": str(cfg.gradient_accumulation_steps),
        "beta": str(cfg.beta),
        "max_length": str(cfg.max_length),
        "max_prompt_length": str(cfg.max_prompt_length),
        "save_steps": str(cfg.save_steps),
        "logging_steps": str(cfg.logging_steps),
        "seed": str(cfg.seed),
        "use_lora": str(cfg.use_lora).lower(),
        "lora_rank": str(cfg.lora_rank),
        "lora_alpha": str(cfg.lora_alpha),
        "lora_dropout": str(cfg.lora_dropout),
        "lora_target_modules_csv": ",".join(cfg.lora_target_modules),
    }

    commands: List[Dict[str, Any]]
    if cfg.trainer_command:
        base_cmd = _format_command_template(cfg.trainer_command, mapping)
        if cfg.use_lora:
            lora_cli_variant, lora_flags = _lora_flag_variants(cfg)[0]
            base_cmd = [*base_cmd, *lora_flags]
        else:
            lora_cli_variant = "disabled"
        commands = [
            {
                "command": base_cmd,
                "backend_variant": "trainer_command",
                "lora_cli_variant": lora_cli_variant,
            }
        ]
    else:
        commands = _default_backend_commands(cfg, train_jsonl)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(
            json.dumps(
                {
                    "event": "mlx_dpo_launch_config",
                    "lora": lora_settings,
                    "candidate_commands": [
                        {
                            "backend_variant": item["backend_variant"],
                            "lora_cli_variant": item["lora_cli_variant"],
                            "command": shlex.join(item["command"]),
                        }
                        for item in commands
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    failures: List[Dict[str, Any]] = []
    for item in commands:
        cmd = item["command"]
        try:
            rc = _run_streaming_command(cmd, log_path)
        except FileNotFoundError as exc:
            failures.append(
                {
                    "command": cmd,
                    "backend_variant": item["backend_variant"],
                    "lora_cli_variant": item["lora_cli_variant"],
                    "error": f"not_found: {exc}",
                }
            )
            continue
        except Exception as exc:  # pragma: no cover
            failures.append(
                {
                    "command": cmd,
                    "backend_variant": item["backend_variant"],
                    "lora_cli_variant": item["lora_cli_variant"],
                    "error": str(exc),
                }
            )
            continue

        if rc == 0:
            return {
                "status": "ok",
                "command": cmd,
                "command_str": shlex.join(cmd),
                "backend_variant": item["backend_variant"],
                "lora_cli_variant": item["lora_cli_variant"],
                "lora": lora_settings,
                "log_file": log_path.as_posix(),
            }
        failures.append(
            {
                "command": cmd,
                "backend_variant": item["backend_variant"],
                "lora_cli_variant": item["lora_cli_variant"],
                "return_code": rc,
            }
        )

    raise RuntimeError(
        "Failed to run mlx-lm-dpo backend. "
        f"Tried {len(commands)} command(s). Last failures: {failures[-3:]}"
    )


def _run_optional_merge(cfg: MLXTrainConfig, output_dir: Path) -> Dict[str, Any]:
    if not cfg.enable_optional_merge:
        return {"status": "skipped", "reason": "enable_optional_merge=false"}

    mapping = {
        "model_id": cfg.model_id,
        "output_dir": output_dir.as_posix(),
        "adapter_dir": cfg.adapter_dir_path().as_posix(),
        "merged_dir": cfg.merged_dir_path().as_posix(),
    }

    default_cmds = [
        [
            sys.executable,
            "-m",
            "mlx_lm.fuse",
            "--model",
            cfg.model_id,
            "--adapter-path",
            cfg.adapter_dir_path().as_posix(),
            "--save-path",
            cfg.merged_dir_path().as_posix(),
        ],
        [
            sys.executable,
            "-m",
            "mlx_lm.merge",
            "--model",
            cfg.model_id,
            "--adapter-path",
            cfg.adapter_dir_path().as_posix(),
            "--output",
            cfg.merged_dir_path().as_posix(),
        ],
    ]
    commands = (
        [_format_command_template(cfg.merge_command, mapping)] if cfg.merge_command else default_cmds
    )
    log_path = output_dir / "merge_optional.log"
    failures: List[Dict[str, Any]] = []
    for cmd in commands:
        try:
            rc = _run_streaming_command(cmd, log_path)
        except FileNotFoundError as exc:
            failures.append({"command": cmd, "error": f"not_found: {exc}"})
            continue
        except Exception as exc:  # pragma: no cover
            failures.append({"command": cmd, "error": str(exc)})
            continue
        if rc == 0:
            return {"status": "ok", "command": cmd, "log_file": log_path.as_posix()}
        failures.append({"command": cmd, "return_code": rc})

    return {"status": "failed", "failures": failures, "log_file": log_path.as_posix()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PVP DPO on Mac using MLX backend.")
    parser.add_argument("config", help="Path to MLX JSON config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MLXTrainConfig.from_json_file(args.config)
    _seed_everything(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = ensure_free_disk(cfg.min_free_gb, output_dir)
    print(f"free_gb={free_gb}")

    dataset, resolved_paths = _load_dataset(cfg.dataset_path)
    total_rows = len(dataset)
    print(f"Loaded dataset rows={total_rows} from {len(resolved_paths)} path(s).")
    if cfg.max_train_rows > 0:
        total_rows = min(total_rows, cfg.max_train_rows)
        dataset = dataset.select(range(total_rows))
        print(f"Trimmed dataset to max_train_rows={cfg.max_train_rows}")

    prepared_jsonl = (
        Path(cfg.prepared_train_jsonl)
        if cfg.prepared_train_jsonl
        else (output_dir / "prepared" / "train_dpo.jsonl")
    )
    prepared_report = _prepare_dpo_jsonl(dataset, prepared_jsonl, max_rows=cfg.max_train_rows)
    print(json.dumps(prepared_report, ensure_ascii=False))

    backend_status = _run_mlx_dpo(cfg, prepared_jsonl, output_dir)
    print(json.dumps({"backend_status": backend_status}, ensure_ascii=False))

    merge_status = _run_optional_merge(cfg, output_dir)
    print(json.dumps({"merge_status": merge_status}, ensure_ascii=False))

    manifest_path = _save_adapter_manifest(cfg, cfg.adapter_dir_path(), prepared_report, backend_status)
    print(f"adapter_manifest={manifest_path.as_posix()}")


if __name__ == "__main__":
    main()
