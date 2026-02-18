from __future__ import annotations

from dataclasses import asdict, dataclass
import glob
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MLXTrainConfig:
    # Required fields from plan.
    model_id: str
    dataset_path: str
    output_dir: str
    num_train_epochs: float
    learning_rate: float
    beta: float
    max_length: int
    max_prompt_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    reference_free: bool
    save_steps: int
    logging_steps: int
    report_to: str
    seed: int
    enable_optional_merge: bool
    strict_manifest_validation: bool
    min_free_gb: int

    # Optional operational fields.
    dpo_backend: str = "auto"
    trainer_command: Optional[List[str]] = None
    merge_command: Optional[List[str]] = None
    max_train_rows: int = 0
    prepared_train_jsonl: Optional[str] = None
    adapter_subdir: str = "adapter"
    merged_subdir: str = "merged"

    @classmethod
    def from_json_file(cls, path: str | Path) -> "MLXTrainConfig":
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path.as_posix()}")
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Top-level config must be a JSON object.")

        allowed = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        unknown = sorted(set(raw.keys()) - allowed)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")

        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not str(self.model_id).strip():
            raise ValueError("model_id must be non-empty.")
        if not str(self.dataset_path).strip():
            raise ValueError("dataset_path must be non-empty.")
        if not str(self.output_dir).strip():
            raise ValueError("output_dir must be non-empty.")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be > 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.beta < 0:
            raise ValueError("beta must be >= 0.")
        if self.max_length <= 0 or self.max_prompt_length <= 0:
            raise ValueError("max_length and max_prompt_length must be > 0.")
        if self.max_prompt_length >= self.max_length:
            raise ValueError("max_prompt_length must be smaller than max_length.")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be > 0.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0.")
        if self.save_steps <= 0 or self.logging_steps <= 0:
            raise ValueError("save_steps and logging_steps must be > 0.")
        if self.min_free_gb < 0:
            raise ValueError("min_free_gb must be >= 0.")
        if self.max_train_rows < 0:
            raise ValueError("max_train_rows must be >= 0.")
        if self.trainer_command is not None and not isinstance(self.trainer_command, list):
            raise ValueError("trainer_command must be a list of string tokens or null.")
        if self.merge_command is not None and not isinstance(self.merge_command, list):
            raise ValueError("merge_command must be a list of string tokens or null.")
        if self.report_to.strip().lower() != "none":
            # The plan intentionally disables integrations by default for overnight reliability.
            raise ValueError("report_to must be 'none' for this MLX pipeline.")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def output_dir_path(self) -> Path:
        return Path(self.output_dir)

    def adapter_dir_path(self) -> Path:
        return self.output_dir_path() / self.adapter_subdir

    def merged_dir_path(self) -> Path:
        return self.output_dir_path() / self.merged_subdir


def resolve_dataset_paths(dataset_path: str) -> List[Path]:
    value = str(dataset_path).strip()
    if not value:
        return []

    if "," in value:
        parts = [p.strip() for p in value.split(",") if p.strip()]
    elif any(ch in value for ch in "*?[]"):
        parts = sorted(glob.glob(value))
    elif Path(value).exists():
        parts = [value]
    else:
        part_matches = sorted(glob.glob(f"{value}.part*"))
        parts = part_matches

    return [Path(p) for p in parts]


def ensure_free_disk(min_free_gb: int, at_path: str | Path) -> int:
    probe = Path(at_path)
    if not probe.exists():
        probe = probe.parent if probe.parent.exists() else Path(".")
    usage = shutil.disk_usage(str(probe))
    free_gb = int(usage.free / (1024 ** 3))
    if free_gb < int(min_free_gb):
        raise RuntimeError(
            f"Not enough free disk space: free={free_gb}GB, required>={min_free_gb}GB "
            f"at {probe.as_posix()}"
        )
    return free_gb
