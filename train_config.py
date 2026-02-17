from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, List, Optional

import torch


def _parse_torch_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Unsupported torch dtype value type: {type(value).__name__}.")

    normalized = value.strip().lower().replace("torch.", "")
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if normalized not in mapping:
        supported = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported torch dtype '{value}'. Supported values: {supported}.")
    return mapping[normalized]


def _parse_fixed_image_size(value: Any) -> Optional[tuple[int, int]]:
    if value is None:
        return None

    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return value

    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])

    raw: str
    if isinstance(value, str):
        raw = value
    elif isinstance(value, tuple) and all(isinstance(v, str) for v in value):
        raw = "".join(value)
    else:
        raise TypeError(f"Unsupported fixed_image_size value type: {type(value).__name__}.")

    dims = [int(part) for part in re.findall(r"\d+", raw)]
    if len(dims) == 1:
        return dims[0], dims[0]
    if len(dims) == 2:
        return dims[0], dims[1]
    raise ValueError(
        f"Invalid fixed_image_size '{value}'. Use one integer (square) or two integers like 448,448."
    )


def _parse_device_map(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, int)):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Unsupported device_map value type: {type(value).__name__}.")
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    if normalized.isdigit():
        return int(normalized)
    return value


@dataclass
class PVPModelConfig:
    vision_model_name: str = "Qwen/Qwen2.5-Omni-7B"
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 4096
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    device_map: Any = "auto"
    num_style_tokens: int = 16
    perceiver_depth: int = 6
    perceiver_heads: int = 8
    projector_hidden_mult: int = 4
    vision_accepts_video: bool = False
    vision_hidden_size: Optional[int] = None
    fixed_image_size: Optional[str] = "448,448"

    def __post_init__(self) -> None:
        self.torch_dtype = _parse_torch_dtype(self.torch_dtype)
        self.bnb_4bit_compute_dtype = _parse_torch_dtype(self.bnb_4bit_compute_dtype)
        self.device_map = _parse_device_map(self.device_map)
        self.fixed_image_size = _parse_fixed_image_size(self.fixed_image_size)


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_gradient_checkpointing: str = "unsloth"


@dataclass
class DPOTrainingConfig:
    output_dir: str = "outputs/pvp_dpo"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    num_train_epochs: float = 1.0
    max_length: int = 4096
    max_prompt_length: int = 2048
    beta: float = 0.1
    logging_steps: int = 10
    save_steps: int = 250
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch_fused"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    ddp_find_unused_parameters: Optional[bool] = False
    remove_unused_columns: bool = False
    report_to: Optional[str] = None
    seed: int = 42


def build_pvp_model(
    model_cfg: PVPModelConfig,
    lora_cfg: Optional[LoRAConfig] = None,
) -> "PVPModel":
    from modeling_pvp import PVPModel

    lora_r = lora_cfg.r if lora_cfg is not None else None
    lora_alpha = lora_cfg.lora_alpha if lora_cfg is not None else 32
    lora_dropout = lora_cfg.lora_dropout if lora_cfg is not None else 0.05
    lora_target_modules = (
        lora_cfg.target_modules if lora_cfg is not None else None
    )
    lora_use_gradient_checkpointing = (
        lora_cfg.use_gradient_checkpointing if lora_cfg is not None else "unsloth"
    )

    return PVPModel.from_pretrained(
        vision_model_name=model_cfg.vision_model_name,
        llm_model_name=model_cfg.llm_model_name,
        max_seq_length=model_cfg.max_seq_length,
        attn_implementation=model_cfg.attn_implementation,
        torch_dtype=model_cfg.torch_dtype,
        load_in_4bit=model_cfg.load_in_4bit,
        bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=model_cfg.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
        device_map=model_cfg.device_map,
        num_style_tokens=model_cfg.num_style_tokens,
        perceiver_depth=model_cfg.perceiver_depth,
        perceiver_heads=model_cfg.perceiver_heads,
        projector_hidden_mult=model_cfg.projector_hidden_mult,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_use_gradient_checkpointing=lora_use_gradient_checkpointing,
        vision_accepts_video=model_cfg.vision_accepts_video,
        vision_hidden_size=model_cfg.vision_hidden_size,
        fixed_image_size=model_cfg.fixed_image_size,
    )


def build_dpo_config(cfg: DPOTrainingConfig) -> "DPOConfig":
    from trl import DPOConfig

    kwargs = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        beta=cfg.beta,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        ddp_find_unused_parameters=cfg.ddp_find_unused_parameters,
        remove_unused_columns=cfg.remove_unused_columns,
        report_to=cfg.report_to,
        seed=cfg.seed,
    )
    return DPOConfig(**kwargs)
