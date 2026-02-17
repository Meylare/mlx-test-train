from __future__ import annotations

import glob
import inspect
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import HfArgumentParser, set_seed
from trl import DPOTrainer

from train_config import (
    DPOTrainingConfig,
    LoRAConfig,
    PVPModelConfig,
    build_dpo_config,
    build_pvp_model,
)


def _configure_distributed_runtime(
    model_args: PVPModelConfig,
    training_args: DPOTrainingConfig,
) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 1:
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # In DDP each process must own exactly one GPU.
    if model_args.device_map in (None, "auto"):
        model_args.device_map = local_rank

    if training_args.ddp_find_unused_parameters is None:
        training_args.ddp_find_unused_parameters = False

    if rank == 0:
        print(
            f"DDP enabled: world_size={world_size}, "
            f"device_map={model_args.device_map}, "
            f"ddp_find_unused_parameters={training_args.ddp_find_unused_parameters}"
        )


@dataclass
class ScriptConfig:
    dataset_path: Optional[str] = None
    use_dummy_dataset: bool = True
    num_dummy_samples: int = 20


@dataclass
class PVPDataCollator:
    tokenizer: Any

    def _pad_1d(self, values: List[Any], pad_value: int) -> torch.Tensor:
        tensors = [torch.as_tensor(v, dtype=torch.long) for v in values]
        max_len = max(t.numel() for t in tensors)
        out = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
        for i, tensor in enumerate(tensors):
            out[i, : tensor.numel()] = tensor
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}

        keys = features[0].keys()
        for key in keys:
            values = [f[key] for f in features]
            if key in {
                "style_pixel_values",
                "chosen_target_pixel_values",
                "rejected_target_pixel_values",
                "style_vision_features",
                "chosen_target_vision_features",
                "rejected_target_vision_features",
            }:
                batch[key] = torch.stack([torch.as_tensor(v) for v in values])
                continue
            if key.endswith("_input_ids"):
                batch[key] = self._pad_1d(values, pad_value=self.tokenizer.pad_token_id)
                continue
            if key.endswith("_attention_mask"):
                batch[key] = self._pad_1d(values, pad_value=0)
                continue
            if key.endswith("_labels"):
                batch[key] = self._pad_1d(values, pad_value=-100)
                continue
            batch[key] = values

        return batch


def create_dummy_dataset(
    num_samples: int = 10,
    image_size: int | tuple[int, int] = 448,
    style_videos: int = 20,
    frames_per_video: int = 2,
) -> Dataset:
    if isinstance(image_size, int):
        height = image_size
        width = image_size
    else:
        height, width = image_size

    data = []
    for i in range(num_samples):
        data.append(
            {
                "prompt": f"Analyze virality potential from creator style. Sample {i}",
                "chosen": "The hook in the first seconds boosts retention and shareability.",
                "rejected": "This is average and has no clear retention strategy.",
                "style_pixel_values": torch.randn(
                    style_videos, frames_per_video, 3, height, width
                ),
                "chosen_target_pixel_values": torch.randn(
                    frames_per_video, 3, height, width
                ),
                "rejected_target_pixel_values": torch.randn(
                    frames_per_video, 3, height, width
                ),
            }
        )
    return Dataset.from_list(data)


def _contains_glob(path: str) -> bool:
    return any(ch in path for ch in "*?[]")


def _expand_dataset_paths(dataset_path: str) -> List[str]:
    path = str(dataset_path).strip()
    if not path:
        return []

    if "," in path:
        out = [p.strip() for p in path.split(",") if p.strip()]
        return out

    if _contains_glob(path):
        return sorted(glob.glob(path))

    if os.path.exists(path):
        return [path]

    part_matches = sorted(glob.glob(f"{path}.part*"))
    if part_matches:
        return part_matches

    return []


def _load_dataset_single(path: str) -> Dataset:
    if path.endswith(".pt"):
        rows = torch.load(path, map_location="cpu")
        if not isinstance(rows, list):
            raise ValueError(f"PT dataset must be a list of row dicts: {path}")
        return Dataset.from_list(rows)
    return load_from_disk(path)


def load_train_dataset(script_cfg: ScriptConfig, model_cfg: PVPModelConfig) -> Dataset:
    if script_cfg.use_dummy_dataset:
        image_size = model_cfg.fixed_image_size if model_cfg.fixed_image_size is not None else (448, 448)
        return create_dummy_dataset(
            num_samples=script_cfg.num_dummy_samples,
            image_size=image_size,
        )

    if script_cfg.dataset_path is None:
        raise ValueError(
            "Provide --dataset_path when --use_dummy_dataset=False."
        )

    resolved_paths = _expand_dataset_paths(script_cfg.dataset_path)
    if not resolved_paths:
        raise FileNotFoundError(f"Dataset path does not exist: {script_cfg.dataset_path}")

    datasets_list = [_load_dataset_single(path) for path in resolved_paths]
    if len(datasets_list) == 1:
        return datasets_list[0]

    print(f"Loading dataset from {len(datasets_list)} parts.")
    return concatenate_datasets(datasets_list)


def _infer_feature_dim(value: Any) -> Optional[int]:
    try:
        tensor = torch.as_tensor(value)
    except Exception:
        return None
    if tensor.ndim < 1:
        return None
    last_dim = int(tensor.shape[-1])
    if last_dim <= 0:
        return None
    return last_dim


def _configure_precomputed_mode_from_dataset(
    model_args: PVPModelConfig,
    train_dataset: Dataset,
) -> None:
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    sample = train_dataset[0]
    feature_keys = [
        "style_vision_features",
        "chosen_target_vision_features",
        "rejected_target_vision_features",
    ]
    present = [k for k in feature_keys if k in sample]

    if model_args.skip_vision_tower:
        if not present:
            raise ValueError(
                "skip_vision_tower=True but dataset has no precomputed vision keys: "
                f"{feature_keys}. Use precomputed dataset or disable skip_vision_tower."
            )

        inferred_dims: List[int] = []
        for key in present:
            dim = _infer_feature_dim(sample[key])
            if dim is not None:
                inferred_dims.append(dim)
        if not inferred_dims:
            raise ValueError(
                "Could not infer vision feature dim from precomputed fields. "
                "Check dataset serialization format."
            )

        inferred_dim = inferred_dims[0]
        if any(x != inferred_dim for x in inferred_dims):
            raise ValueError(
                f"Inconsistent precomputed feature dims in sample: {inferred_dims}. "
                "Expected same last dim D for all vision fields."
            )

        if model_args.vision_hidden_size is None:
            model_args.vision_hidden_size = inferred_dim
            print(f"Inferred vision_hidden_size={inferred_dim} from precomputed dataset.")
        elif int(model_args.vision_hidden_size) != inferred_dim:
            raise ValueError(
                f"vision_hidden_size={model_args.vision_hidden_size} does not match "
                f"dataset feature dim D={inferred_dim}."
            )


class MultimodalDPOTrainer(DPOTrainer):
    def concatenated_inputs(self, batch: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        concatenated_batch = super().concatenated_inputs(batch, *args, **kwargs)

        chosen_target = batch.get("chosen_target_pixel_values")
        rejected_target = batch.get("rejected_target_pixel_values")
        if chosen_target is not None and rejected_target is not None:
            concatenated_batch["concatenated_target_pixel_values"] = torch.cat(
                [chosen_target, rejected_target],
                dim=0,
            )

        style = batch.get("style_pixel_values")
        if style is not None:
            concatenated_batch["concatenated_style_pixel_values"] = torch.cat(
                [style, style],
                dim=0,
            )

        chosen_target_feats = batch.get("chosen_target_vision_features")
        rejected_target_feats = batch.get("rejected_target_vision_features")
        if chosen_target_feats is not None and rejected_target_feats is not None:
            concatenated_batch["concatenated_target_vision_features"] = torch.cat(
                [chosen_target_feats, rejected_target_feats],
                dim=0,
            )

        style_feats = batch.get("style_vision_features")
        if style_feats is not None:
            concatenated_batch["concatenated_style_vision_features"] = torch.cat(
                [style_feats, style_feats],
                dim=0,
            )

        return concatenated_batch


def build_trainer(
    model: torch.nn.Module,
    tokenizer: Any,
    dpo_config: Any,
    train_dataset: Dataset,
) -> DPOTrainer:
    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        data_collator=PVPDataCollator(tokenizer),
    )

    sig = inspect.signature(DPOTrainer.__init__)
    params = sig.parameters

    if "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    if "max_length" in params:
        trainer_kwargs["max_length"] = dpo_config.max_length
    if "max_prompt_length" in params:
        trainer_kwargs["max_prompt_length"] = dpo_config.max_prompt_length

    return MultimodalDPOTrainer(**trainer_kwargs)


def main() -> None:
    parser = HfArgumentParser((PVPModelConfig, LoRAConfig, DPOTrainingConfig, ScriptConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, lora_args, training_args, script_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, lora_args, training_args, script_args = parser.parse_args_into_dataclasses()

    _configure_distributed_runtime(model_args, training_args)
    set_seed(training_args.seed)
    dpo_config = build_dpo_config(training_args)
    dpo_config.remove_unused_columns = False

    train_dataset = load_train_dataset(script_args, model_args)
    _configure_precomputed_mode_from_dataset(model_args, train_dataset)

    model = build_pvp_model(model_args, lora_args)
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    trainer = build_trainer(model, tokenizer, dpo_config, train_dataset)

    trainer.train()

    output_dir = dpo_config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    torch.save(model.resampler.state_dict(), os.path.join(output_dir, "resampler.pt"))
    torch.save(model.projector.state_dict(), os.path.join(output_dir, "projector.pt"))


if __name__ == "__main__":
    main()
