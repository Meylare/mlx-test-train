# PVP Mac MLX Pipeline (`training_mlx`)

This directory adds a parallel local pipeline for macOS + MLX and keeps the original `training/` torch pipeline untouched.

## What Is Included
- `precompute_pvp_vision_features_mac.py`: local hybrid precompute (torch + transformers), same row schema as old pipeline.
- `train_dpo_mlx.py`: MLX DPO launcher with dataset conversion, adapter artifact manifest, optional merge (best-effort).
- `merge_precomputed_shards.py`: merge `.part*` shards (PT or HF datasets).
- `mlx_train_config.py`: strict config schema and disk/path validation.
- `configs/smoke.json`, `configs/full_template.json`: templates for smoke/full.
- `run_smoke_mac.sh`, `run_full_mac.sh`: end-to-end scripts.

## Row Schema Compatibility
Precompute rows keep the existing keys:
- `prompt`
- `chosen`
- `rejected`
- `style_vision_features`
- `chosen_target_vision_features`
- `rejected_target_vision_features`
- `author_name`
- `pair_index`

## Install
```bash
python3 -m pip install -r training_mlx/requirements_mac_mlx.txt
```

## Config Contract (JSON)
Required fields:
- `model_id`
- `dataset_path`
- `output_dir`
- `num_train_epochs`
- `learning_rate`
- `beta`
- `max_length`
- `max_prompt_length`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `reference_free`
- `save_steps`
- `logging_steps`
- `report_to`
- `seed`
- `enable_optional_merge`
- `strict_manifest_validation`
- `min_free_gb`

Additional optional fields:
- `dpo_backend`
- `trainer_command`
- `merge_command`
- `max_train_rows`
- `prepared_train_jsonl`
- `adapter_subdir`
- `merged_subdir`

## CLI Interfaces
Precompute:
```bash
python -m training_mlx.precompute_pvp_vision_features_mac \
  --manifest dataset_50vid_of_prof/pvp_pairs_50authors.json \
  --output-hf-dir dataset_50vid_of_prof/pvp_precomputed_hf \
  --summary-json dataset_50vid_of_prof/pvp_precompute_summary.json
```

Train:
```bash
python -m training_mlx.train_dpo_mlx training_mlx/configs/full_template.json
```

Merge shards:
```bash
python -m training_mlx.merge_precomputed_shards \
  --input-glob "dataset_50vid_of_prof/pvp_precomputed_hf.part*" \
  --output dataset_50vid_of_prof/pvp_precomputed_hf_merged
```

## Runbook
Smoke:
```bash
bash training_mlx/run_smoke_mac.sh
```

Full:
```bash
bash training_mlx/run_full_mac.sh
```

Background full run:
```bash
nohup bash training_mlx/run_full_mac.sh > training_mlx/logs/full.out 2>&1 &
tail -f training_mlx/logs/full.out
```

## Adapter vs Merged
- **Adapter** is the required artifact (compact, robust).
- **Merged** is optional and best-effort.
- Adapter compatibility requires the same base model family/tokenizer/revision.  
  Q4/full can be compatible only when architecture + tokenizer + revision match.

## Reliability Guards
- Strict manifest path checks (`--strict-manifest-validation`).
- Disk guard (`min_free_gb`) before work and during precompute flush.
- Chunked flush (`flush_every_rows`) to cap RAM usage.
- Optional merge failure does not fail training success.

## Test Scenarios
1. Manifest validation catches missing `localPath`.
2. Smoke precompute produces `rows_total > 0`.
3. Dataset loading works from `*.part*` globs.
4. Smoke train runs at least one backend invocation.
5. Resume strategy is backend-driven via output directory reuse.
6. Optional merge failure is non-fatal and adapter manifest is still written.
