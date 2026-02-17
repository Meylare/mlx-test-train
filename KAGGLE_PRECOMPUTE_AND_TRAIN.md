# Kaggle: PVP Precompute + DPO Train

## 1) Runtime
- GPU: `2x T4`
- Python: `3.10+`
- Recommended: Internet ON for model downloads.

## 2) Prepare workspace
```bash
cd /kaggle/working
cp -r /kaggle/input/<your-code-dataset>/Mac_qwen-train .
cd Mac_qwen-train
```

## 3) Install deps
```bash
pip install -U transformers datasets trl unsloth torchvision einops
```

If `torchvision` video backend fails in Kaggle, install ffmpeg:
```bash
apt-get update && apt-get install -y ffmpeg
```

## 4) Precompute vision features (before Resampler) on 2x T4
Run sharded precompute with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
  precompute_pvp_vision_features.py \
  --use-torchrun-sharding \
  --manifest dataset_50vid_of_prof/pvp_pairs_50authors.json \
  --output-pt dataset_50vid_of_prof/pvp_precomputed_rows.pt \
  --output-hf-dir /kaggle/working/pvp_precomputed_hf \
  --summary-json dataset_50vid_of_prof/pvp_precompute_summary.json \
  --skipped-json dataset_50vid_of_prof/pvp_precompute_skipped.json \
  --image-size 224 \
  --style-seconds 10 --style-fps 1 \
  --target-seconds 10 --target-fps 1
```

This creates shard files (suffix `.shardXXofYY`). Merge them:

```bash
python merge_precomputed_shards.py \
  --pt-pattern "dataset_50vid_of_prof/pvp_precomputed_rows.shard*.pt" \
  --skipped-pattern "dataset_50vid_of_prof/pvp_precompute_skipped.shard*.json" \
  --summary-pattern "dataset_50vid_of_prof/pvp_precompute_summary.shard*.json" \
  --hf-pattern "/kaggle/working/pvp_precomputed_hf.shard*" \
  --output-pt dataset_50vid_of_prof/pvp_precomputed_rows.pt \
  --output-skipped-json dataset_50vid_of_prof/pvp_precompute_skipped.json \
  --output-summary-json dataset_50vid_of_prof/pvp_precompute_summary.json \
  --output-hf-dir /kaggle/working/pvp_precomputed_hf
```

## 5) Train DPO on precomputed features (2x T4, DDP)
1. Open `kaggle_train_precomputed.json`
2. Confirm `"dataset_path": "/kaggle/working/pvp_precomputed_hf"`
3. Launch with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 \
  train_dpo.py kaggle_train_precomputed.json
```

Outputs:
- `/kaggle/working/outputs/pvp_dpo_precomputed`
- `resampler.pt` and `projector.pt` saved with model.

## 6) What is trained
- `VisionTower`: frozen
- `PerceiverResampler`: trainable
- `Projector`: trainable
- `LLM LoRA adapters`: trainable

This is the intended hybrid mode: precompute only frozen vision features, keep style compression trainable.
