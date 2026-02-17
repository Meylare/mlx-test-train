# Kaggle: Safe One-Click Pipeline (Omni Precompute -> Train)

Use these notebook cells in order.  
Goal: catch major failures early, then run full pipeline unattended.

## Cell 1: GPU and runtime check
```bash
!nvidia-smi
```

## Cell 2: Clone repo
```bash
%%bash
set -euo pipefail
cd /kaggle/working
if [ ! -d /kaggle/working/mlx-test-train ]; then
  git clone -b v0_DPO https://github.com/Meylare/mlx-test-train.git
fi
cd /kaggle/working/mlx-test-train
git fetch origin v0_DPO
git checkout v0_DPO
git pull origin v0_DPO
git log -1 --oneline
```

## Cell 3: Define input paths and copy data to writable dir
Attach two Kaggle datasets in UI:
- videos dataset (`.../downloads_shopping/...`)
- manifest dataset (`pvp_pairs_50authors.json`)

```bash
%%bash
set -euo pipefail
cd /kaggle/working/mlx-test-train

VIDEO_ROOT="/kaggle/input/datasets/meylareand/pvp-shopping-videos"
MANIFEST_FILE="/kaggle/input/datasets/meylareand/pvp-pairs-50authors/pvp_pairs_50authors.json"
TARGET="/kaggle/working/mlx-test-train/dataset_50vid_of_prof"

mkdir -p "$TARGET"
cp -r "$VIDEO_ROOT/downloads_shopping" "$TARGET/"
cp "$MANIFEST_FILE" "$TARGET/pvp_pairs_50authors.json"

echo "Data copied."
```

## Cell 4: Preflight data validation (fast fail)
```bash
%%bash
set -euo pipefail
cd /kaggle/working/mlx-test-train

test -f dataset_50vid_of_prof/pvp_pairs_50authors.json
mp4_count="$(find dataset_50vid_of_prof/downloads_shopping -name "*.mp4" | wc -l)"
echo "mp4_count=$mp4_count"
if [ "$mp4_count" -lt 1000 ]; then
  echo "Too few videos copied. Stop."
  exit 2
fi
python - << 'PY'
import json
p = "dataset_50vid_of_prof/pvp_pairs_50authors.json"
d = json.load(open(p, "r", encoding="utf-8"))
authors = d.get("authors", [])
assert len(authors) >= 40, f"authors too few: {len(authors)}"
print("manifest OK, authors:", len(authors))
PY
```

## Cell 5: Install dependencies
```bash
%%bash
set -euo pipefail
pip install -q -U transformers datasets trl unsloth torchvision einops av
apt-get update -y >/dev/null
apt-get install -y ffmpeg >/dev/null
python - << 'PY'
import torch
import transformers
import datasets
import trl
import av
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("trl", trl.__version__)
print("av", av.__version__)
print("cuda available", torch.cuda.is_available(), "gpu_count", torch.cuda.device_count())
PY
```

## Cell 6: Smoke precompute (1 author, very short) to catch Omni errors now
```bash
%%bash
set -euo pipefail
cd /kaggle/working/mlx-test-train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - << 'PY'
import json
from pathlib import Path
src = "dataset_50vid_of_prof/pvp_pairs_50authors.json"
out = "dataset_50vid_of_prof/pvp_pairs_smoke1.json"
d = json.load(open(src, "r", encoding="utf-8"))
base = Path("/kaggle/working/mlx-test-train")

def p_exists(p: str) -> bool:
    if not p:
        return False
    pp = Path(p)
    if pp.exists():
        return True
    pp2 = base / p
    return pp2.exists()

picked = None
for a in d.get("authors", []):
    style_ok = 0
    for sv in (a.get("style_videos") or []):
        if p_exists((sv.get("localPath") or "").strip()):
            style_ok += 1
    pair_ok = 0
    for pair in (a.get("pairs") or []):
        h = (pair.get("hit_video") or {}).get("localPath") or ""
        r = (pair.get("anti_hit_video") or {}).get("localPath") or ""
        if p_exists(h.strip()) and p_exists(r.strip()):
            pair_ok += 1
    if style_ok >= 3 and pair_ok >= 1:
        picked = a
        break

if picked is None:
    raise RuntimeError("Could not find any author for smoke with existing style+pair files.")

d["authors"] = [picked]
json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("written", out, "authors", len(d["authors"]), "picked", picked.get("author", {}).get("name"))
PY

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 python precompute_pvp_vision_features.py \
  --manifest dataset_50vid_of_prof/pvp_pairs_smoke1.json \
  --output-pt dataset_50vid_of_prof/pvp_precomputed_rows_smoke.pt \
  --output-hf-dir /kaggle/working/pvp_precomputed_hf_smoke \
  --summary-json dataset_50vid_of_prof/pvp_precompute_summary_smoke.json \
  --skipped-json dataset_50vid_of_prof/pvp_precompute_skipped_smoke.json \
  --vision-model-name "Qwen/Qwen2.5-Omni-7B" \
  --torch-dtype float16 \
  --device-map auto \
  --image-size 224 \
  --style-seconds 3 --style-fps 0.5 \
  --target-seconds 3 --target-fps 0.5 \
  2>&1 | tee /kaggle/working/precompute_smoke.log

test -f dataset_50vid_of_prof/pvp_precompute_summary_smoke.json
echo "SMOKE PRECOMPUTE OK"
```

## Cell 7: Smoke train (few steps) to catch training errors now
```bash
%%bash
set -euo pipefail
cd /kaggle/working/mlx-test-train

python - << 'PY'
import json
cfg = json.load(open("kaggle_train_precomputed.json", "r", encoding="utf-8"))
cfg["dataset_path"] = "dataset_50vid_of_prof/pvp_precomputed_rows_smoke.pt"
cfg["output_dir"] = "/kaggle/working/outputs/pvp_dpo_smoke"
cfg["num_train_epochs"] = 0.02
cfg["logging_steps"] = 1
cfg["save_steps"] = 999999
cfg["max_length"] = 1024
cfg["max_prompt_length"] = 256
cfg["per_device_train_batch_size"] = 1
cfg["gradient_accumulation_steps"] = 2
cfg["skip_vision_tower"] = True
cfg["vision_hidden_size"] = None
cfg["torch_dtype"] = "float16"
cfg["bnb_4bit_compute_dtype"] = "float16"
cfg["bf16"] = False
cfg["fp16"] = True
cfg["reference_free"] = True
cfg["precompute_ref_log_probs"] = True
json.dump(cfg, open("/kaggle/working/kaggle_train_precomputed_smoke.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("written /kaggle/working/kaggle_train_precomputed_smoke.json")
PY

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_dpo.py /kaggle/working/kaggle_train_precomputed_smoke.json \
  2>&1 | tee /kaggle/working/train_smoke.log

echo "SMOKE TRAIN OK"
```

## Cell 8: Full unattended run (precompute -> train)
Run this only after Cell 6 and Cell 7 pass.
This cell uses RAM-safe chunked precompute (`--flush-every-rows 8`) and then trains from HF parts (`...part*`).

```bash
%%bash
set -euo pipefail
cd /kaggle/working/mlx-test-train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== FULL PRECOMPUTE START $(date) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 python precompute_pvp_vision_features.py \
  --manifest dataset_50vid_of_prof/pvp_pairs_50authors.json \
  --output-pt dataset_50vid_of_prof/pvp_precomputed_rows.pt \
  --output-hf-dir /kaggle/working/pvp_precomputed_hf \
  --summary-json dataset_50vid_of_prof/pvp_precompute_summary.json \
  --skipped-json dataset_50vid_of_prof/pvp_precompute_skipped.json \
  --vision-model-name "Qwen/Qwen2.5-Omni-7B" \
  --torch-dtype float16 \
  --device-map auto \
  --image-size 224 \
  --style-seconds 6 --style-fps 1 \
  --target-seconds 6 --target-fps 1 \
  --flush-every-rows 8 \
  2>&1 | tee /kaggle/working/precompute_full.log

echo "=== BUILD TRAIN CONFIG $(date) ==="
python - << 'PY'
import json
cfg = json.load(open("kaggle_train_precomputed.json", "r", encoding="utf-8"))
cfg["dataset_path"] = "/kaggle/working/pvp_precomputed_hf.part*"
cfg["output_dir"] = "/kaggle/working/outputs/pvp_dpo_precomputed"
cfg["skip_vision_tower"] = True
cfg["vision_hidden_size"] = None
cfg["torch_dtype"] = "float16"
cfg["bnb_4bit_compute_dtype"] = "float16"
cfg["bf16"] = False
cfg["fp16"] = True
cfg["reference_free"] = True
cfg["precompute_ref_log_probs"] = True
json.dump(cfg, open("/kaggle/working/kaggle_train_precomputed_runtime.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("written /kaggle/working/kaggle_train_precomputed_runtime.json")
PY

echo "=== FULL TRAIN START $(date) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 \
  train_dpo.py /kaggle/working/kaggle_train_precomputed_runtime.json \
  2>&1 | tee /kaggle/working/train_full.log

echo "=== DONE $(date) ==="
```

## Cell 9: Final artifacts check
```bash
!ls -lah /kaggle/working/outputs/pvp_dpo_precomputed
!python - << 'PY'
import json
p = "/kaggle/working/mlx-test-train/dataset_50vid_of_prof/pvp_precompute_summary.json"
d = json.load(open(p, "r", encoding="utf-8"))
print("rows_total:", d.get("rows_total"))
print("max_rows_buffered:", d.get("max_rows_buffered"))
print("flush_every_rows:", d.get("flush_every_rows"))
print("hf_parts:", len(d.get("output_hf_parts", []) or []))
PY
!tail -n 80 /kaggle/working/precompute_full.log
!tail -n 80 /kaggle/working/train_full.log
```
