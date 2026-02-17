# Kaggle: Stable Omni Precompute -> DPO Train

Minimal set of cells for a reliable overnight run.

## Cell 1: Runtime + disk check
```bash
!nvidia-smi
!df -h /kaggle/working
!du -h --max-depth=1 /kaggle/working | sort -h | tail -n 20
```

## Cell 2: Clone/update repo
```bash
%%bash
set -euo pipefail
cd /kaggle/working

# If repo exists and is dirty from prior experiments, prefer fresh clone.
if [ -d /kaggle/working/mlx-test-train/.git ]; then
  cd /kaggle/working/mlx-test-train
  if [ -n "$(git status --porcelain)" ]; then
    cd /kaggle/working
    mv /kaggle/working/mlx-test-train /kaggle/working/mlx-test-train.bak.$(date +%s)
  fi
fi

if [ ! -d /kaggle/working/mlx-test-train/.git ]; then
  git clone -b v0_DPO https://github.com/Meylare/mlx-test-train.git
fi

cd /kaggle/working/mlx-test-train
git fetch origin v0_DPO
git checkout v0_DPO
git pull --ff-only origin v0_DPO
git log -1 --oneline
```

## Cell 3: Copy datasets into writable workspace
Attach datasets in Kaggle UI first.

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

test -f "$TARGET/pvp_pairs_50authors.json"
mp4_count="$(find "$TARGET/downloads_shopping" -name "*.mp4" | wc -l)"
echo "mp4_count=$mp4_count"
```

## Cell 4: Install deps
```bash
%%bash
set -euo pipefail
pip install -q -U transformers datasets trl unsloth torchvision einops av
apt-get update -y >/dev/null
apt-get install -y ffmpeg >/dev/null
python - << 'PY'
import torch, transformers, datasets, trl, av
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('trl', trl.__version__)
print('av', av.__version__)
print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())
PY
```

## Cell 5: Disk cleanup (safe)
Removes only smoke artifacts/logs/caches from previous attempts.

```bash
%%bash
set -euo pipefail

rm -rf /kaggle/working/pvp_precomputed_hf_smoke*
rm -rf /kaggle/working/outputs/pvp_dpo_smoke*
rm -rf /kaggle/working/mlx-test-train/dataset_50vid_of_prof/*smoke*
rm -f  /kaggle/working/*smoke*.log /kaggle/working/precompute_full.log /kaggle/working/train_full.log

# Optional: huge HF cache cleanup (uncomment only if disk is still low)
# rm -rf /root/.cache/huggingface/hub

df -h /kaggle/working
```

## Cell 6: Smoke precompute (fast)
```bash
%%bash
set -euo pipefail
cd /kaggle/working/mlx-test-train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - << 'PY'
import json
from pathlib import Path

src = 'dataset_50vid_of_prof/pvp_pairs_50authors.json'
out = 'dataset_50vid_of_prof/pvp_pairs_smoke1.json'
d = json.load(open(src, 'r', encoding='utf-8'))
base = Path('/kaggle/working/mlx-test-train')

def p_exists(p: str) -> bool:
    if not p:
        return False
    pp = Path(p)
    return pp.exists() or (base / p).exists()

picked = None
for a in d.get('authors', []):
    style_ok = sum(1 for sv in (a.get('style_videos') or []) if p_exists((sv.get('localPath') or '').strip()))
    pair_ok = 0
    for pair in (a.get('pairs') or []):
        h = ((pair.get('hit_video') or {}).get('localPath') or '').strip()
        r = ((pair.get('anti_hit_video') or {}).get('localPath') or '').strip()
        if p_exists(h) and p_exists(r):
            pair_ok += 1
    if style_ok >= 3 and pair_ok >= 1:
        picked = a
        break

if picked is None:
    raise RuntimeError('No author with valid local files found for smoke.')

d['authors'] = [picked]
json.dump(d, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('written', out, 'picked', picked.get('author', {}).get('name'))
PY

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 python precompute_pvp_vision_features.py \
  --manifest dataset_50vid_of_prof/pvp_pairs_smoke1.json \
  --output-pt dataset_50vid_of_prof/pvp_precomputed_rows_smoke.pt \
  --output-hf-dir '' \
  --summary-json dataset_50vid_of_prof/pvp_precompute_summary_smoke.json \
  --skipped-json dataset_50vid_of_prof/pvp_precompute_skipped_smoke.json \
  --vision-model-name 'Qwen/Qwen2.5-Omni-7B' \
  --torch-dtype float16 \
  --device-map auto \
  --image-size 224 \
  --style-seconds 3 --style-fps 0.5 \
  --target-seconds 3 --target-fps 0.5 \
  2>&1 | tee /kaggle/working/precompute_smoke.log

python - << 'PY'
import json
s = json.load(open('dataset_50vid_of_prof/pvp_precompute_summary_smoke.json', 'r', encoding='utf-8'))
print('rows_total:', s.get('rows_total'))
assert int(s.get('rows_total', 0)) > 0, s
print('SMOKE PRECOMPUTE OK')
PY
```

## Cell 7: Smoke train (no-save mode, must pass)
```bash
%%bash
set -euo pipefail
cd /kaggle/working/mlx-test-train

python - << 'PY'
import json
cfg = json.load(open('kaggle_train_precomputed.json', 'r', encoding='utf-8'))
cfg['dataset_path'] = 'dataset_50vid_of_prof/pvp_precomputed_rows_smoke.pt'
cfg['output_dir'] = '/kaggle/working/outputs/pvp_dpo_smoke'
cfg['num_train_epochs'] = 0.02
cfg['logging_steps'] = 1
cfg['save_steps'] = 999999
cfg['max_seq_length'] = 1024
cfg['max_length'] = 1024
cfg['max_prompt_length'] = 256
cfg['per_device_train_batch_size'] = 1
cfg['gradient_accumulation_steps'] = 2
cfg['skip_vision_tower'] = True
cfg['vision_hidden_size'] = None
cfg['torch_dtype'] = 'float16'
cfg['bnb_4bit_compute_dtype'] = 'float16'
cfg['bf16'] = False
cfg['fp16'] = True
cfg['reference_free'] = True
cfg['precompute_ref_log_probs'] = True
cfg['gradient_checkpointing'] = False
cfg['report_to'] = 'none'
json.dump(cfg, open('/kaggle/working/kaggle_train_precomputed_smoke.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('written /kaggle/working/kaggle_train_precomputed_smoke.json')
PY

PVP_SKIP_TRAINER_SAVE=1 PVP_SKIP_FINAL_SAVE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
python train_dpo.py /kaggle/working/kaggle_train_precomputed_smoke.json \
  2>&1 | tee /kaggle/working/train_smoke.log

python - << 'PY'
import pathlib
t = pathlib.Path('/kaggle/working/train_smoke.log').read_text(encoding='utf-8', errors='ignore')
assert 'train_runtime' in t, 'train_runtime not found in smoke log'
print('SMOKE TRAIN OK')
PY
```

## Cell 8: Full overnight run (precompute + train)
This version is RAM-safe for precompute and robust to save errors.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/mlx-test-train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hard stop if disk is already low before full run.
free_gb="$(df --output=avail -BG /kaggle/working | tail -n1 | tr -dc '0-9')"
echo "free_gb=$free_gb"
if [ "$free_gb" -lt 18 ]; then
  echo 'Not enough free disk for full run. Clean disk first.'
  exit 3
fi

echo "=== FULL PRECOMPUTE START $(date) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 python precompute_pvp_vision_features.py \
  --manifest dataset_50vid_of_prof/pvp_pairs_50authors.json \
  --output-pt dataset_50vid_of_prof/pvp_precomputed_rows.pt \
  --output-hf-dir /kaggle/working/pvp_precomputed_hf \
  --summary-json dataset_50vid_of_prof/pvp_precompute_summary.json \
  --skipped-json dataset_50vid_of_prof/pvp_precompute_skipped.json \
  --vision-model-name 'Qwen/Qwen2.5-Omni-7B' \
  --torch-dtype float16 \
  --device-map auto \
  --image-size 224 \
  --style-seconds 6 --style-fps 1 \
  --target-seconds 6 --target-fps 1 \
  --flush-every-rows 8 \
  2>&1 | tee /kaggle/working/precompute_full.log

python - << 'PY'
import json
s = json.load(open('dataset_50vid_of_prof/pvp_precompute_summary.json', 'r', encoding='utf-8'))
rows = int(s.get('rows_total', 0))
print('rows_total:', rows)
if rows <= 0:
    raise SystemExit('rows_total=0 after full precompute')
PY

echo "=== BUILD TRAIN CONFIG $(date) ==="
python - << 'PY'
import json
cfg = json.load(open('kaggle_train_precomputed.json', 'r', encoding='utf-8'))
cfg['dataset_path'] = '/kaggle/working/pvp_precomputed_hf.part*'
cfg['output_dir'] = '/kaggle/working/outputs/pvp_dpo_precomputed'
cfg['skip_vision_tower'] = True
cfg['vision_hidden_size'] = None
cfg['torch_dtype'] = 'float16'
cfg['bnb_4bit_compute_dtype'] = 'float16'
cfg['bf16'] = False
cfg['fp16'] = True
cfg['reference_free'] = True
cfg['precompute_ref_log_probs'] = True
cfg['gradient_checkpointing'] = False
cfg['report_to'] = 'none'
# Avoid huge periodic checkpoints; rely on final save.
cfg['save_steps'] = 999999
json.dump(cfg, open('/kaggle/working/kaggle_train_precomputed_runtime.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('written /kaggle/working/kaggle_train_precomputed_runtime.json')
PY

echo "=== FULL TRAIN START $(date) ==="
PVP_IGNORE_SAVE_ERRORS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 \
  train_dpo.py /kaggle/working/kaggle_train_precomputed_runtime.json \
  2>&1 | tee /kaggle/working/train_full.log

echo "=== DONE $(date) ==="
```

## Cell 9: Final check
```bash
!df -h /kaggle/working
!ls -lah /kaggle/working/outputs/pvp_dpo_precomputed || true
!python - << 'PY'
import json
p = '/kaggle/working/mlx-test-train/dataset_50vid_of_prof/pvp_precompute_summary.json'
d = json.load(open(p, 'r', encoding='utf-8'))
print('rows_total:', d.get('rows_total'))
print('max_rows_buffered:', d.get('max_rows_buffered'))
print('flush_every_rows:', d.get('flush_every_rows'))
print('hf_parts:', len(d.get('output_hf_parts', []) or []))
PY
!tail -n 120 /kaggle/working/precompute_full.log || true
!tail -n 120 /kaggle/working/train_full.log || true
```

## Quick answer: "Does the model alone take 40GB?"
No. Usually not the model alone.
Disk is consumed by the sum of:
- HF model caches (vision + LLM + snapshots),
- precomputed dataset shards,
- logs and temporary files,
- checkpoints/final artifacts.

So total workspace usage can approach 30-50GB even when one model in 4-bit is much smaller.
