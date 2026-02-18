#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

mkdir -p training_mlx/logs outputs_mlx

CONFIG_PATH="${1:-training_mlx/configs/full_template.json}"

echo "=== FULL: validate config ==="
"${PYTHON_BIN}" - << PY
import json
from pathlib import Path
p = Path("${CONFIG_PATH}")
cfg = json.loads(p.read_text(encoding="utf-8"))
mid = str(cfg.get("model_id", ""))
if mid.startswith("REPLACE_") or not mid.strip():
    raise SystemExit(f"Please set model_id in {p.as_posix()} before full run.")
print(f"config_ok model_id={mid}")
PY

echo "=== FULL: precompute start $(date) ==="
"${PYTHON_BIN}" -m training_mlx.precompute_pvp_vision_features_mac \
  --manifest dataset_50vid_of_prof/pvp_pairs_50authors.json \
  --output-pt dataset_50vid_of_prof/pvp_precomputed_rows.pt \
  --output-hf-dir dataset_50vid_of_prof/pvp_precomputed_hf \
  --summary-json dataset_50vid_of_prof/pvp_precompute_summary.json \
  --skipped-json dataset_50vid_of_prof/pvp_precompute_skipped.json \
  --vision-model-name "Qwen/Qwen2.5-Omni-7B" \
  --torch-dtype float16 \
  --device-map auto \
  --image-size 224 \
  --style-seconds 6 --style-fps 1 \
  --target-seconds 6 --target-fps 1 \
  --flush-every-rows 8 \
  --strict-manifest-validation \
  --min-free-gb 10 \
  2>&1 | tee training_mlx/logs/precompute_full.log

echo "=== FULL: build runtime config ==="
"${PYTHON_BIN}" - << PY
import json
from pathlib import Path

cfg_path = Path("${CONFIG_PATH}")
runtime_path = Path("training_mlx/configs/full.runtime.json")
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
cfg["dataset_path"] = "dataset_50vid_of_prof/pvp_precomputed_hf.part*"
cfg["output_dir"] = "outputs_mlx/pvp_dpo_precomputed"
runtime_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"written {runtime_path.as_posix()}")
PY

echo "=== FULL: free disk after precompute ==="
df -h .

echo "=== FULL: cleanup PT chunks (keep HF parts) ==="
rm -f dataset_50vid_of_prof/pvp_precomputed_rows.pt*

echo "=== FULL: train start $(date) ==="
"${PYTHON_BIN}" -m training_mlx.train_dpo_mlx training_mlx/configs/full.runtime.json \
  2>&1 | tee training_mlx/logs/train_full.log

echo "=== FULL: done $(date) ==="
echo "Artifacts:"
echo "  adapter: outputs_mlx/pvp_dpo_precomputed/adapter/adapter_manifest.json"
echo "  logs: training_mlx/logs/precompute_full.log training_mlx/logs/train_full.log"
