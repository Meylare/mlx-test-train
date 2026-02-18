#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

mkdir -p training_mlx/logs outputs_mlx
DATASET_ROOT="${DATASET_ROOT:-datasetV}"
MANIFEST_SMOKE="${DATASET_ROOT}/pvp_pairs_smoke1.json"
PRECOMP_PT="${DATASET_ROOT}/pvp_precomputed_rows_smoke.pt"
PRECOMP_HF="${DATASET_ROOT}/pvp_precomputed_hf_smoke"
PRECOMP_SUMMARY="${DATASET_ROOT}/pvp_precompute_summary_smoke.json"
PRECOMP_SKIPPED="${DATASET_ROOT}/pvp_precompute_skipped_smoke.json"

echo "=== SMOKE: build mini manifest ==="
"${PYTHON_BIN}" - << 'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get("DATASET_ROOT", "datasetV"))
src = root / "pvp_pairs_50authors.json"
dst = root / "pvp_pairs_smoke1.json"
d = json.loads(src.read_text(encoding="utf-8"))
authors = d.get("authors", [])
if not authors:
    raise SystemExit("No authors in source manifest")
a0 = authors[0]
a0 = dict(a0)
a0["pairs"] = (a0.get("pairs") or [])[:5]
d["authors"] = [a0]
dst.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"written {dst.as_posix()} author={a0.get('author',{}).get('name')} pairs={len(a0['pairs'])}")
PY

echo "=== SMOKE: precompute ==="
"${PYTHON_BIN}" -m training_mlx.precompute_pvp_vision_features_mac \
  --manifest "${MANIFEST_SMOKE}" \
  --output-pt "${PRECOMP_PT}" \
  --output-hf-dir "${PRECOMP_HF}" \
  --summary-json "${PRECOMP_SUMMARY}" \
  --skipped-json "${PRECOMP_SKIPPED}" \
  --vision-model-name "Qwen/Qwen2.5-Omni-7B" \
  --torch-dtype float16 \
  --device-map auto \
  --image-size 224 \
  --style-seconds 3 --style-fps 0.5 \
  --target-seconds 3 --target-fps 0.5 \
  --flush-every-rows 5 \
  --strict-manifest-validation \
  --min-free-gb 8 \
  2>&1 | tee training_mlx/logs/precompute_smoke.log

echo "=== SMOKE: runtime config ==="
"${PYTHON_BIN}" - << 'PY'
import json
import os
from pathlib import Path

cfg_path = Path("training_mlx/configs/smoke.json")
runtime_path = Path("training_mlx/configs/smoke.runtime.json")
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
if cfg.get("model_id", "").startswith("REPLACE_"):
    raise SystemExit("Please set model_id in training_mlx/configs/smoke.json first.")
dataset_root = os.environ.get("DATASET_ROOT", "datasetV")
cfg["dataset_path"] = f"{dataset_root}/pvp_precomputed_hf_smoke.part*"
cfg["output_dir"] = "outputs_mlx/pvp_dpo_smoke"
runtime_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"written {runtime_path.as_posix()}")
PY

echo "=== SMOKE: train ==="
"${PYTHON_BIN}" -m training_mlx.train_dpo_mlx training_mlx/configs/smoke.runtime.json \
  2>&1 | tee training_mlx/logs/train_smoke.log

"${PYTHON_BIN}" - << 'PY'
from pathlib import Path
artifact = Path("outputs_mlx/pvp_dpo_smoke/adapter/adapter_manifest.json")
if not artifact.exists():
    raise SystemExit("SMOKE failed: adapter manifest not found")
print(f"SMOKE OK: {artifact.as_posix()}")
PY
