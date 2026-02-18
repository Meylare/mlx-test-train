from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .precompute_pvp_vision_features_mac import (
    _encode_video_frames,
    _fit_to_count,
    _parse_device_map,
    _parse_dtype,
    _sample_video_frames,
)
from modeling_pvp import VisionTower

try:
    from transformers import AutoImageProcessor
except Exception:
    AutoImageProcessor = None


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _feature_metrics(style_feats: np.ndarray, target_feats: np.ndarray) -> Dict[str, float]:
    style = style_feats.astype(np.float32, copy=False)
    target = target_feats.astype(np.float32, copy=False)

    style_flat = style.reshape(-1)
    target_flat = target.reshape(-1)

    style_norm = float(np.linalg.norm(style_flat))
    target_norm = float(np.linalg.norm(target_flat))
    denom = max(style_norm * target_norm, 1e-8)
    style_target_cosine = float(np.dot(style_flat[: min(style_flat.size, target_flat.size)],
                                       target_flat[: min(style_flat.size, target_flat.size)]) / denom)

    return {
        "style_mean": float(style.mean()),
        "style_std": float(style.std()),
        "style_norm": style_norm,
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "target_norm": target_norm,
        "style_target_cosine": style_target_cosine,
    }


def _run_generate(
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.generate",
        "--model",
        model_id,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--temp",
        str(temperature),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mlx_lm.generate failed with code {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def _build_decision_prompt(metrics: Dict[str, float]) -> str:
    return (
        "You are an expert short-video analyst.\n"
        "Use the provided style/target embedding diagnostics and classify virality likelihood.\n"
        "Return exactly one token: VIRAL or NOT_VIRAL.\n\n"
        "[PVP_FEATURES] "
        f"style_mean={metrics['style_mean']:.6f} "
        f"style_std={metrics['style_std']:.6f} "
        f"style_norm={metrics['style_norm']:.6f} "
        f"target_mean={metrics['target_mean']:.6f} "
        f"target_std={metrics['target_std']:.6f} "
        f"target_norm={metrics['target_norm']:.6f} "
        f"style_target_cosine={metrics['style_target_cosine']:.6f}"
    )


def _build_explain_prompt(label: str, metrics: Dict[str, float]) -> str:
    return (
        "You are an expert short-video analyst.\n"
        "You already predicted the label below. Explain why in clear text for a human.\n"
        "Output format:\n"
        "1) Final label: <VIRAL|NOT_VIRAL>\n"
        "2) Why: 3-5 short bullets with concrete reasoning.\n"
        "3) Confidence: low/medium/high with one sentence.\n\n"
        f"Predicted label: {label}\n"
        "[PVP_FEATURES] "
        f"style_mean={metrics['style_mean']:.6f} "
        f"style_std={metrics['style_std']:.6f} "
        f"style_norm={metrics['style_norm']:.6f} "
        f"target_mean={metrics['target_mean']:.6f} "
        f"target_std={metrics['target_std']:.6f} "
        f"target_norm={metrics['target_norm']:.6f} "
        f"style_target_cosine={metrics['style_target_cosine']:.6f}"
    )


def _encode_video(
    vision_tower: VisionTower,
    vision_image_processor: Optional[Any],
    video_path: Path,
    seconds: float,
    fps: float,
    image_size: int,
    vision_accepts_video: bool,
) -> torch.Tensor:
    frames = _sample_video_frames(
        video_path=video_path,
        seconds=seconds,
        fps=fps,
        image_size=image_size,
    )
    feats = _encode_video_frames(
        vision_tower=vision_tower,
        frames_tchw=frames,
        vision_accepts_video=vision_accepts_video,
        vision_image_processor=vision_image_processor,
    )
    return feats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local video inference for PVP MLX pipeline.")
    parser.add_argument("--model-id", required=True, help="MLX base model id/path.")
    parser.add_argument(
        "--style-video",
        action="append",
        required=True,
        help="Path to style video. Pass multiple times.",
    )
    parser.add_argument("--target-video", required=True, help="Path to target video.")
    parser.add_argument("--vision-model-name", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--vision-accepts-video", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--style-seconds", type=float, default=6.0)
    parser.add_argument("--style-fps", type=float, default=1.0)
    parser.add_argument("--target-seconds", type=float, default=6.0)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--style-videos-per-author", type=int, default=20)
    parser.add_argument("--decision-max-tokens", type=int, default=8)
    parser.add_argument("--explain-max-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json-output", default="", help="Optional path to save JSON result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    style_paths = [Path(p) for p in args.style_video]
    target_path = Path(args.target_video)
    missing = [p.as_posix() for p in style_paths + [target_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input videos: {missing}")

    dtype = _parse_dtype(args.torch_dtype)
    device_map = _parse_device_map(args.device_map)

    vision_tower = VisionTower(
        model_name=args.vision_model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    vision_accepts_video = bool(args.vision_accepts_video)

    vision_image_processor = None
    model_type = str(getattr(getattr(vision_tower.model, "config", object()), "model_type", "")).lower()
    if ("qwen2_5_omni" in model_type) or ("qwen2.5-omni" in str(args.vision_model_name).lower()):
        if AutoImageProcessor is None:
            raise ImportError("transformers AutoImageProcessor is required for Qwen2.5-Omni.")
        vision_image_processor = AutoImageProcessor.from_pretrained(
            args.vision_model_name,
            trust_remote_code=True,
        )

    style_feats_list: List[torch.Tensor] = []
    for style_path in style_paths:
        style_feats = _encode_video(
            vision_tower=vision_tower,
            vision_image_processor=vision_image_processor,
            video_path=style_path,
            seconds=args.style_seconds,
            fps=args.style_fps,
            image_size=args.image_size,
            vision_accepts_video=vision_accepts_video,
        )
        style_feats_list.append(style_feats)

    style_feats_list = _fit_to_count(style_feats_list, max(1, int(args.style_videos_per_author)))
    first_shape = tuple(style_feats_list[0].shape)
    style_feats_list = [x for x in style_feats_list if tuple(x.shape) == first_shape]
    if not style_feats_list:
        raise RuntimeError("Could not build aligned style feature set.")
    style_tensor = torch.stack(style_feats_list, dim=0).to(torch.float16).contiguous()

    target_feats = _encode_video(
        vision_tower=vision_tower,
        vision_image_processor=vision_image_processor,
        video_path=target_path,
        seconds=args.target_seconds,
        fps=args.target_fps,
        image_size=args.image_size,
        vision_accepts_video=vision_accepts_video,
    ).to(torch.float16).contiguous()

    metrics = _feature_metrics(_to_numpy(style_tensor), _to_numpy(target_feats))

    decision_prompt = _build_decision_prompt(metrics)
    decision_raw = _run_generate(
        model_id=args.model_id,
        prompt=decision_prompt,
        max_tokens=args.decision_max_tokens,
        temperature=args.temperature,
    )
    label = "NOT_VIRAL"
    upper = decision_raw.upper()
    if "VIRAL" in upper and "NOT_VIRAL" not in upper:
        label = "VIRAL"
    elif "NOT_VIRAL" in upper:
        label = "NOT_VIRAL"

    explain_prompt = _build_explain_prompt(label, metrics)
    explanation = _run_generate(
        model_id=args.model_id,
        prompt=explain_prompt,
        max_tokens=args.explain_max_tokens,
        temperature=max(args.temperature, 0.2),
    )

    result = {
        "label": label,
        "decision_raw": decision_raw,
        "explanation": explanation,
        "metrics": metrics,
        "inputs": {
            "style_videos": [p.as_posix() for p in style_paths],
            "target_video": target_path.as_posix(),
        },
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_output.strip():
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written {output_path.as_posix()}")


if __name__ == "__main__":
    main()
