import os
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np
from decord import VideoReader, cpu
from PIL import Image


def load_video_frames(
    path: str,
    max_frames: int = 6,
) -> Tuple[List[np.ndarray], Optional[float], Optional[float], int]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")

    vr = VideoReader(path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {path}")

    fps = None
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = None

    duration = None
    if fps and fps > 0:
        duration = total_frames / fps

    indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
    raw_frames = vr.get_batch(indices).asnumpy()

    resized_frames: List[np.ndarray] = []
    for f in raw_frames:
        img = Image.fromarray(f)
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        resized_frames.append(np.array(img))

    return resized_frames, duration, fps, total_frames


def build_prompt_text(system_prompt: Optional[str], transcript: Optional[str]) -> str:
    parts: List[str] = []
    if system_prompt:
        parts.append(system_prompt)
    if transcript:
        parts.append(f"Транскрипция аудио: {transcript}")
    return "\n".join(parts).strip()


def prepare_inputs(
    processor,
    frames: List[np.ndarray],
    prompt_text: str,
    video_ref: str = "video",
) -> Dict[str, mx.array]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_ref},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input],
        videos=[frames],
        padding=True,
        return_tensors="pt",
    )

    def _to_numpy(x):
        return x.detach().cpu().numpy()

    def safe_squeeze(x):
        arr = _to_numpy(x)
        if arr.ndim > 0 and arr.shape[0] == 1:
            return np.squeeze(arr, axis=0)
        return arr

    result: Dict[str, mx.array] = {
        "input_ids": mx.array(safe_squeeze(inputs["input_ids"])),
        "attention_mask": mx.array(safe_squeeze(inputs["attention_mask"])),
    }

    pv_key = "pixel_values_videos" if "pixel_values_videos" in inputs else "pixel_values"
    if pv_key in inputs:
        result["pixel_values_videos"] = mx.array(safe_squeeze(inputs[pv_key]))

    grid_key = "video_grid_thw" if "video_grid_thw" in inputs else "image_grid_thw"
    if grid_key in inputs:
        result["video_grid_thw"] = mx.array(safe_squeeze(inputs[grid_key]))

    return result
