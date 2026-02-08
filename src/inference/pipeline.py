import json
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from src.inference.advisor import TimecodeAdvisor, VideoAdvice
from src.inference.db import get_video_row, init_db
from src.inference.gcs import download_gcs_uri
from src.inference.optimizer import ViralOptimizer
from src.inference.preprocess import (
    build_prompt_text,
    load_video_frames,
    prepare_inputs,
)
from src.models.viral_model import (
    ViralPredictorModel,
    load_lora_weights,
    load_viral_head,
)


class ViralHeadScorer(nn.Module):
    def __init__(self, viral_head: nn.Module):
        super().__init__()
        self.viral_head = viral_head

    def __call__(self, x: mx.array) -> mx.array:
        return self.viral_head(x)


def _ensure_batch(x: mx.array) -> mx.array:
    if x.ndim == 1:
        return mx.expand_dims(x, axis=0)
    return x


def _ensure_batch_dict(inputs: Dict[str, mx.array]) -> Dict[str, mx.array]:
    batched = {}
    for k, v in inputs.items():
        batched[k] = _ensure_batch(v)
    return batched


def _video_grid_to_thw(video_grid_thw: mx.array) -> Tuple[int, int, int]:
    if video_grid_thw is None:
        raise ValueError("video_grid_thw is required to extract video embeddings.")
    grid = np.array(video_grid_thw.tolist())
    grid = np.squeeze(grid)
    if grid.ndim != 1 or grid.shape[0] != 3:
        raise ValueError(f"Unexpected video_grid_thw shape: {grid.shape}")
    t, h, w = (int(grid[0]), int(grid[1]), int(grid[2]))
    return t, h, w


def _collect_video_token_ids(processor) -> List[int]:
    ids: List[int] = []
    tokenizer = getattr(processor, "tokenizer", None)
    candidates = {
        "<video>",
        "<image>",
        "<|video|>",
        "<|image|>",
        "<video_token>",
        "<image_token>",
        "<image_placeholder>",
        "<video_placeholder>",
    }

    if tokenizer is not None:
        for tok in getattr(tokenizer, "additional_special_tokens", []) or []:
            if isinstance(tok, str) and ("video" in tok.lower() or "image" in tok.lower()):
                candidates.add(tok)

        special_map = getattr(tokenizer, "special_tokens_map_extended", None)
        if isinstance(special_map, dict):
            for tok in special_map.values():
                if isinstance(tok, list):
                    for t in tok:
                        if isinstance(t, str) and ("video" in t.lower() or "image" in t.lower()):
                            candidates.add(t)
                elif isinstance(tok, str) and ("video" in tok.lower() or "image" in tok.lower()):
                    candidates.add(tok)

        for attr in ("image_token_id", "video_token_id"):
            tid = getattr(tokenizer, attr, None)
            if isinstance(tid, int):
                ids.append(tid)

        def _to_id(tok: str) -> Optional[int]:
            if not hasattr(tokenizer, "convert_tokens_to_ids"):
                return None
            try:
                tid = tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                return None
            if not isinstance(tid, int):
                return None
            unk = getattr(tokenizer, "unk_token_id", None)
            if unk is not None and tid == unk:
                return None
            return tid

        for tok in candidates:
            tid = _to_id(tok)
            if tid is not None:
                ids.append(tid)

    for attr in ("image_token_id", "video_token_id"):
        tid = getattr(processor, attr, None)
        if isinstance(tid, int):
            ids.append(tid)

    return list({int(x) for x in ids})


def _contiguous_runs(positions: List[int]) -> List[Tuple[int, int]]:
    if not positions:
        return []
    runs = []
    start = positions[0]
    prev = positions[0]
    for p in positions[1:]:
        if p == prev + 1:
            prev = p
            continue
        runs.append((start, prev))
        start = p
        prev = p
    runs.append((start, prev))
    return runs


def _find_video_span(
    input_ids: List[int],
    num_video_tokens: int,
    candidate_ids: List[int],
) -> Tuple[int, int]:
    best = None
    best_diff = None
    best_id = None

    for tid in candidate_ids:
        positions = [i for i, t in enumerate(input_ids) if t == tid]
        if not positions:
            continue
        for start, end in _contiguous_runs(positions):
            length = end - start + 1
            if length >= num_video_tokens:
                diff = length - num_video_tokens
                if best_diff is None or diff < best_diff:
                    best = (start, start + num_video_tokens)
                    best_diff = diff
                    best_id = tid

    if best is not None:
        return best[0], best[1]

    counts = {tid: input_ids.count(tid) for tid in candidate_ids}
    raise ValueError(
        "Could not locate video token span. "
        f"num_video_tokens={num_video_tokens}, seq_len={len(input_ids)}, "
        f"candidate_ids={candidate_ids}, counts={counts}"
    )


def _extract_video_embeddings(
    h: mx.array,
    input_ids: mx.array,
    video_grid_thw: mx.array,
    processor,
) -> mx.array:
    if video_grid_thw is None:
        raise ValueError("video_grid_thw is required for advice pipeline.")
    t, h_grid, w_grid = _video_grid_to_thw(video_grid_thw)
    num_video_tokens = t * h_grid * w_grid

    ids = _collect_video_token_ids(processor)
    if not ids:
        raise ValueError("Unable to determine video token ids from processor/tokenizer.")

    ids_1d = input_ids
    if ids_1d.ndim == 2 and ids_1d.shape[0] == 1:
        ids_1d = ids_1d[0]
    input_list = [int(x) for x in ids_1d.tolist()]

    start, end = _find_video_span(input_list, num_video_tokens, ids)

    video_tokens = h[:, start:end, :]
    batch, _, dim = video_tokens.shape
    video_tokens = video_tokens.reshape((batch, t, h_grid * w_grid, dim))
    video_frames = mx.mean(video_tokens, axis=2)
    return video_frames


def _advice_to_dict(advice: VideoAdvice) -> Dict[str, Any]:
    def _segment_to_dict(seg) -> Dict[str, Any]:
        return {
            "segment_id": seg.segment_id,
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
            "start_time_sec": seg.start_time_sec,
            "end_time_sec": seg.end_time_sec,
            "delta_norm": seg.delta_norm,
            "max_delta_norm": seg.max_delta_norm,
            "intensity": seg.intensity.value,
            "direction_vector": seg.direction_vector.tolist(),
            "contribution_percent": seg.contribution_percent,
        }

    def _frame_to_dict(fr) -> Dict[str, Any]:
        return {
            "frame_idx": fr.frame_idx,
            "time_sec": fr.time_sec,
            "delta_norm": fr.delta_norm,
            "rank": fr.rank,
            "advice_text": fr.advice_text,
        }

    return {
        "video_duration_sec": advice.video_duration_sec,
        "total_frames": advice.total_frames,
        "overall_score_improvement": advice.overall_score_improvement,
        "segments": [_segment_to_dict(s) for s in advice.segments],
        "top_frames": [_frame_to_dict(f) for f in advice.top_frames],
        "summary": advice.summary,
        "detailed_advice": advice.detailed_advice,
    }


def run_inference(
    video_id: str,
    db_path: str,
    model_path: str,
    lora_path: str,
    head_path: str,
    max_frames: int,
    cache_dir: str,
    use_cache: bool,
    with_advice: bool,
) -> Dict[str, Any]:
    init_db(db_path)
    row = get_video_row(db_path, video_id)

    gcs_uri = row.get("gcs_uri")
    transcript = row.get("transcript")
    system_prompt = row.get("system_prompt")
    metadata_json = row.get("metadata_json")

    local_video_path = download_gcs_uri(gcs_uri, cache_dir, use_cache=use_cache)
    frames, duration_sec, fps, total_frames = load_video_frames(local_video_path, max_frames)

    prompt_text = build_prompt_text(system_prompt, transcript)

    model = ViralPredictorModel(model_path)
    load_lora_weights(model, lora_path)
    load_viral_head(model, head_path)

    inputs = prepare_inputs(model.processor, frames, prompt_text, video_ref=local_video_path)
    inputs = _ensure_batch_dict(inputs)

    score = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values_videos=inputs.get("pixel_values_videos"),
        video_grid_thw=inputs.get("video_grid_thw"),
    )
    score_value = float(score.squeeze().item())

    result: Dict[str, Any] = {
        "video_id": video_id,
        "gcs_uri": gcs_uri,
        "score": score_value,
    }

    if metadata_json:
        try:
            result["metadata"] = json.loads(metadata_json)
        except Exception:
            result["metadata_raw"] = metadata_json

    if with_advice:
        inputs_embeds = model.backbone.get_input_embeddings(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values_videos"),
            video_grid_thw=inputs.get("video_grid_thw"),
        )
        position_ids, _ = model.backbone.language_model.get_rope_index(
            inputs["input_ids"],
            image_grid_thw=None,
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=inputs["attention_mask"],
        )
        h = model.backbone.language_model.model(
            inputs=None,
            inputs_embeds=inputs_embeds.inputs_embeds,
            position_ids=position_ids,
        )

        video_frames = _extract_video_embeddings(
            h=h,
            input_ids=inputs["input_ids"],
            video_grid_thw=inputs.get("video_grid_thw"),
            processor=model.processor,
        )

        scorer = ViralHeadScorer(model.viral_head)
        optimizer = ViralOptimizer(scorer)
        opt_result = optimizer.optimize(video_frames, verbose=False)

        fps_value = fps if fps and fps > 0 else 2.0
        advisor = TimecodeAdvisor(fps=fps_value)
        advice = advisor.analyze(
            delta_z=opt_result.delta_z,
            per_frame_delta_norm=opt_result.per_frame_delta_norm,
            score_improvement=opt_result.score_improvement,
            video_duration_sec=duration_sec,
        )

        result["advice"] = _advice_to_dict(advice)
        result["advice_text"] = advisor.format_advice(advice, verbose=True)

    return result
