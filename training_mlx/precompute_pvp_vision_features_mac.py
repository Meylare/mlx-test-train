from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from modeling_pvp import VisionTower

try:
    from einops import rearrange
except Exception as exc:  # pragma: no cover
    rearrange = None
    _EINOPS_IMPORT_ERROR = exc

try:
    from transformers import AutoImageProcessor
except Exception as exc:  # pragma: no cover
    AutoImageProcessor = None
    _AUTO_IMAGE_PROCESSOR_IMPORT_ERROR = exc

try:
    from torchvision.io import read_video
    from torchvision.transforms import functional as TF
except Exception as exc:  # pragma: no cover
    read_video = None
    TF = None
    _TORCHVISION_IMPORT_ERROR = exc

try:
    import av  # noqa: F401
except Exception as exc:  # pragma: no cover
    av = None
    _PYAV_IMPORT_ERROR = exc


REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_einops() -> None:
    if rearrange is None:
        raise ImportError("einops is required for tensor shape flattening.") from _EINOPS_IMPORT_ERROR


def _require_torchvision() -> None:
    if read_video is None or TF is None:
        raise ImportError(
            "torchvision is required for video decoding. Install torchvision in your local environment."
        ) from _TORCHVISION_IMPORT_ERROR
    if av is None:
        raise ImportError(
            "PyAV is required for torchvision.read_video. Install with `pip install av`."
        ) from _PYAV_IMPORT_ERROR


def _require_auto_image_processor() -> None:
    if AutoImageProcessor is None:
        raise ImportError(
            "transformers AutoImageProcessor is required for Qwen2.5-Omni vision preprocessing."
        ) from _AUTO_IMAGE_PROCESSOR_IMPORT_ERROR


def _parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = value.strip().lower()
    if key not in mapping:
        supported = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported dtype '{value}'. Supported: {supported}")
    return mapping[key]


def _parse_device_map(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, int)):
        return value
    if not isinstance(value, str):
        return value
    raw = value.strip().lower()
    if raw in {"", "none", "null"}:
        return None
    if raw.isdigit():
        return int(raw)
    return value


def _ensure_free_disk(min_free_gb: int, target_path: Path) -> int:
    if min_free_gb <= 0:
        return int(shutil.disk_usage(str(target_path)).free / (1024 ** 3))
    probe = target_path if target_path.exists() else target_path.parent
    usage = shutil.disk_usage(str(probe))
    free_gb = int(usage.free / (1024 ** 3))
    if free_gb < int(min_free_gb):
        raise RuntimeError(
            f"Not enough free disk space at {probe.as_posix()}: "
            f"free={free_gb}GB, required>={min_free_gb}GB."
        )
    return free_gb


def _resolve_video_path(local_path: str, manifest_path: Path) -> Path:
    candidate = Path(local_path)
    if candidate.is_absolute():
        return candidate
    # 1) Path relative to current working directory.
    if candidate.exists():
        return candidate.resolve()
    # 2) Path relative to manifest directory.
    manifest_relative = (manifest_path.parent / candidate).resolve()
    if manifest_relative.exists():
        return manifest_relative
    # 3) Path relative to repository root.
    repo_relative = (REPO_ROOT / candidate).resolve()
    return repo_relative


def _sample_video_frames(
    video_path: Path,
    seconds: float,
    fps: float,
    image_size: int,
) -> torch.Tensor:
    _require_torchvision()
    read_kwargs: Dict[str, Any] = {"pts_unit": "sec"}
    if seconds > 0:
        read_kwargs["start_pts"] = 0.0
        read_kwargs["end_pts"] = float(seconds)
    try:
        frames, _, info = read_video(str(video_path), **read_kwargs)
    except TypeError:
        frames, _, info = read_video(str(video_path), pts_unit="sec")
    if frames.ndim != 4 or frames.size(0) == 0:
        raise ValueError(f"No readable frames in {video_path.as_posix()}")

    source_fps = float(info.get("video_fps", 0.0) or 0.0)
    if source_fps > 0 and seconds > 0:
        max_src_frames = int(math.ceil(seconds * source_fps))
        max_src_frames = max(1, min(max_src_frames, frames.size(0)))
        frames = frames[:max_src_frames]

    target_frames = int(round(seconds * fps)) if seconds > 0 else int(round(fps))
    target_frames = max(1, target_frames)
    idx = torch.linspace(0, frames.size(0) - 1, steps=target_frames).long()
    sampled = frames[idx]
    sampled = sampled.permute(0, 3, 1, 2).contiguous().float() / 255.0
    resized = torch.stack(
        [TF.resize(frame, [image_size, image_size], antialias=True) for frame in sampled],
        dim=0,
    )
    return resized


def _encode_video_frames(
    vision_tower: VisionTower,
    frames_tchw: torch.Tensor,
    vision_accepts_video: bool,
    vision_image_processor: Optional[Any] = None,
) -> torch.Tensor:
    _require_einops()
    if frames_tchw.ndim != 4:
        raise ValueError(f"Expected [T, C, H, W], got {frames_tchw.shape}")

    vision_param = next(vision_tower.parameters())
    device = vision_param.device
    dtype = vision_param.dtype
    with torch.no_grad():
        if vision_image_processor is not None:
            images = [
                (
                    frame.clamp(0.0, 1.0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                for frame in frames_tchw
            ]
            image_inputs = vision_image_processor(images=images, return_tensors="pt")
            pixel_values = image_inputs.get("pixel_values")
            image_grid_thw = image_inputs.get("image_grid_thw")
            if pixel_values is None or image_grid_thw is None:
                raise ValueError(
                    f"AutoImageProcessor output is incomplete. keys={list(image_inputs.keys())}"
                )
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            image_grid_thw = image_grid_thw.to(device=device)
            feats = vision_tower(pixel_values, image_grid_thw=image_grid_thw)
            if feats.ndim == 2:
                pass
            elif feats.ndim == 3:
                feats = rearrange(feats, "b n d -> (b n) d")
            elif feats.ndim == 4:
                feats = rearrange(feats, "b t n d -> (b t n) d")
            else:
                raise ValueError(f"Unexpected Omni features shape: {feats.shape}")
            return feats.detach().cpu()

        if vision_accepts_video:
            feats = vision_tower(frames_tchw.unsqueeze(0).to(device=device, dtype=dtype))
            if feats.ndim == 4:
                feats = rearrange(feats, "b t n d -> b (t n) d")
            elif feats.ndim != 3:
                raise ValueError(f"Unexpected video features shape: {feats.shape}")
            feats = feats[0]
        else:
            feats = vision_tower(frames_tchw.to(device=device, dtype=dtype))
            if feats.ndim == 3:
                feats = rearrange(feats, "t n d -> (t n) d")
            elif feats.ndim == 4:
                feats = rearrange(feats, "t s n d -> (t s n) d")
            else:
                raise ValueError(f"Unexpected frame features shape: {feats.shape}")
    return feats.detach().cpu()


def _fit_to_count(items: List[torch.Tensor], count: int) -> List[torch.Tensor]:
    if not items:
        return items
    if len(items) >= count:
        return items[:count]
    out = list(items)
    i = 0
    while len(out) < count:
        out.append(items[i % len(items)])
        i += 1
    return out


def _build_prompt(author_name: str) -> str:
    return (
        "You are given the creator style history and one target video. "
        f"Creator: @{author_name}. "
        "Decide if this target video is likely to be a viral hit for this creator. "
        "Answer with one token: VIRAL or NOT_VIRAL."
    )


def _to_hf_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
        "style_vision_features": row["style_vision_features"].numpy(),
        "chosen_target_vision_features": row["chosen_target_vision_features"].numpy(),
        "rejected_target_vision_features": row["rejected_target_vision_features"].numpy(),
        "author_name": row["author_name"],
        "pair_index": row["pair_index"],
    }


def _validate_manifest_paths(
    manifest: Dict[str, Any],
    manifest_path: Path,
    strict: bool,
) -> Dict[str, Any]:
    authors = manifest.get("authors") or []
    issues: List[Dict[str, Any]] = []
    checked_pairs = 0
    checked_styles = 0
    for author_row in authors:
        author = author_row.get("author") or {}
        author_name = str(author.get("name") or "")
        style_count_existing = 0
        for style_video in author_row.get("style_videos") or []:
            checked_styles += 1
            local_path = (style_video.get("localPath") or "").strip()
            if not local_path:
                issues.append(
                    {"author": author_name, "type": "style", "reason": "missing_local_path"}
                )
                continue
            resolved = _resolve_video_path(local_path, manifest_path)
            if resolved.exists():
                style_count_existing += 1
            else:
                issues.append(
                    {
                        "author": author_name,
                        "type": "style",
                        "reason": "local_file_missing",
                        "path": local_path,
                    }
                )

        if style_count_existing == 0:
            issues.append({"author": author_name, "type": "author", "reason": "no_style_videos_available"})

        for pair in author_row.get("pairs") or []:
            checked_pairs += 1
            hit = pair.get("hit_video") or pair.get("hit") or {}
            anti = pair.get("anti_hit_video") or pair.get("antiHit") or {}
            hit_path = (hit.get("localPath") or "").strip()
            anti_path = (anti.get("localPath") or "").strip()
            if not hit_path or not anti_path:
                issues.append(
                    {
                        "author": author_name,
                        "pair_index": pair.get("pair_index"),
                        "type": "pair",
                        "reason": "missing_local_path",
                    }
                )
                continue
            hit_file = _resolve_video_path(hit_path, manifest_path)
            anti_file = _resolve_video_path(anti_path, manifest_path)
            if not hit_file.exists() or not anti_file.exists():
                issues.append(
                    {
                        "author": author_name,
                        "pair_index": pair.get("pair_index"),
                        "type": "pair",
                        "reason": "local_file_missing",
                        "hit_path": hit_path,
                        "anti_path": anti_path,
                    }
                )

    report = {
        "authors_total": len(authors),
        "checked_pairs": checked_pairs,
        "checked_styles": checked_styles,
        "issues_count": len(issues),
        "issues_preview": issues[:20],
    }
    if strict and issues:
        raise ValueError(
            "Manifest strict validation failed before precompute. "
            f"issues_count={len(issues)} preview={issues[:10]}"
        )
    return report


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authors_all = manifest.get("authors") or []
    if not authors_all:
        raise ValueError(f"No authors in manifest: {manifest_path.as_posix()}")

    precheck = _validate_manifest_paths(
        manifest=manifest,
        manifest_path=manifest_path,
        strict=bool(args.strict_manifest_validation),
    )
    print(json.dumps({"manifest_precheck": precheck}, ensure_ascii=False))

    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)
    if args.use_torchrun_sharding:
        num_shards = int(os.environ.get("WORLD_SIZE", "1"))
        shard_index = int(os.environ.get("LOCAL_RANK", "0"))
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}], got {shard_index}")

    authors = [a for idx, a in enumerate(authors_all) if idx % num_shards == shard_index]
    if not authors:
        raise ValueError(
            f"No authors assigned to shard {shard_index}/{num_shards} from manifest {manifest_path.as_posix()}"
        )

    dtype = _parse_dtype(args.torch_dtype)
    if dtype == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("BF16 is not supported on this GPU. Falling back to FP16 for precompute.")
        dtype = torch.float16

    device_map = _parse_device_map(args.device_map)
    if args.use_torchrun_sharding and (device_map == "auto" or device_map is None):
        device_map = int(os.environ.get("LOCAL_RANK", "0"))

    shard_tag = ""
    if num_shards > 1:
        shard_tag = f".shard{shard_index:02d}of{num_shards:02d}"

    output_pt = Path(args.output_pt)
    if shard_tag:
        output_pt = output_pt.with_name(f"{output_pt.stem}{shard_tag}{output_pt.suffix}")
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    _ensure_free_disk(int(args.min_free_gb), output_pt.parent)

    output_hf_dir: Optional[Path] = None
    if args.output_hf_dir:
        output_hf_dir = Path(args.output_hf_dir)
        if shard_tag:
            output_hf_dir = output_hf_dir.with_name(f"{output_hf_dir.name}{shard_tag}")
        output_hf_dir.parent.mkdir(parents=True, exist_ok=True)
        _ensure_free_disk(int(args.min_free_gb), output_hf_dir.parent)

    flush_every_rows = max(0, int(args.flush_every_rows))
    rows_buffer: List[Dict[str, Any]] = []
    rows_total = 0
    rows_buffer_max = 0
    output_pt_parts: List[str] = []
    output_hf_parts: List[str] = []
    output_part_index = 0
    skipped: List[Dict[str, Any]] = []
    authors_with_style_cache = 0

    vision_tower = VisionTower(
        model_name=args.vision_model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    vision_accepts_video = bool(args.vision_accepts_video)
    model_type = str(getattr(getattr(vision_tower.model, "config", object()), "model_type", "")).lower()
    print(f"Vision model_type detected: {model_type}")
    vision_image_processor = None
    if ("qwen2_5_omni" in model_type) or ("qwen2.5-omni" in str(args.vision_model_name).lower()):
        _require_auto_image_processor()
        vision_image_processor = AutoImageProcessor.from_pretrained(
            args.vision_model_name,
            trust_remote_code=True,
        )
        print("Using AutoImageProcessor path for Qwen2.5-Omni (image_grid_thw enabled).")

    hf_dataset_cls = None
    if output_hf_dir is not None:
        from datasets import Dataset as _HFDataset

        hf_dataset_cls = _HFDataset

    def _flush_rows_if_needed(force: bool = False) -> None:
        nonlocal output_part_index, rows_total, rows_buffer_max
        if not rows_buffer:
            return
        if not force and flush_every_rows <= 0:
            return
        if not force and len(rows_buffer) < flush_every_rows:
            return
        _ensure_free_disk(int(args.min_free_gb), output_pt.parent)

        rows_buffer_max = max(rows_buffer_max, len(rows_buffer))
        if flush_every_rows > 0:
            part_tag = f".part{output_part_index:05d}"
            output_pt_chunk = output_pt.with_name(f"{output_pt.stem}{part_tag}{output_pt.suffix}")
        else:
            part_tag = ""
            output_pt_chunk = output_pt

        torch.save(rows_buffer, output_pt_chunk)
        output_pt_parts.append(output_pt_chunk.as_posix())

        if output_hf_dir is not None:
            hf_rows = [_to_hf_row(x) for x in rows_buffer]
            ds = hf_dataset_cls.from_list(hf_rows)
            if flush_every_rows > 0:
                output_hf_chunk = output_hf_dir.with_name(f"{output_hf_dir.name}{part_tag}")
            else:
                output_hf_chunk = output_hf_dir
            if output_hf_chunk.exists():
                shutil.rmtree(output_hf_chunk, ignore_errors=True)
            output_hf_chunk.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(output_hf_chunk))
            output_hf_parts.append(output_hf_chunk.as_posix())
            del ds
            del hf_rows

        rows_total += len(rows_buffer)
        rows_buffer.clear()
        output_part_index += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target_style_count = int(manifest.get("meta", {}).get("style_per_author", args.style_videos_per_author))
    target_style_count = max(1, target_style_count)

    for author_row in authors:
        author = author_row.get("author") or {}
        author_name = str(author.get("name") or "")
        if not author_name:
            skipped.append({"reason": "missing_author_name"})
            continue

        style_videos = author_row.get("style_videos") or []
        style_feats_list: List[torch.Tensor] = []
        for style_video in style_videos:
            local_path = (style_video.get("localPath") or "").strip()
            if not local_path:
                continue
            video_path = _resolve_video_path(local_path, manifest_path)
            if not video_path.exists():
                continue
            try:
                frames = _sample_video_frames(
                    video_path=video_path,
                    seconds=args.style_seconds,
                    fps=args.style_fps,
                    image_size=args.image_size,
                )
                feats = _encode_video_frames(
                    vision_tower=vision_tower,
                    frames_tchw=frames,
                    vision_accepts_video=vision_accepts_video,
                    vision_image_processor=vision_image_processor,
                )
                style_feats_list.append(feats)
                del frames
            except Exception as exc:
                skipped.append(
                    {
                        "author": author_name,
                        "video_id": style_video.get("id"),
                        "type": "style",
                        "reason": f"style_encode_error: {exc}",
                    }
                )

        style_feats_list = _fit_to_count(style_feats_list, target_style_count)
        if not style_feats_list:
            skipped.append({"author": author_name, "type": "author", "reason": "no_style_videos_encoded"})
            continue

        first_shape = tuple(style_feats_list[0].shape)
        aligned_style = [x for x in style_feats_list if tuple(x.shape) == first_shape]
        aligned_style = _fit_to_count(aligned_style, target_style_count)
        if len(aligned_style) != target_style_count:
            skipped.append({"author": author_name, "type": "author", "reason": "style_feature_shape_mismatch"})
            continue

        style_tensor = torch.stack(aligned_style, dim=0).to(torch.float16).contiguous()
        authors_with_style_cache += 1

        for pair in author_row.get("pairs") or []:
            hit = pair.get("hit_video") or pair.get("hit") or {}
            anti = pair.get("anti_hit_video") or pair.get("antiHit") or {}
            hit_path = (hit.get("localPath") or "").strip()
            anti_path = (anti.get("localPath") or "").strip()
            if not hit_path or not anti_path:
                skipped.append(
                    {
                        "author": author_name,
                        "pair_index": pair.get("pair_index"),
                        "type": "pair",
                        "reason": "missing_local_path",
                    }
                )
                continue

            hit_file = _resolve_video_path(hit_path, manifest_path)
            anti_file = _resolve_video_path(anti_path, manifest_path)
            if not hit_file.exists() or not anti_file.exists():
                skipped.append(
                    {
                        "author": author_name,
                        "pair_index": pair.get("pair_index"),
                        "type": "pair",
                        "reason": "local_file_missing",
                    }
                )
                continue

            try:
                hit_frames = _sample_video_frames(
                    video_path=hit_file,
                    seconds=args.target_seconds,
                    fps=args.target_fps,
                    image_size=args.image_size,
                )
                anti_frames = _sample_video_frames(
                    video_path=anti_file,
                    seconds=args.target_seconds,
                    fps=args.target_fps,
                    image_size=args.image_size,
                )
                hit_feats = _encode_video_frames(
                    vision_tower=vision_tower,
                    frames_tchw=hit_frames,
                    vision_accepts_video=vision_accepts_video,
                    vision_image_processor=vision_image_processor,
                )
                anti_feats = _encode_video_frames(
                    vision_tower=vision_tower,
                    frames_tchw=anti_frames,
                    vision_accepts_video=vision_accepts_video,
                    vision_image_processor=vision_image_processor,
                )
                del hit_frames
                del anti_frames
            except Exception as exc:
                skipped.append(
                    {
                        "author": author_name,
                        "pair_index": pair.get("pair_index"),
                        "type": "pair",
                        "reason": f"target_encode_error: {exc}",
                    }
                )
                continue

            if tuple(hit_feats.shape) != tuple(anti_feats.shape):
                skipped.append(
                    {
                        "author": author_name,
                        "pair_index": pair.get("pair_index"),
                        "type": "pair",
                        "reason": f"target_shape_mismatch: hit={tuple(hit_feats.shape)} anti={tuple(anti_feats.shape)}",
                    }
                )
                continue

            rows_buffer.append(
                {
                    "prompt": _build_prompt(author_name),
                    "chosen": "VIRAL",
                    "rejected": "NOT_VIRAL",
                    "style_vision_features": style_tensor,
                    "chosen_target_vision_features": hit_feats.to(torch.float16).contiguous(),
                    "rejected_target_vision_features": anti_feats.to(torch.float16).contiguous(),
                    "author_name": author_name,
                    "pair_index": int(pair.get("pair_index") or 0),
                }
            )
            del hit_feats
            del anti_feats
            _flush_rows_if_needed(force=False)

        del style_tensor
        del style_feats_list
        del aligned_style
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _flush_rows_if_needed(force=True)

    output_pt_summary = output_pt.as_posix()
    if flush_every_rows > 0:
        output_pt_summary = output_pt.with_name(f"{output_pt.stem}.part*{output_pt.suffix}").as_posix()
    output_hf_summary = None
    if output_hf_dir is not None:
        output_hf_summary = output_hf_dir.as_posix()
        if flush_every_rows > 0:
            output_hf_summary = output_hf_dir.with_name(f"{output_hf_dir.name}.part*").as_posix()

    summary = {
        "manifest": manifest_path.as_posix(),
        "rows_total": rows_total,
        "authors_in_manifest": len(authors_all),
        "authors_in_shard": len(authors),
        "authors_with_style_cache": authors_with_style_cache,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "style_seconds": args.style_seconds,
        "style_fps": args.style_fps,
        "target_seconds": args.target_seconds,
        "target_fps": args.target_fps,
        "image_size": args.image_size,
        "device_map": device_map,
        "torch_dtype_runtime": str(dtype),
        "output_pt": output_pt_summary,
        "output_pt_parts": output_pt_parts,
        "output_hf_dir": output_hf_summary,
        "output_hf_parts": output_hf_parts,
        "flush_every_rows": flush_every_rows,
        "max_rows_buffered": rows_buffer_max,
        "skipped_count": len(skipped),
        "strict_manifest_validation": bool(args.strict_manifest_validation),
        "min_free_gb": int(args.min_free_gb),
    }

    skipped_path = Path(args.skipped_json)
    if shard_tag:
        skipped_path = skipped_path.with_name(f"{skipped_path.stem}{shard_tag}{skipped_path.suffix}")
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = Path(args.summary_json)
    if shard_tag:
        summary_path = summary_path.with_name(f"{summary_path.stem}{shard_tag}{summary_path.suffix}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if rows_total == 0:
        reason_counter = collections.Counter()
        for item in skipped:
            reason = str(item.get("reason") or "unknown")
            reason_counter[reason] += 1
        top_reasons = reason_counter.most_common(5)
        raise RuntimeError(
            "No training rows were produced during precompute. "
            f"Top skipped reasons: {top_reasons}. "
            f"Check {skipped_path.as_posix()} and localPath/video availability."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute PVP vision features on local Mac.")
    parser.add_argument("--manifest", default="dataset_50vid_of_prof/pvp_pairs_50authors.json")
    parser.add_argument("--output-pt", default="dataset_50vid_of_prof/pvp_precomputed_rows.pt")
    parser.add_argument(
        "--output-hf-dir",
        default="dataset_50vid_of_prof/pvp_precomputed_hf",
        help="HF output dir; pass empty string to disable.",
    )
    parser.add_argument("--skipped-json", default="dataset_50vid_of_prof/pvp_precompute_skipped.json")
    parser.add_argument("--summary-json", default="dataset_50vid_of_prof/pvp_precompute_summary.json")
    parser.add_argument("--vision-model-name", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--vision-accepts-video", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--style-seconds", type=float, default=6.0)
    parser.add_argument("--style-fps", type=float, default=1.0)
    parser.add_argument("--target-seconds", type=float, default=6.0)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--style-videos-per-author", type=int, default=20)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--flush-every-rows", type=int, default=8)
    parser.add_argument("--use-torchrun-sharding", action="store_true")
    parser.add_argument(
        "--strict-manifest-validation",
        action="store_true",
        help="Fail before precompute when any localPath is missing/broken.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=int,
        default=10,
        help="Hard stop when free disk is below this threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_hf_dir is not None and str(args.output_hf_dir).strip() == "":
        args.output_hf_dir = None
    run(args)


if __name__ == "__main__":
    main()
