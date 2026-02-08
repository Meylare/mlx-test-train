#!/usr/bin/env python3
import argparse
import json
import os
import sys

from src.inference.pipeline import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run viral model inference by video_id.")
    parser.add_argument("--video-id", required=True, help="Video ID to fetch from SQLite.")
    parser.add_argument("--db-path", default="data/viral.db", help="Path to SQLite DB.")
    parser.add_argument(
        "--model-path",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        help="Backbone model path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/epoch_3",
        help="Checkpoint directory with lora_adapter.npz and viral_head.npz.",
    )
    parser.add_argument("--lora-path", default=None, help="Override LoRA adapter path.")
    parser.add_argument("--head-path", default=None, help="Override viral head path.")
    parser.add_argument("--max-frames", type=int, default=6, help="Number of frames to sample.")
    parser.add_argument("--out-json", default=None, help="Optional JSON output path.")
    parser.add_argument("--cache-dir", default=".cache/videos", help="GCS download cache dir.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache usage.")
    advice_group = parser.add_mutually_exclusive_group()
    advice_group.add_argument("--advice", dest="advice", action="store_true", help="Enable timecode advice.")
    advice_group.add_argument("--no-advice", dest="advice", action="store_false", help="Disable timecode advice.")
    parser.set_defaults(advice=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    lora_path = args.lora_path or os.path.join(checkpoint_dir, "lora_adapter.npz")
    head_path = args.head_path or os.path.join(checkpoint_dir, "viral_head.npz")

    result = run_inference(
        video_id=args.video_id,
        db_path=args.db_path,
        model_path=args.model_path,
        lora_path=lora_path,
        head_path=head_path,
        max_frames=args.max_frames,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        with_advice=args.advice,
    )

    print(f"video_id: {result.get('video_id')}")
    print(f"gcs_uri: {result.get('gcs_uri')}")
    print(f"score: {result.get('score'):.6f}")

    if "advice_text" in result:
        print("\n--- advice ---")
        print(result["advice_text"])

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
