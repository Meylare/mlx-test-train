import json
from pathlib import Path

import torch

try:
    from datasets import Dataset  # optional
except Exception:
    Dataset = None


def make_mock_rows(num_samples=3, S=20, T=2, H=448, W=448):
    rows = []
    for i in range(num_samples):
        rows.append(
            {
                "prompt": f"Analyze creator style and predict virality. Sample {i}",
                "chosen": "Strong hook + clear payoff + high replay potential.",
                "rejected": "This video is average and lacks a strong hook.",
                "style_pixel_values": torch.randn(S, T, 3, H, W),
                "chosen_target_pixel_values": torch.randn(T, 3, H, W),
                "rejected_target_pixel_values": torch.randn(T, 3, H, W),
            }
        )
    return rows


def save_preview_json(rows, path="mock_dataset_preview.json"):
    preview = []
    for r in rows:
        preview.append(
            {
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
                "style_shape": list(r["style_pixel_values"].shape),
                "chosen_target_shape": list(r["chosen_target_pixel_values"].shape),
                "rejected_target_shape": list(r["rejected_target_pixel_values"].shape),
            }
        )
    Path(path).write_text(json.dumps(preview, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    rows = make_mock_rows(num_samples=3)
    x = rows[0]

    print("Rows:", len(rows))
    print("prompt:", x["prompt"])
    print("chosen:", x["chosen"])
    print("rejected:", x["rejected"])
    print("style shape:", tuple(x["style_pixel_values"].shape))
    print("chosen target shape:", tuple(x["chosen_target_pixel_values"].shape))
    print("rejected target shape:", tuple(x["rejected_target_pixel_values"].shape))

    torch.save(rows, "mock_dataset_rows.pt")
    print("Saved tensor rows to: mock_dataset_rows.pt")

    preview_path = save_preview_json(rows)
    print(f"Saved readable preview to: {preview_path}")

    if Dataset is not None:
        try:
            ds = Dataset.from_list(rows)
            ds.save_to_disk("mock_dataset_hf")
            print("Saved Hugging Face dataset to: mock_dataset_hf")
        except Exception as exc:
            print(f"Could not save Hugging Face dataset: {exc}")
    else:
        print("datasets package unavailable; skipped Hugging Face dataset export.")
