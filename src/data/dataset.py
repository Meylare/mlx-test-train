"""
Viral Video Dataset for MLX training.

Loads video frames via decord, processes them with the Qwen2.5-VL processor,
includes audio transcript as text in the prompt, and yields batches of
mx.array tensors via batch_iterator().
"""

import json
import os
from typing import Dict, List, Optional, Iterator

import mlx.core as mx
import numpy as np
from decord import VideoReader, cpu


class ViralVideoDataset:
    """
    Dataset that reads a JSONL file of video metadata and targets.

    Each line in the JSONL is expected to have::

        {
            "id": "shortCode",
            "video_path": "/path/to/video.mp4",
            "system_prompt": "Никнейм: ... Био: ... Длительность видео: 10 сек.",
            "aux_caption": "...",                   // optional, дубль описания
            "transcript": "Расшифровка аудио ...",  // optional
            "targets": {
                "viral_index": 1.23,
                "baseline_views": 5000,             // optional
                "real_views": 12000                  // optional
            }
        }

    Parameters
    ----------
    jsonl_path : str
        Path to the JSONL training file.
    processor : object
        The processor returned by ``mlx_vlm.load()`` — handles tokenisation
        and video frame processing.
    max_frames : int
        Number of frames to uniformly sample from each video.
    """

    def __init__(
        self,
        jsonl_path: str,
        processor,
        max_frames: int = 10,
    ):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Файл датасета не найден: {jsonl_path}")

        self.data: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        self.processor = processor
        self.max_frames = max_frames
        print(f"[Dataset] Загружено {len(self.data)} примеров из {jsonl_path}")

    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------
    # Single-item processing
    # ------------------------------------------------------------------

    def _process_item(self, idx: int) -> Optional[Dict[str, mx.array]]:
        """Process one sample. Returns dict of mx.arrays or None on failure."""
        item = self.data[idx]
        video_path = item["video_path"]

        # 1. Load video frames -------------------------------------------------
        frames = self._load_video(video_path)

        # 2. Build text prompt -------------------------------------------------
        # system_prompt уже содержит: никнейм, био, описание, отметки,
        # локацию, музыку, таймстамп, длительность.
        # aux_caption дублирует поле "Описание" из system_prompt — не добавляем.
        prompt_parts = [item.get("system_prompt", "")]

        # Транскрипция аудио (если есть в датасете)
        transcript = item.get("transcript", "")
        if transcript:
            prompt_parts.append(f"Транскрипция аудио: {transcript}")

        prompt_text = "\n".join(prompt_parts)

        # 3. Format as Qwen2.5-VL chat messages --------------------------------
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # 4. Tokenise via processor --------------------------------------------
        try:
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # NOTE: Qwen2.5-VL processor may only support return_tensors="pt".
            # In that case we get PyTorch tensors and convert via .numpy().
            # If "np" is supported, use it directly to avoid the torch dependency.
            try:
                inputs = self.processor(
                    text=[text_input],
                    videos=[frames],
                    padding=True,
                    return_tensors="np",
                )
                _to_numpy = lambda x: x  # already numpy
            except (ValueError, TypeError):
                # Fallback to PyTorch tensors → numpy
                inputs = self.processor(
                    text=[text_input],
                    videos=[frames],
                    padding=True,
                    return_tensors="pt",
                )
                _to_numpy = lambda x: x.numpy()
        except Exception as e:
            print(f"[Dataset] Ошибка процессора для {video_path}: {e}")
            return None

        # 5. Build result dict with mx.arrays ----------------------------------
        label = np.array(item["targets"]["viral_index"], dtype=np.float32)

        result: Dict[str, mx.array] = {
            "input_ids": mx.array(_to_numpy(inputs["input_ids"]).squeeze(0)),
            "attention_mask": mx.array(_to_numpy(inputs["attention_mask"]).squeeze(0)),
            "labels": mx.array(label),
        }

        # Pixel values for video
        if "pixel_values_videos" in inputs:
            result["pixel_values_videos"] = mx.array(
                _to_numpy(inputs["pixel_values_videos"]).squeeze(0)
            )
        elif "pixel_values" in inputs:
            result["pixel_values_videos"] = mx.array(
                _to_numpy(inputs["pixel_values"]).squeeze(0)
            )

        # Grid dimensions (T, H, W)
        if "video_grid_thw" in inputs:
            result["video_grid_thw"] = mx.array(
                _to_numpy(inputs["video_grid_thw"]).squeeze(0)
            )
        elif "image_grid_thw" in inputs:
            result["video_grid_thw"] = mx.array(
                _to_numpy(inputs["image_grid_thw"]).squeeze(0)
            )

        return result

    # ------------------------------------------------------------------
    # Video loading via decord
    # ------------------------------------------------------------------

    def _load_video(self, path: str) -> List[np.ndarray]:
        """Load *max_frames* uniformly-sampled RGB frames from *path*."""
        if not path or not os.path.exists(path):
            print(f"[Dataset] Видео не найдено: {path}")
            return self._black_frames()
        try:
            vr = VideoReader(path, ctx=cpu(0))
            total = len(vr)
            if total <= 0:
                return self._black_frames()
            indices = np.linspace(0, total - 1, self.max_frames).astype(int)
            frames = vr.get_batch(indices).asnumpy()  # [N, H, W, 3] RGB
            return list(frames)
        except Exception as e:
            print(f"[Dataset] Ошибка чтения {path}: {e}")
            return self._black_frames()

    def _black_frames(self) -> List[np.ndarray]:
        """Return a list of black placeholder frames."""
        return [
            np.zeros((224, 224, 3), dtype=np.uint8)
            for _ in range(self.max_frames)
        ]

    # ------------------------------------------------------------------
    # Batch iterator (replaces torch DataLoader)
    # ------------------------------------------------------------------

    def batch_iterator(
        self,
        batch_size: int = 1,
        shuffle: bool = True,
    ) -> Iterator[Dict[str, mx.array]]:
        """
        Yield collated batches of mx.array dicts.

        Shuffles indices each time the iterator is created (i.e. each epoch).
        Skips samples that fail to process.
        """
        indices = list(range(len(self.data)))
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            samples = []
            for idx in batch_indices:
                s = self._process_item(idx)
                if s is not None:
                    samples.append(s)

            if not samples:
                continue

            yield self._collate(samples)

    @staticmethod
    def _collate(samples: List[Dict[str, mx.array]]) -> Dict[str, mx.array]:
        """
        Collate a list of single-sample dicts into a batched dict.

        For batch_size=1 this is trivially adding a batch dimension.
        For batch_size>1 we pad input_ids / attention_mask to the max
        sequence length in the batch.
        Uses only keys present in all samples to avoid KeyError when
        processor returns different keys (e.g. missing video for one sample).
        """
        if len(samples) == 1:
            return {
                k: mx.expand_dims(v, axis=0) for k, v in samples[0].items()
            }

        # Keys present in every sample (avoids KeyError if one sample lacks pixel_values_videos etc.)
        keys_common = set(samples[0].keys())
        for s in samples[1:]:
            keys_common &= set(s.keys())

        max_seq = max(s["input_ids"].shape[0] for s in samples)
        batched: Dict[str, list] = {k: [] for k in keys_common}

        for s in samples:
            seq_len = s["input_ids"].shape[0]
            pad_len = max_seq - seq_len
            if pad_len > 0:
                s["input_ids"] = mx.concatenate(
                    [s["input_ids"], mx.zeros((pad_len,), dtype=s["input_ids"].dtype)]
                )
                s["attention_mask"] = mx.concatenate(
                    [s["attention_mask"], mx.zeros((pad_len,), dtype=s["attention_mask"].dtype)]
                )
            for k in keys_common:
                batched[k].append(mx.expand_dims(s[k], axis=0))

        return {k: mx.concatenate(v, axis=0) for k, v in batched.items()}
