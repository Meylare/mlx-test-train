import json
import os
from typing import Dict, List, Optional, Iterator

import mlx.core as mx
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

class ViralVideoDataset:
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
        print(f"[Dataset] Загружено {len(self.data)} примеров. Лимит кадров: {max_frames}")

    def __len__(self) -> int:
        return len(self.data)

    def _process_item(self, idx: int) -> Optional[Dict[str, mx.array]]:
        item = self.data[idx]
        video_path = item["video_path"]

        # 1. Загрузка и РУЧНОЙ РЕСАЙЗ кадров (чтобы не вылетело по памяти)
        frames = self._load_video(video_path)
        if frames is None:
            return None

        # 2. Формирование текста
        prompt_parts = [item.get("system_prompt", "")]
        transcript = item.get("transcript", "")
        if transcript:
            prompt_parts.append(f"Транскрипция аудио: {transcript}")
        prompt_text = "\n".join(prompt_parts)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # 3. Токенизация и процессинг
        try:
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Передаем УЖЕ сжатые кадры. return_tensors="pt" обязательно для видео.
            inputs = self.processor(
                text=[text_input],
                videos=[frames],
                padding=True,
                return_tensors="pt",
            )
            _to_numpy = lambda x: x.detach().cpu().numpy()
        except Exception as e:
            print(f"[Dataset] Ошибка процессора для {video_path}: {e}")
            return None

        # Вспомогательная функция для безопасного удаления батч-размерности
        def safe_squeeze(x):
            arr = _to_numpy(x)
            if arr.ndim > 0 and arr.shape[0] == 1:
                return np.squeeze(arr, axis=0)
            return arr

        # 4. Сборка результата
        label = np.array(item["targets"]["viral_index"], dtype=np.float32)

        result: Dict[str, mx.array] = {
            "input_ids": mx.array(safe_squeeze(inputs["input_ids"])),
            "attention_mask": mx.array(safe_squeeze(inputs["attention_mask"])),
            "labels": mx.array(label),
        }

        # Обработка видео-тензоров
        pv_key = "pixel_values_videos" if "pixel_values_videos" in inputs else "pixel_values"
        if pv_key in inputs:
            result["pixel_values_videos"] = mx.array(safe_squeeze(inputs[pv_key]))

        # Обработка сетки
        grid_key = "video_grid_thw" if "video_grid_thw" in inputs else "image_grid_thw"
        if grid_key in inputs:
            result["video_grid_thw"] = mx.array(safe_squeeze(inputs[grid_key]))

        return result

    def _load_video(self, path: str) -> Optional[List[np.ndarray]]:
        if not path or not os.path.exists(path):
            return None
        try:
            vr = VideoReader(path, ctx=cpu(0))
            total = len(vr)
            if total <= 0: return None
            
            # Равномерно берем кадры
            indices = np.linspace(0, total - 1, self.max_frames).astype(int)
            raw_frames = vr.get_batch(indices).asnumpy() # [N, H, W, C]
            
            # ЖЕСТКИЙ РЕСАЙЗ до 224x224 для экономии памяти
            resized_frames = []
            for f in raw_frames:
                img = Image.fromarray(f)
                img = img.resize((224, 224), Image.Resampling.BILINEAR)
                resized_frames.append(np.array(img))
            
            return resized_frames
        except Exception as e:
            print(f"[Dataset] Ошибка чтения {path}: {e}")
            return None

    def batch_iterator(self, batch_size: int = 1, shuffle: bool = True) -> Iterator[Dict[str, mx.array]]:
        indices = list(range(len(self.data)))
        if shuffle: np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            samples = [self._process_item(idx) for idx in batch_indices]
            samples = [s for s in samples if s is not None]
            if not samples: continue
            yield self._collate(samples)

    @staticmethod
    def _collate(samples: List[Dict[str, mx.array]]) -> Dict[str, mx.array]:
        if len(samples) == 1:
            return {k: mx.expand_dims(v, axis=0) for k, v in samples[0].items()}
        
        keys_common = set(samples[0].keys())
        for s in samples[1:]: keys_common &= set(s.keys())
        
        max_seq = max(s["input_ids"].shape[0] for s in samples)
        batched = {k: [] for k in keys_common}
        for s in samples:
            seq_len = s["input_ids"].shape[0]
            pad_len = max_seq - seq_len
            if pad_len > 0:
                s["input_ids"] = mx.concatenate([s["input_ids"], mx.zeros((pad_len,), dtype=s["input_ids"].dtype)])
                s["attention_mask"] = mx.concatenate([s["attention_mask"], mx.zeros((pad_len,), dtype=s["attention_mask"].dtype)])
            for k in keys_common:
                batched[k].append(mx.expand_dims(s[k], axis=0))
        return {k: mx.concatenate(v, axis=0) for k, v in batched.items()}