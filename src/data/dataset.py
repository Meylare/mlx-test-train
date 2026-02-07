import mlx.core as mx
import numpy as np
from mlx_vlm.utils import load_processor
import json
import os
import cv2
from pathlib import Path

class ViralVideoDataset:
    def __init__(self, jsonl_path, model_path, max_frames=10):
        self.data = []
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Файл датасета не найден: {jsonl_path}")
            
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"[Dataset] Загружено {len(self.data)} примеров из {jsonl_path}")
        
        # Загружаем процессор через mlx-vlm (требует объект Path)
        self.processor = load_processor(Path(model_path))
        self.max_frames = max_frames

    def __len__(self):
        return len(self.data)

    def get_sample(self, idx):
        item = self.data[idx]
        video_path = item.get('video_path', '')
        
        # Загружаем кадры видео через OpenCV
        frames = self._load_video(video_path)

        prompt = f"<|im_start|>system\nYou are a viral video expert.<|im_end|>\n<|im_start|>user\n<|video|>{item.get('system_prompt', '')}<|im_end|>\n<|im_start|>assistant\n"
        
        # Библиотека transformers для Qwen2.5-VL сейчас поддерживает только "pt" (PyTorch)
        # Мы получаем тензоры и сразу переводим их в numpy, а затем в MLX
        inputs = self.processor(text=[prompt], videos=[frames], return_tensors="pt")

        batch = {
            "input_ids": mx.array(inputs["input_ids"].numpy()),
            "attention_mask": mx.array(inputs["attention_mask"].numpy()),
            "label": mx.array([item['targets']['viral_index']], mx.float32)
        }
        
        if "pixel_values_videos" in inputs:
            batch["pixel_values_videos"] = mx.array(inputs["pixel_values_videos"].numpy())
        if "video_grid_thw" in inputs:
            batch["video_grid_thw"] = mx.array(inputs["video_grid_thw"].numpy())
            
        return batch

    def _load_video(self, path):
        if not path or not os.path.exists(path):
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.max_frames)]

        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.max_frames)]
            
        # Равномерный сэмплинг кадров
        indices = np.linspace(0, total_frames - 1, self.max_frames).astype(int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # CV2 читает в BGR, переводим в RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        return frames