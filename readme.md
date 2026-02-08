## 📦 О структуре чекпоинтов (LoRA)

В процессе обучения в папке `checkpoints/epoch_N/` сохраняются только измененные части модели, а не вся нейросеть целиком.

### Почему файлы такие маленькие?
Мы используем метод **LoRA (Low-Rank Adaptation)**. Вместо того чтобы перезаписывать все 7 миллиардов параметров модели Qwen2.5-VL (что весило бы >15 ГБ), мы обучаем только дополнительные "адаптеры" — крошечные слои, которые накладываются поверх основной модели.

**В папке каждой эпохи вы найдете:**
1. `lora_adapter.npz` (~200 МБ) — обученные знания модели о виральности.
2. `viral_head.npz` (~5 МБ) — веса нашей финальной "головы", которая переводит выводы нейросети в числовой индекс.

### Как это использовать для предсказаний?
Сама база (огромный "замок" Qwen) остается неизменной в кэше вашего компьютера. Чтобы запустить готовую модель, скрипт инференса будет:
1. Загружать чистую модель `Qwen2.5-VL-7B-Instruct-4bit`.
2. Накладывать сверху ваш `lora_adapter.npz`.
3. Подключать `viral_head.npz`.

Это позволяет экономить десятки гигабайт места на диске и делиться обученной моделью, просто пересылая маленькие файлы.

## Inference by video_id (SQLite + GCS)

### 1) Prepare SQLite DB
Create a JSONL file with one object per line:
```json
{"video_id":"abc123","gcs_uri":"gs://my-bucket/path/video.mp4","transcript":"...","system_prompt":"...","metadata":{"title":"..."}}
```
Ingest it:
```bash
python scripts/ingest_metadata.py --input-jsonl data/videos.jsonl
```

### 2) Authenticate to GCS (ADC)
```bash
gcloud auth application-default login
```

### 3) Run inference
```bash
python scripts/run_inference.py --video-id abc123 --checkpoint-dir checkpoints/epoch_3
```

Optional JSON output:
```bash
python scripts/run_inference.py --video-id abc123 --out-json outputs/abc123.json
```

Notes:
- `gcs_uri` can also be a local file path for quick testing.
- Default cache dir: `.cache/videos` (disable with `--no-cache`).
