import json
import os

# --- НАСТРОЙКИ ---
INPUT_FILE = "dataset_val_ready_test.jsonl"       # Ваш исходный файл
OUTPUT_FILE = "dataset_val_ready_test2.jsonl"        # Куда сохранить результат
BUCKET_NAME = "slon_bucket2"          # Имя вашего бакета (БЕЗ gs://)
SUBFOLDER = "100k_videos"                           # Папка внутри бакета (оставьте пустым "", если файлы в корне)
# -----------------

print(f"Меняем пути в {INPUT_FILE} на gs://{BUCKET_NAME}...")

processed_count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    
    for line in f_in:
        line = line.strip()
        if not line: continue
        
        try:
            data = json.loads(line)
            
            # 1. Ищем текущий путь (пробуем разные ключи)
            current_path = data.get("gcs_uri") or data.get("video_path") or data.get("path")
            
            if not current_path:
                print(f"Пропуск строки {processed_count+1}: нет пути к видео")
                continue

            # 2. Вытаскиваем только имя файла (например, "video123.mp4")
            # os.path.basename сработает и для "/local/path/vid.mp4" и для "gs://old/vid.mp4"
            filename = os.path.basename(current_path)
            
            # 3. Собираем новый путь
            if SUBFOLDER:
                new_uri = f"gs://{BUCKET_NAME}/{SUBFOLDER}/{filename}"
            else:
                new_uri = f"gs://{BUCKET_NAME}/{filename}"
            
            # 4. Обновляем поле gcs_uri (оно главное для ingest_metadata.py)
            data["gcs_uri"] = new_uri
            
            # Если был ключ video_path, обновим и его для совместимости
            if "video_path" in data:
                data["video_path"] = new_uri

            # 5. Сохраняем
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            processed_count += 1
            
        except json.JSONDecodeError:
            print(f"Ошибка JSON в строке {processed_count+1}")

print(f"Готово! Обработано видео: {processed_count}")
print(f"Результат сохранен в: {OUTPUT_FILE}")