import os
import json
import argparse
from pathlib import Path
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def get_video_ids(jsonl_paths):
    """Собирает все уникальные ID из файлов выборки."""
    video_ids = set()
    for path in jsonl_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        video_ids.add(json.loads(line)['id'])
    return video_ids

def download_video(video_id, bucket, output_dir):
    """Скачивает одно видео по ID."""
    blob_name = f"100k_videos/{video_id}.mp4" # Путь внутри бакета (подправьте, если другой)
    blob = bucket.blob(blob_name)
    output_path = output_dir / f"{video_id}.mp4"
    
    if output_path.exists():
        return False # Уже скачано
    
    try:
        blob.download_to_filename(str(output_path))
        return True
    except Exception:
        # Попробуем без префикса videos/ если не нашли
        try:
            blob = bucket.blob(f"{video_id}.mp4")
            blob.download_to_filename(str(output_path))
            return True
        except:
            return None # Ошибка скачивания

def main():
    parser = argparse.ArgumentParser(description="Скачивание видео из GCP Bucket")
    parser.add_argument("--key", required=True, help="Путь к JSON ключу GCP")
    parser.add_argument("--bucket", required=True, help="Имя бакета в GCP")
    parser.add_argument("--output", default="data/videos", help="Папка для сохранения")
    args = parser.parse_args()

    # 1. Настройка GCP
    client = storage.Client.from_service_account_json(args.key)
    bucket = client.bucket(args.bucket)

    # 2. Создание папки
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Получение списка ID
    print("Собираю список видео из JSONL...")
    ids = get_video_ids(["dataset_train_ready.jsonl", "dataset_val_ready.jsonl"])
    print(f"Нужно скачать видео: {len(ids)}")

    # 4. Многопоточное скачивание (быстрее)
    print("Начинаю скачивание...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(
            executor.map(lambda vid: download_video(video_id=vid, bucket=bucket, output_dir=output_dir), ids),
            total=len(ids)
        ))

    downloaded = sum(1 for r in results if r is True)
    existed = sum(1 for r in results if r is False)
    errors = sum(1 for r in results if r is None)

    print(f"\nГотово!")
    print(f"Скачано новых: {downloaded}")
    print(f"Уже было на диске: {existed}")
    if errors > 0:
        print(f"Не удалось найти в бакете: {errors}")

if __name__ == "__main__":
    main()