import json
import random
import os

input_file = "dataset_val_filtered.jsonl"

# Устанавливаем seed на основе текущего времени для более случайного перемешивания
random.seed(os.urandom(16))

print("Читаю файл...")
videos = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            videos.append(json.loads(line.strip()))

print(f"Всего записей: {len(videos)}")

# Показываем первые 5 VI до перемешивания
print("\nПервые 5 VI до перемешивания:")
for i, v in enumerate(videos[:5]):
    print(f"  {i+1}. VI = {v['targets']['viral_index']:.4f}")

# Перемешиваем несколько раз для лучшей случайности
print("\nПеремешиваю записи...")
for _ in range(3):
    random.shuffle(videos)

# Показываем первые 5 VI после перемешивания
print("\nПервые 5 VI после перемешивания:")
for i, v in enumerate(videos[:5]):
    print(f"  {i+1}. VI = {v['targets']['viral_index']:.4f}")

# Сохраняем обратно
print("\nСохраняю перемешанный файл...")
with open(input_file, "w", encoding="utf-8") as f:
    for video in videos:
        f.write(json.dumps(video, ensure_ascii=False) + "\n")

print("Готово! Файл перемешан.")
