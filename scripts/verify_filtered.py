import json

# Читаем отфильтрованный файл
with open("dataset_val_filtered.jsonl", "r", encoding="utf-8") as f:
    videos = [json.loads(line.strip()) for line in f if line.strip()]

# Получаем все VI
viral_indices = [v["targets"]["viral_index"] for v in videos]
viral_indices.sort(reverse=True)

print(f"Всего видео: {len(videos)}")
print(f"\nТоп-10 самых высоких VI:")
for i, vi in enumerate(viral_indices[:10], 1):
    print(f"  {i}. {vi:.4f}")

print(f"\nТоп-10 самых низких VI:")
for i, vi in enumerate(viral_indices[-10:], 1):
    print(f"  {i}. {vi:.4f}")

# Проверяем распределение
high_vi = [vi for vi in viral_indices if vi > 1]
low_vi = [vi for vi in viral_indices if vi < -1]
middle_vi = [vi for vi in viral_indices if -1 <= vi <= 1]

print(f"\nРаспределение:")
print(f"  VI > 1: {len(high_vi)} видео")
print(f"  VI < -1: {len(low_vi)} видео")
print(f"  -1 <= VI <= 1: {len(middle_vi)} видео")
