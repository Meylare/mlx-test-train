"""
Скрипт для визуализации распределения Viral index по JSONL датасету.
Требуется: pip install matplotlib
"""
import json
import matplotlib
matplotlib.use("Agg")  # без GUI, только сохранение в файл
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_viral_indices(path: Path) -> list[float]:
    """Загружает viral_index из каждой строки JSONL."""
    vi_list = []
    if not path.exists():
        print(f"Ошибка: Файл {path} не найден.")
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Пробуем достать из targets.viral_index
                vi = obj.get("targets", {}).get("viral_index")
                if vi is not None:
                    vi_list.append(float(vi))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return vi_list

def main():
    parser = argparse.ArgumentParser(description="Визуализация Viral Index")
    parser.add_argument("input", nargs="?", default="dataset_train.jsonl", help="Путь к JSONL файлу")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    print(f"Обработка файла: {input_path}")
    vi = load_viral_indices(input_path)
    
    if not vi:
        print("Нет данных viral_index для отображения.")
        return

    vi = np.array(vi)
    print(f"Загружено записей: {len(vi)}")
    print(f"Viral index: min={vi.min():.3f}, max={vi.max():.3f}, mean={vi.mean():.3f}, std={vi.std():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Гистограмма
    ax1 = axes[0]
    ax1.hist(vi, bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Viral index")
    ax1.set_ylabel("Количество видео")
    ax1.set_title(f"Распределение Viral index\n({input_path.name})")
    ax1.grid(True, alpha=0.3)

    # Опционально: KDE (плотность)
    ax2 = axes[1]
    ax2.hist(vi, bins=50, density=True, edgecolor="black", alpha=0.7, label="Гистограмма (норм.)")
    try:
        from scipy import stats
        kde = stats.gaussian_kde(vi)
        x_range = np.linspace(vi.min(), vi.max(), 200)
        ax2.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
    except ImportError:
        print("Примечание: scipy не установлен, KDE не будет отрисован.")
    
    ax2.set_xlabel("Viral index")
    ax2.set_ylabel("Плотность")
    ax2.set_title("Плотность распределения")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_name = f"viral_index_{input_path.stem}.png"
    out_path = PROJECT_ROOT / output_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"График сохранён: {out_path}")

if __name__ == "__main__":
    main()
