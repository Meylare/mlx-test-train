"""
MLX training loop for the Viral Predictor.

Uses ``nn.value_and_grad`` for gradient computation, manual gradient
accumulation, Huber loss, and separate checkpoint saving for LoRA adapters
and the MLP head.
"""

import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from tqdm import tqdm

from src.data.dataset import ViralVideoDataset
from src.models.viral_model import ViralPredictorModel, save_lora_weights

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JSONL_TRAIN = "dataset_val_filtered.jsonl"   # Обучающий датасет
JSONL_VAL = ""                                # Валидационный (пока нет)
MODEL_PATH = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 8    # Эффективный батч = 1 * 8 = 8
LR = 2e-4
NUM_EPOCHS = 3
MAX_FRAMES = 10
CHECKPOINT_DIR = "checkpoints"


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def huber_loss(pred: mx.array, target: mx.array, delta: float = 1.0) -> mx.array:
    """Huber loss (smooth L1), устойчив к выбросам viral_index."""
    diff = pred.squeeze() - target.squeeze()
    abs_diff = mx.abs(diff)
    loss = mx.where(
        abs_diff <= delta,
        0.5 * diff * diff,
        delta * (abs_diff - 0.5 * delta),
    )
    return mx.mean(loss)


# ---------------------------------------------------------------------------
# Loss function for nn.value_and_grad
# ---------------------------------------------------------------------------

def make_loss_fn(model: ViralPredictorModel):
    """
    Возвращает замыкание loss_fn(batch) -> scalar loss.
    nn.value_and_grad дифференцирует по параметрам model.
    """
    def loss_fn(batch: dict) -> mx.array:
        preds = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values_videos=batch.get("pixel_values_videos"),
            video_grid_thw=batch.get("video_grid_thw"),
        )
        return huber_loss(preds, batch["labels"])

    return loss_fn


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: ViralPredictorModel, epoch_dir: str) -> None:
    """Сохраняет LoRA-адаптеры и MLP-голову отдельно."""
    os.makedirs(epoch_dir, exist_ok=True)

    # 1. LoRA адаптеры (~200MB)
    lora_path = os.path.join(epoch_dir, "lora_adapter.npz")
    save_lora_weights(model, lora_path)
    print(f"  LoRA saved: {lora_path}")

    # 2. MLP голова (~5MB)
    head_path = os.path.join(epoch_dir, "viral_head.npz")
    head_params = {}
    _flatten_params(dict(model.viral_head.parameters()), head_params)
    mx.savez(head_path, **head_params)
    print(f"  MLP head saved: {head_path}")


def _flatten_params(tree: dict, out: dict, prefix: str = "") -> None:
    """Разворачивает вложенный dict параметров в плоский для mx.savez."""
    for k, v in tree.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, mx.array):
            out[name] = v
        elif isinstance(v, dict):
            _flatten_params(v, out, name)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model: ViralPredictorModel, val_dataset: ViralVideoDataset) -> float:
    """Прогоняет валидацию, возвращает средний loss."""
    total_loss = 0.0
    n = 0
    for batch in val_dataset.batch_iterator(batch_size=1, shuffle=False):
        preds = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values_videos=batch.get("pixel_values_videos"),
            video_grid_thw=batch.get("video_grid_thw"),
        )
        loss = huber_loss(preds, batch["labels"])
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    """Полный цикл обучения."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Загрузка модели ----------------------------------------------------
    print(f"Загрузка модели: {MODEL_PATH} ...")
    model = ViralPredictorModel(MODEL_PATH)

    # 2. Загрузка датасетов -------------------------------------------------
    print("Загрузка данных ...")
    train_dataset = ViralVideoDataset(
        jsonl_path=JSONL_TRAIN,
        processor=model.processor,
        max_frames=MAX_FRAMES,
    )

    val_dataset = None
    if JSONL_VAL and os.path.exists(JSONL_VAL):
        val_dataset = ViralVideoDataset(
            jsonl_path=JSONL_VAL,
            processor=model.processor,
            max_frames=MAX_FRAMES,
        )

    # 3. Оптимизатор --------------------------------------------------------
    optimizer = optim.AdamW(learning_rate=LR)

    # 4. Функция градиентов -------------------------------------------------
    loss_fn = make_loss_fn(model)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # 5. Тренировочный цикл -------------------------------------------------
    print(f"Старт обучения: {NUM_EPOCHS} эпох, "
          f"grad_accum={GRAD_ACCUMULATION_STEPS}, lr={LR}, "
          f"samples={len(train_dataset)}")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        n_steps = 0
        accumulated_grads = None
        accum_count = 0
        epoch_start = time.time()

        progress = tqdm(
            train_dataset.batch_iterator(batch_size=BATCH_SIZE, shuffle=True),
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            total=len(train_dataset) // BATCH_SIZE,
        )

        for step, batch in enumerate(progress):
            # --- Forward + backward ---
            loss, grads = loss_and_grad_fn(batch)

            # --- Gradient accumulation ---
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(
                    lambda a, b: a + b, accumulated_grads, grads
                )
            accum_count += 1

            # --- Update weights every GRAD_ACCUMULATION_STEPS ---
            if accum_count >= GRAD_ACCUMULATION_STEPS:
                # Усредняем градиенты
                accumulated_grads = tree_map(
                    lambda g: g * (1.0 / GRAD_ACCUMULATION_STEPS),
                    accumulated_grads,
                )
                # Применяем к модели
                optimizer.update(model, accumulated_grads)
                # Принудительное вычисление (MLX ленивый)
                mx.eval(model.parameters(), optimizer.state)

                accumulated_grads = None
                accum_count = 0

            step_loss = loss.item()
            total_loss += step_loss
            n_steps += 1
            progress.set_postfix({"loss": f"{step_loss:.4f}"})

        # --- Итоги эпохи ---
        avg_loss = total_loss / max(n_steps, 1)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch + 1} done.  "
              f"Train loss: {avg_loss:.4f}  "
              f"Time: {elapsed:.0f}s")

        # --- Валидация ---
        if val_dataset is not None:
            val_loss = validate(model, val_dataset)
            print(f"  Val loss: {val_loss:.4f}")

        # --- Сохранение чекпоинта ---
        epoch_dir = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}")
        save_checkpoint(model, epoch_dir)

    print("-" * 60)
    print("Обучение завершено.")


if __name__ == "__main__":
    train()
