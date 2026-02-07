import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from tqdm import tqdm

from src.data.dataset import ViralVideoDataset
from src.models.viral_model import ViralPredictorModel, save_lora_weights

# Конфигурация
JSONL_TRAIN = "dataset_train_ready.jsonl"
MODEL_PATH = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 8
LR = 2e-4
NUM_EPOCHS = 3
MAX_FRAMES = 6  # Начнем с 6 кадров для стабильности на 18ГБ
CHECKPOINT_DIR = "checkpoints"

def huber_loss(pred: mx.array, target: mx.array, delta: float = 1.0) -> mx.array:
    diff = pred.squeeze() - target.squeeze()
    abs_diff = mx.abs(diff)
    loss = mx.where(abs_diff <= delta, 0.5 * diff * diff, delta * (abs_diff - 0.5 * delta))
    return mx.mean(loss)

def make_loss_fn(model: ViralPredictorModel):
    def loss_fn(batch: dict) -> mx.array:
        preds = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values_videos=batch.get("pixel_values_videos"),
            video_grid_thw=batch.get("video_grid_thw"),
        )
        return huber_loss(preds, batch["labels"])
    return loss_fn

def save_checkpoint(model: ViralPredictorModel, epoch_dir: str):
    os.makedirs(epoch_dir, exist_ok=True)
    save_lora_weights(model, os.path.join(epoch_dir, "lora_adapter.npz"))
    # Сохранение головы (MLP)
    head_params = {}
    def _flatten(tree, out, prefix=""):
        for k, v in tree.items():
            name = f"{prefix}.{k}" if prefix else k
            if isinstance(v, mx.array): out[name] = v
            elif isinstance(v, dict): _flatten(v, out, name)
    _flatten(dict(model.viral_head.parameters()), head_params)
    mx.savez(os.path.join(epoch_dir, "viral_head.npz"), **head_params)

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"Загрузка модели: {MODEL_PATH} ...")
    model = ViralPredictorModel(MODEL_PATH)

    print("Загрузка датасета...")
    train_dataset = ViralVideoDataset(JSONL_TRAIN, model.processor, MAX_FRAMES)
    
    optimizer = optim.AdamW(learning_rate=LR)
    loss_and_grad_fn = nn.value_and_grad(model, make_loss_fn(model))

    print(f"Старт обучения. Эффективный батч: {GRAD_ACCUMULATION_STEPS}")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        n_steps = 0
        accumulated_grads = None
        accum_count = 0
        
        progress = tqdm(train_dataset.batch_iterator(BATCH_SIZE), total=len(train_dataset))

        for step, batch in enumerate(progress):
            # 1. Считаем градиенты
            loss, grads = loss_and_grad_fn(batch)
            
            # 2. МОМЕНТАЛЬНОЕ ВЫЧИСЛЕНИЕ (Освобождает RAM)
            mx.eval(loss, grads)

            # 3. Аккумуляция
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, grads)
            accum_count += 1

            # 4. Обновление весов каждые 8 шагов
            if accum_count >= GRAD_ACCUMULATION_STEPS:
                accumulated_grads = tree_map(lambda g: g * (1.0 / GRAD_ACCUMULATION_STEPS), accumulated_grads)
                optimizer.update(model, accumulated_grads)
                
                # Очистка памяти после обновления
                mx.eval(model.parameters(), optimizer.state)
                
                accumulated_grads = None
                accum_count = 0

            step_loss = loss.item()
            total_loss += step_loss
            n_steps += 1
            progress.set_postfix({"loss": f"{step_loss:.4f}"})

        print(f"Epoch {epoch+1} завершена. Средний loss: {total_loss/n_steps:.4f}")
        save_checkpoint(model, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}"))

if __name__ == "__main__":
    train()