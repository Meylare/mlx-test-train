import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import os
import time
from tqdm import tqdm

from src.models.viral_model import ViralPredictorModel, save_lora_weights
from src.data.dataset import ViralVideoDataset

# --- CONFIG ---
# Используем ваши готовые файлы
JSONL_TRAIN = "dataset_train_ready.jsonl"
JSONL_VAL = "dataset_val_ready.jsonl"
MODEL_PATH = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
LR = 1e-5
NUM_EPOCHS = 5
CHECKPOINT_DIR = "checkpoints"

def loss_fn(model, batch):
    # Прямой проход
    preds = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values_videos=batch.get("pixel_values_videos"),
        video_grid_thw=batch.get("video_grid_thw")
    )
    
    target = batch["label"]
    
    # Huber Loss (устойчив к выбросам)
    delta = 1.0
    err = preds - target
    abs_err = mx.abs(err)
    
    loss = mx.where(
        abs_err <= delta,
        0.5 * mx.square(err),
        delta * (abs_err - 0.5 * delta)
    )
    
    return mx.mean(loss)

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Загрузка модели: {MODEL_PATH}...")
    model = ViralPredictorModel(MODEL_PATH)
    
    # Оптимизатор
    optimizer = optim.AdamW(learning_rate=LR)
    
    # Функция для вычисления потерь и градиентов
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    print("Загрузка данных...")
    train_ds = ViralVideoDataset(JSONL_TRAIN, MODEL_PATH)
    val_ds = ViralVideoDataset(JSONL_VAL, MODEL_PATH) if os.path.exists(JSONL_VAL) else None

    print(f"🚀 Старт обучения ({len(train_ds)} примеров)...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        pbar = tqdm(range(len(train_ds)), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for i in pbar:
            batch = train_ds.get_sample(i)
            
            # Шаг градиентного спуска
            loss, grads = loss_and_grad_fn(model, batch)
            model.update(optimizer.apply_gradients(grads, model))
            
            # В MLX вычисления ленивые, mx.eval запускает их в Metal
            mx.eval(model.parameters(), optimizer.state)
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_ds)
        print(f"✅ Эпоха {epoch+1} завершена. Средний Loss: {avg_loss:.4f}")

        # Сохранение весов LoRA (только адаптеры, чтобы не тратить место)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"viral_lora_epoch_{epoch+1}.safetensors")
        save_lora_weights(model, ckpt_path)
        print(f"💾 Чекпоинт сохранен: {ckpt_path}")

if __name__ == "__main__":
    train()