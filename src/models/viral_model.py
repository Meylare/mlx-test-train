import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load
from mlx_vlm.utils import load_config
from typing import Optional, Tuple, Union, Any
import math

# ---------------------------------------------------------------------------
# LoRA Implementation for MLX
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.linear = linear
        
        # Получаем логические размерности слоя.
        # В MLX QuantizedLinear имеет атрибуты bits и weight.
        if hasattr(linear, "in_features"):
            input_dims = linear.in_features
            output_dims = linear.out_features
        elif hasattr(linear, "input_dims"):
            input_dims = linear.input_dims
            output_dims = linear.output_dims
        else:
            output_dims, input_dims = linear.weight.shape
            if hasattr(linear, "bits"):
                # Для квантованных слоев восстанавливаем реальную размерность
                input_dims *= (32 // linear.bits)
        
        # Дополнительная проверка для 4-битных моделей Qwen (3584 -> 448 packed)
        if input_dims == 448:
            input_dims = 3584
            
        self.lora_a = mx.random.normal((input_dims, r)) * (1 / math.sqrt(input_dims))
        self.lora_b = mx.zeros((r, output_dims))
        self.scale = alpha / r
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x: mx.array) -> mx.array:
        res = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return res + self.scale * z

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int, alpha: float, dropout: float):
        return cls(linear, r, alpha, dropout)

LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

def _apply_lora_to_module(module: nn.Module, lora_r: int, lora_alpha: float, lora_dropout: float) -> int:
    count = 0
    # Поддерживаем и обычные, и квантованные слои
    linear_classes = (nn.Linear,)
    if hasattr(nn, "QuantizedLinear"):
        linear_classes += (nn.QuantizedLinear,)
        
    # Проходим по всем именованным модулям 
    for name, child in module.named_modules():
        if isinstance(child, linear_classes):
            # Проверяем, является ли последний компонент имени целью для LoRA
            leaf_name = name.split(".")[-1]
            if leaf_name in LORA_TARGETS:
                # Находим родительский модуль
                parts = name.split(".")
                parent = module
                for part in parts[:-1]:
                    if hasattr(parent, part):
                        parent = getattr(parent, part)
                    elif isinstance(parent, (list, tuple)) and part.isdigit():
                        parent = parent[int(part)]
                    else:
                        parent = None
                        break
                
                if parent is not None:
                    # Заменяем обычный/квантованный Linear на наш LoRALinear
                    setattr(parent, parts[-1], LoRALinear.from_linear(child, lora_r, lora_alpha, lora_dropout))
                    count += 1
    return count

def freeze_non_lora(model: nn.Module) -> None:
    # Замораживаем всё
    model.freeze()
    
    # Размораживаем только то, что нужно
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Размораживаем весь LoRALinear, а затем точечно замораживаем базовый слой
            module.unfreeze()
            module.linear.freeze()
            
    # Размораживаем голову целиком
    if hasattr(model, "viral_head"):
        model.viral_head.unfreeze()

def save_lora_weights(model: nn.Module, path: str) -> None:
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            weights[f"{name}.lora_a"] = module.lora_a
            weights[f"{name}.lora_b"] = module.lora_b
        elif "viral_head" in name and isinstance(module, nn.Linear):
            # Сохраняем веса обучаемой головы
            weights[f"{name}.weight"] = module.weight
            if hasattr(module, "bias") and module.bias is not None:
                weights[f"{name}.bias"] = module.bias
    mx.savez(path, **weights)

# ---------------------------------------------------------------------------
# MLP Head & Main Model
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    def __init__(self, hidden_size: int = 3584):
        super().__init__()
        self.l1 = nn.Linear(hidden_size, 1024)
        self.l2 = nn.Linear(1024, 256)
        self.l3 = nn.Linear(256, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dropout(self.gelu(self.l1(x)))
        x = self.gelu(self.l2(x))
        return self.l3(x)

class ViralPredictorModel(nn.Module):
    def __init__(self, model_path: str, lora_r: int = 16, lora_alpha: float = 32.0, lora_dropout: float = 0.05):
        super().__init__()
        self.backbone, self.processor = load(model_path)
        self.config = load_config(model_path)

        # Определение hidden_size (размерность скрытого состояния)
        h_size = 3584  # Дефолт для Qwen2.5-VL-7B
        config = getattr(self.backbone, "config", None)
        if config:
            if hasattr(config, "text_config"):
                h_size = getattr(config.text_config, "hidden_size", h_size)
            elif hasattr(config, "hidden_size"):
                h_size = config.hidden_size
            elif isinstance(config, dict):
                if "text_config" in config:
                    h_size = config["text_config"].get("hidden_size", h_size)
                else:
                    h_size = config.get("hidden_size", h_size)

        # Применяем LoRA
        n_replaced = _apply_lora_to_module(self.backbone, lora_r, lora_alpha, lora_dropout)
        print(f"[ViralPredictorModel] LoRA applied to {n_replaced} layers")

        self.viral_head = MLPHead(h_size)
        freeze_non_lora(self)

    def __call__(self, input_ids, attention_mask, pixel_values_videos=None, video_grid_thw=None, **kwargs):
        # 1. Получаем эмбеддинги (текст + видео)
        input_features = self.backbone.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values_videos,
            video_grid_thw=video_grid_thw
        )
        
        # 2. Рассчитываем position_ids (нужно для Qwen2.5-VL)
        # Мы используем тот же механизм, что и в самом LanguageModel
        position_ids, _ = self.backbone.language_model.get_rope_index(
            input_ids,
            image_grid_thw=None, # Видео уже обработаны в get_input_embeddings
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask
        )

        # 3. Получаем скрытые состояния из языковой модели
        h = self.backbone.language_model.model(
            inputs=None,
            inputs_embeds=input_features.inputs_embeds,
            position_ids=position_ids
        )
        
        # h - это скрытые состояния последнего слоя [batch, seq_len, hidden_size]
        batch_size = input_ids.shape[0]
        
        # Находим индекс последнего токена по маске
        last_token_idx = mx.sum(attention_mask, axis=1).astype(mx.int32) - 1
        
        # Гарантируем, что индекс в пределах допустимого
        seq_len = h.shape[1]
        last_token_idx = mx.minimum(last_token_idx, seq_len - 1)
        
        # Берем вектор последнего токена
        h_final = h[mx.arange(batch_size), last_token_idx]
        
        return self.viral_head(h_final)