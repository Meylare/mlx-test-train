import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load
from mlx_vlm.utils import load_config
from typing import Optional, Tuple, Union, Any, Dict
import math
import numpy as np


# ---------------------------------------------------------------------------
# Совместимость: named_modules() для любой версии MLX
# ---------------------------------------------------------------------------

def _named_modules(module: nn.Module, prefix: str = ""):
    """
    Рекурсивно обходит дерево модулей, возвращая (name, module).
    Работает и если nn.Module.named_modules() есть, и если нет.
    """
    # Если mlx-vlm / новая версия MLX уже добавила named_modules — используем
    if hasattr(module, "named_modules") and callable(module.named_modules):
        yield from module.named_modules()
        return

    # Иначе обходим вручную через children()
    yield prefix, module
    children = module.children() if callable(getattr(module, "children", None)) else {}
    if isinstance(children, dict):
        for name, child in children.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Module):
                yield from _named_modules(child, child_prefix)
            elif isinstance(child, (list, tuple)):
                for i, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        yield from _named_modules(item, f"{child_prefix}.{i}")


# ---------------------------------------------------------------------------
# LoRA Implementation for MLX
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.linear = linear
        
        # Получаем логические размерности слоя.
        # Приоритет: атрибуты .input_dims / .in_features -> вычисление из weight.shape
        if hasattr(linear, "input_dims"):
            # QuantizedLinear в MLX хранит логические размерности
            input_dims = linear.input_dims
            output_dims = linear.output_dims
        elif hasattr(linear, "in_features"):
            input_dims = linear.in_features
            output_dims = linear.out_features
        else:
            # Fallback: вычисляем из формы весов
            output_dims, packed_input = linear.weight.shape
            if hasattr(linear, "bits") and linear.bits < 32:
                # Квантованный слой: weight.shape[1] = input_dims * bits / 32
                input_dims = packed_input * (32 // linear.bits)
            else:
                input_dims = packed_input
            
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
    for name, child in _named_modules(module):
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
    for name, module in _named_modules(model):
        if isinstance(module, LoRALinear):
            # Размораживаем весь LoRALinear, а затем точечно замораживаем базовый слой
            module.unfreeze()
            module.linear.freeze()
            
    # Размораживаем голову целиком
    if hasattr(model, "viral_head"):
        model.viral_head.unfreeze()

def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only LoRA adapter weights. MLP head is saved separately by the trainer (viral_head.npz)."""
    weights = {}
    for name, module in _named_modules(model):
        if isinstance(module, LoRALinear):
            weights[f"{name}.lora_a"] = module.lora_a
            weights[f"{name}.lora_b"] = module.lora_b
    mx.savez(path, **weights)


def load_lora_weights(model: nn.Module, path: str) -> None:
    with np.load(path) as data:
        tensors: Dict[str, mx.array] = {k: mx.array(data[k]) for k in data.files}

    expected = []
    for name, module in _named_modules(model):
        if isinstance(module, LoRALinear):
            expected.append(f"{name}.lora_a")
            expected.append(f"{name}.lora_b")

    missing = [k for k in expected if k not in tensors]
    extra = [k for k in tensors.keys() if k not in expected]
    if missing or extra:
        raise ValueError(
            f"LoRA weights mismatch. Missing: {missing}, Extra: {extra}"
        )

    for name, module in _named_modules(model):
        if isinstance(module, LoRALinear):
            key_a = f"{name}.lora_a"
            key_b = f"{name}.lora_b"
            w_a = tensors[key_a]
            w_b = tensors[key_b]
            if module.lora_a.shape != w_a.shape or module.lora_b.shape != w_b.shape:
                raise ValueError(
                    f"LoRA shape mismatch for {name}: "
                    f"expected {module.lora_a.shape}/{module.lora_b.shape}, "
                    f"got {w_a.shape}/{w_b.shape}"
                )
            module.lora_a = w_a
            module.lora_b = w_b


def load_viral_head(model: nn.Module, path: str) -> None:
    with np.load(path) as data:
        tensors: Dict[str, mx.array] = {k: mx.array(data[k]) for k in data.files}

    if not hasattr(model, "viral_head"):
        raise ValueError("Model has no viral_head to load.")

    expected: Dict[str, mx.array] = {}

    def _flatten(tree: Dict[str, Any], out: Dict[str, mx.array], prefix: str = "") -> None:
        for k, v in tree.items():
            name = f"{prefix}.{k}" if prefix else k
            if isinstance(v, mx.array):
                out[name] = v
            elif isinstance(v, dict):
                _flatten(v, out, name)

    _flatten(dict(model.viral_head.parameters()), expected)

    missing = [k for k in expected.keys() if k not in tensors]
    extra = [k for k in tensors.keys() if k not in expected]
    if missing or extra:
        raise ValueError(
            f"Viral head weights mismatch. Missing: {missing}, Extra: {extra}"
        )

    def _set_by_name(root: nn.Module, name: str, value: mx.array) -> None:
        parts = name.split(".")
        module = root
        for part in parts[:-1]:
            if not hasattr(module, part):
                raise ValueError(f"Missing module path: {name}")
            module = getattr(module, part)
        if not hasattr(module, parts[-1]):
            raise ValueError(f"Missing parameter: {name}")
        setattr(module, parts[-1], value)

    for name, value in tensors.items():
        if expected[name].shape != value.shape:
            raise ValueError(
                f"Viral head shape mismatch for {name}: "
                f"expected {expected[name].shape}, got {value.shape}"
            )
        _set_by_name(model.viral_head, name, value)

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
