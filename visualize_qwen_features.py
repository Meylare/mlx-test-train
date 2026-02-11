"""
Скрипт для визуализации важности признаков обученной Qwen модели (MLX версия).
Создает тепловые карты attention, важность токенов и кадров видео.
"""

import sys
import os

# Проверяем тестовый режим ДО всех импортов
TEST_MODE = '--test_mode' in sys.argv

# Если тестовый режим, перенаправляем на test_visualize.py
if TEST_MODE and __name__ == "__main__":
    # Добавляем путь к проекту
    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Запускаем тестовый скрипт напрямую через subprocess чтобы избежать циклического импорта
    import subprocess
    test_script = os.path.join(os.path.dirname(__file__), 'test_visualize.py')
    result = subprocess.run([sys.executable, test_script], cwd=project_root)
    sys.exit(result.returncode)

# Обычные импорты для реального режима
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple, List
import json
from PIL import Image
import platform
import importlib.util

# Добавляем корневую директорию проекта в sys.path для импортов
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Попытка импорта cv2 с fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2 не доступен, будет использован numpy fallback для наложения")

# Импорты моделей выполняются только при необходимости
ViralPredictorModel = None
ViralVideoDataset = None
load_lora_weights = None
load_viral_head = None

def _lazy_import_models():
    """Ленивый импорт моделей только когда они действительно нужны."""
    global ViralPredictorModel, ViralVideoDataset
    if ViralPredictorModel is None:
        from src.models.viral_model import ViralPredictorModel
        from src.data.dataset import ViralVideoDataset
        ViralPredictorModel = ViralPredictorModel
        ViralVideoDataset = ViralVideoDataset


def is_macos():
    """Проверяет, запущено ли на macOS."""
    return platform.system() == "Darwin"


def _resize_heatmap_numpy(heatmap_2d: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Простое масштабирование тепловой карты через numpy (fallback)."""
    src_h, src_w = heatmap_2d.shape
    
    # Используем простое повторение и обрезку
    if target_h >= src_h and target_w >= src_w:
        # Увеличиваем через повторение
        repeat_h = target_h // src_h
        repeat_w = target_w // src_w
        resized = np.repeat(np.repeat(heatmap_2d, repeat_h, axis=0), repeat_w, axis=1)
        # Обрезаем до нужного размера
        resized = resized[:target_h, :target_w]
    else:
        # Уменьшаем через усреднение
        step_h = src_h / target_h
        step_w = src_w / target_w
        resized = np.zeros((target_h, target_w))
        for i in range(target_h):
            for j in range(target_w):
                start_h = int(i * step_h)
                end_h = int((i + 1) * step_h)
                start_w = int(j * step_w)
                end_w = int((j + 1) * step_w)
                resized[i, j] = heatmap_2d[start_h:end_h, start_w:end_w].mean()
    
    return resized



def _clean_decode_token(processor, token_id: int) -> str:
    """
    Декодирует токен в читаемый вид, исправляя кириллицу и спецсимволы.
    """
    # Декодируем ID обратно в строку
    text = processor.decode([token_id])
    
    # Обработка спецсимволов для красоты графика
    if text == '\n':
        return '\\n'  # Чтобы перенос строки не ломал график
    if text.strip() == '':
        # Если это просто пробел или пустой символ
        if text == ' ':
            return '[SPACE]'
        # Проверяем, не спецтокен ли это
        raw_token = processor.tokenizer.convert_ids_to_tokens([token_id])[0]
        return raw_token # Возвращаем как есть (например, <|im_start|>)
        
    return text.strip()

# ============================================================================
# ФУНКЦИИ ВИЗУАЛИЗАЦИИ (не требуют импорта моделей)
# ============================================================================

def create_attention_heatmap(
    attention_weights: mx.array,
    input_ids: mx.array,
    processor,
    output_path: str,
    max_tokens: int = 30 # Уменьшил дефолт до 30, чтобы надписи влезали
):
    """Создает тепловую карту attention weights (Исправленная версия)."""
    # Конвертируем MLX массив в numpy
    if isinstance(attention_weights, mx.array):
        attn_np = np.array(attention_weights)
    else:
        attn_np = attention_weights
    
    if isinstance(input_ids, mx.array):
        input_ids_np = np.array(input_ids)
    else:
        input_ids_np = input_ids
    
    # Ограничиваем размер для читаемости
    seq_len = min(attn_np.shape[0], max_tokens)
    attn_matrix = attn_np[:seq_len, :seq_len]
    
    # === ИЗМЕНЕНИЕ: Правильное декодирование токенов ===
    token_ids = input_ids_np[:seq_len]
    token_labels = []
    for tid in token_ids:
        label = _clean_decode_token(processor, int(tid))
        if len(label) > 10:
            label = label[:8] + '..'
        token_labels.append(label)
    # ===================================================
    
    # Создание графика
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        attn_matrix,
        xticklabels=token_labels,
        yticklabels=token_labels,
        cmap='viridis', # Более контрастная карта
        cbar_kws={'label': 'Attention Weight'},
        linewidths=0.5,
        linecolor='white',
        square=True
    )
    
    plt.title('Матрица внимания (Attention Map)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('На что смотрят (Key)', fontsize=12)
    plt.ylabel('Кто смотрит (Query)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Тепловая карта attention сохранена: {output_path}")
    plt.close()


def create_token_importance_plot(
    token_importance: mx.array,
    input_ids: mx.array,
    processor,
    output_path: str,
    top_n: int = 20
):
    """
    Создает визуализацию, ИГНОРИРУЯ служебные токены.
    """
    if token_importance is None:
        return
    
    # Конвертируем в numpy
    importance_np = np.array(token_importance) if isinstance(token_importance, mx.array) else token_importance
    token_ids = np.array(input_ids) if isinstance(input_ids, mx.array) else input_ids
    
    # Список токенов, которые мы НЕ хотим видеть на графике
    BANNED_TOKENS = {
        '<|im_start|>', '<|im_end|>', 'assistant', 'user', 'system', 
        '\n', ':', '.', ',', '_', 'Ċ', 'Ġ', 'video', '<|video_pad|>',
        '<|vision_start|>', '<|vision_end|>'
    }

    # Собираем пары (важность, слово)
    candidates = []
    
    for idx, (imp, tid) in enumerate(zip(importance_np, token_ids)):
        # Декодируем
        word = processor.decode([int(tid)]).strip()
        
        # Пропускаем, если это мусор, пустота или спецтег
        if not word: continue
        if word in BANNED_TOKENS: continue
        if len(word) < 2: continue # Пропускаем буквы-одиночки
        if word.startswith('<') and word.endswith('>'): continue # Пропускаем теги
        
        candidates.append((imp, word))
    
    # Если после фильтрации ничего не осталось (бывает на ранних этапах), берем хоть что-то
    if not candidates:
        logger.warning("Все токены были отфильтрованы как мусор. Показываем сырые.")
        candidates = [(importance_np[i], f"Token_{i}") for i in range(min(top_n, len(importance_np)))]

    # Сортируем по важности
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Берем топ-N
    top_candidates = candidates[:top_n]
    
    # Разделяем обратно для графика
    top_vals = [x[0] for x in top_candidates]
    top_labels = [x[1] for x in top_candidates]
    
    # Рисуем (разворачиваем, чтобы самый важный был наверху)
    plt.figure(figsize=(10, 8))
    
    # Используем приятную цветовую схему
    colors = plt.cm.magma(np.linspace(0.3, 0.8, len(top_vals)))
    
    y_pos = np.arange(len(top_vals))
    plt.barh(y_pos, top_vals, color=colors, edgecolor='none')
    plt.yticks(y_pos, top_labels, fontsize=12)
    plt.gca().invert_yaxis() # Самый важный сверху
    
    plt.xlabel('Важность (без учета спецтегов)', fontsize=12, fontweight='bold')
    plt.title(f'Топ-{len(top_vals)} смысловых слов', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ График (фильтрованный) сохранен: {output_path}")
    plt.close()

def create_frame_importance_plot(
    frame_importance: mx.array,
    output_path: str
):
    """Создает визуализацию важности кадров видео."""
    if frame_importance is None:
        logger.warning("Frame importance не вычислена")
        return
    
    # Конвертируем MLX массив в numpy
    if isinstance(frame_importance, mx.array):
        importance_np = np.array(frame_importance)
    elif isinstance(frame_importance, list):
        importance_np = np.array(frame_importance)
    else:
        importance_np = frame_importance
    
    num_frames = len(importance_np)
    
    # Создание графика
    plt.figure(figsize=(max(10, num_frames * 0.3), 6))
    plt.plot(range(num_frames), importance_np, marker='o', linewidth=2, markersize=8, color='#FF6B35')
    plt.fill_between(range(num_frames), importance_np, alpha=0.3, color='#FF6B35')
    
    plt.xlabel('Номер кадра', fontsize=12, fontweight='bold')
    plt.ylabel('Важность кадра (Gradient Magnitude)', fontsize=12, fontweight='bold')
    plt.title('Важность кадров видео для предсказания', fontsize=14, fontweight='bold', pad=20)
    plt.grid(alpha=0.3, linestyle='--')
    plt.xticks(range(num_frames))
    
    # Выделяем топ-3 кадра
    top_3_indices = np.argsort(importance_np)[-3:][::-1]
    for idx in top_3_indices:
        plt.scatter([idx], [importance_np[idx]], s=200, color='red', marker='*', zorder=5)
        plt.text(idx, importance_np[idx] + max(importance_np) * 0.05, f'#{idx}', 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ График важности кадров сохранен: {output_path}")
    plt.close()


def map_tokens_to_frames(
    video_token_importance: mx.array,
    video_grid_thw: Optional[mx.array] = None,
    num_frames: int = 10
) -> List[float]:
    """Сопоставляет важность видео-токенов с кадрами."""
    if video_token_importance is None:
        return None
    
    # Конвертируем MLX массивы в numpy
    if isinstance(video_token_importance, mx.array):
        importance_np = np.array(video_token_importance)
    else:
        importance_np = video_token_importance
    
    if video_grid_thw is not None:
        if isinstance(video_grid_thw, mx.array):
            grid_np = np.array(video_grid_thw)
        else:
            grid_np = video_grid_thw
        
        # Qwen2.5-VL объединяет 2x2 патча в 1 токен, поэтому делим на 4
        T, H, W = grid_np[0].astype(int)
        tokens_per_frame = (H * W) // 4
        
        frame_importance_list = []
        for t in range(T):
            start_idx = t * tokens_per_frame
            end_idx = min(start_idx + tokens_per_frame, len(importance_np))
            frame_tokens = importance_np[start_idx:end_idx]
            frame_importance_list.append(float(frame_tokens.mean()))
        
        return frame_importance_list
    else:
        # Fallback: равномерное распределение
        tokens_per_frame = len(importance_np) // num_frames
        frame_importance_list = []
        for t in range(num_frames):
            start_idx = t * tokens_per_frame
            end_idx = min(start_idx + tokens_per_frame, len(importance_np))
            frame_tokens = importance_np[start_idx:end_idx]
            frame_importance_list.append(float(frame_tokens.mean()))
        
        return frame_importance_list


def create_video_token_heatmap_on_frames(
    frames: List[np.ndarray],
    video_token_importance: mx.array,
    video_grid_thw: Optional[mx.array] = None,
    output_path: str = None,
    alpha: float = 0.5,
    frame_importance_list: Optional[List[float]] = None  # Для обратной совместимости
):
    """
    Создает пространственную тепловую карту важности видео-токенов, наложенную на реальные кадры.
    
    Args:
        frames: Список реальных кадров видео [num_frames] каждый [H_frame, W_frame, 3]
        video_token_importance: Важность каждого видео-токена [num_video_tokens]
        video_grid_thw: Структура токенов [batch, 3] -> (T, H_tokens, W_tokens)
        output_path: Путь для сохранения
        alpha: Прозрачность тепловой карты
        frame_importance_list: Устаревший параметр (для обратной совместимости)
    """
    # Обратная совместимость: если передан frame_importance_list без video_token_importance
    if video_token_importance is None and frame_importance_list is not None:
        logger.warning("Используется устаревший режим с frame_importance_list")
        return _create_simple_frame_heatmap(frames, frame_importance_list, output_path, alpha)
    
    if video_token_importance is None:
        logger.warning("video_token_importance не предоставлена")
        return
    
    # Конвертируем MLX массивы в numpy
    if isinstance(video_token_importance, mx.array):
        importance_np = np.array(video_token_importance)
    else:
        importance_np = video_token_importance
    
    if video_grid_thw is None:
        logger.warning("video_grid_thw не предоставлен, используем равномерное распределение")
        num_frames = len(frames)
        tokens_per_frame = len(importance_np) // num_frames
        frame_importance_list = []
        for t in range(num_frames):
            start_idx = t * tokens_per_frame
            end_idx = min(start_idx + tokens_per_frame, len(importance_np))
            frame_importance_list.append(float(importance_np[start_idx:end_idx].mean()))
        return _create_simple_frame_heatmap(frames, frame_importance_list, output_path, alpha)
    
    # Извлекаем структуру токенов
    if isinstance(video_grid_thw, mx.array):
        grid_np = np.array(video_grid_thw)
    else:
        grid_np = video_grid_thw
    
    T, H_tokens, W_tokens = grid_np[0].astype(int)
    # Qwen2.5-VL объединяет 2x2 патча в 1 токен, поэтому делим на 4
    tokens_per_frame = (H_tokens * W_tokens) // 4
    
    # Проверяем соответствие
    if len(frames) != T:
        logger.warning(f"Несоответствие: {len(frames)} кадров vs {T} временных шагов")
        min_len = min(len(frames), T)
        frames = frames[:min_len]
        T = min_len
    
    num_frames = len(frames)
    
    # Создаем коллаж кадров
    cols = min(5, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else [axes]
    axes = axes.flatten()
    
    # Нормализуем важность всех токенов для единой цветовой шкалы
    importance_min = importance_np.min()
    importance_max = importance_np.max()
    importance_range = importance_max - importance_min + 1e-8
    
    for frame_idx in range(num_frames):
        ax = axes[frame_idx]
        
        # Получаем оригинальный кадр
        frame = frames[frame_idx]
        
        # Конвертируем кадр в RGB если нужно
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        frame_rgb = frame[:, :, :3] if frame.shape[2] >= 3 else frame
        frame_h, frame_w = frame_rgb.shape[:2]
        
        # Извлекаем важность токенов для этого кадра
        start_token_idx = frame_idx * tokens_per_frame
        end_token_idx = min(start_token_idx + tokens_per_frame, len(importance_np))
        frame_tokens_importance = importance_np[start_token_idx:end_token_idx]
        
        # Создаем пространственную карту важности [H_tokens, W_tokens]
        # Учитываем, что токены уже объединены (2x2 -> 1 токен)
        H_actual = int(np.sqrt(len(frame_tokens_importance)))
        W_actual = H_actual
        if H_actual * W_actual != len(frame_tokens_importance):
            # Если не квадрат, используем ближайший вариант
            W_actual = len(frame_tokens_importance) // H_actual
        
        token_importance_2d = frame_tokens_importance[:H_actual * W_actual].reshape(H_actual, W_actual)
        
        # Масштабируем до размера кадра
        # Приоритет: cv2 > scipy > numpy fallback
        if CV2_AVAILABLE:
            try:
                import cv2
                spatial_heatmap = cv2.resize(token_importance_2d.astype(np.float32), 
                                            (frame_w, frame_h), 
                                            interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                logger.warning(f"Ошибка cv2.resize: {e}, используем numpy fallback")
                spatial_heatmap = _resize_heatmap_numpy(token_importance_2d, frame_h, frame_w)
        else:
            # Пробуем scipy
            try:
                from scipy.ndimage import zoom
                zoom_h = frame_h / H_tokens
                zoom_w = frame_w / W_tokens
                spatial_heatmap = zoom(token_importance_2d, (zoom_h, zoom_w), order=1)
            except ImportError:
                # Fallback: простое масштабирование через numpy
                spatial_heatmap = _resize_heatmap_numpy(token_importance_2d, frame_h, frame_w)
        
        # Нормализуем для цветовой карты
        spatial_heatmap_normalized = (spatial_heatmap - importance_min) / importance_range
        
        # Применяем цветовую карту 'hot' (красный = высоко, черный = низко)
        import matplotlib.cm as cm
        colormap = cm.get_cmap('hot')
        heatmap_rgb = colormap(spatial_heatmap_normalized)[:, :, :3]  # [H, W, 3]
        heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
        
        # Накладываем на кадр
        if CV2_AVAILABLE:
            try:
                overlay = cv2.addWeighted(frame_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
            except:
                overlay = (frame_rgb * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
        else:
            overlay = (frame_rgb * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
        
        ax.imshow(overlay)
        
        # Вычисляем среднюю важность для этого кадра
        avg_importance = frame_tokens_importance.mean()
        max_importance = frame_tokens_importance.max()
        
        ax.set_title(f'Frame {frame_idx}\nAvg: {avg_importance:.3f} | Max: {max_importance:.3f}', 
                    fontsize=9, fontweight='bold')
        ax.axis('off')
    
    # Скрываем лишние оси
    for idx in range(num_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Пространственная тепловая карта важности видео-токенов на кадрах', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Тепловая карта на кадрах сохранена: {output_path}")
    plt.close()


def _create_simple_frame_heatmap(
    frames: List[np.ndarray],
    frame_importance_list: List[float],
    output_path: str,
    alpha: float = 0.5
):
    """Устаревшая версия с однородной важностью для всего кадра."""
    num_frames = len(frames)
    importance_array = np.array(frame_importance_list)
    importance_normalized = (importance_array - importance_array.min()) / (importance_array.max() - importance_array.min() + 1e-8)
    
    cols = min(5, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else [axes]
    axes = axes.flatten()
    
    for idx, (frame, importance) in enumerate(zip(frames, importance_normalized)):
        ax = axes[idx]
        
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        H, W = frame.shape[:2]
        heatmap = np.ones((H, W)) * importance
        
        import matplotlib.cm as cm
        colormap = cm.get_cmap('hot')
        heatmap_rgb = colormap(heatmap)[:, :, :3]
        heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
        
        frame_rgb = frame[:, :, :3] if frame.shape[2] >= 3 else frame
        if CV2_AVAILABLE:
            try:
                overlay = cv2.addWeighted(frame_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
            except:
                overlay = (frame_rgb * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
        else:
            overlay = (frame_rgb * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
        
        ax.imshow(overlay)
        ax.set_title(f'Frame {idx}\nImportance: {frame_importance_list[idx]:.4f}', 
                    fontsize=9, fontweight='bold')
        ax.axis('off')
    
    for idx in range(num_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Тепловая карта важности видео-токенов на кадрах', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Тепловая карта на кадрах сохранена: {output_path}")
    plt.close()


# ============================================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С МОДЕЛЬЮ (требуют импорта моделей)
# ============================================================================

def load_model_with_checkpoint(
    model_path: str,
    checkpoint_dir: str,
    epoch: Optional[int] = None
) -> Tuple[ViralPredictorModel, Dict]:
    """
    Загружает модель с checkpoint.
    
    Args:
        model_path: Путь к базовой модели (например, "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        checkpoint_dir: Директория с checkpoints
        epoch: Номер эпохи (если None, берется последняя)
    
    Returns:
        Модель и словарь с метаданными
    """
    _lazy_import_models()
    from src.models.viral_model import load_lora_weights, load_viral_head
    
    logger.info(f"Загрузка модели: {model_path}")
    model = ViralPredictorModel(model_path)
    
    # Определяем путь к checkpoint
    if epoch is None:
        # Ищем последнюю эпоху
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("epoch_")]
        if not checkpoints:
            raise ValueError(f"Checkpoints не найдены в {checkpoint_dir}")
        epochs = [int(d.split("_")[1]) for d in checkpoints]
        epoch = max(epochs)
        logger.info(f"Используется последняя эпоха: {epoch}")
    
    epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    lora_path = os.path.join(epoch_dir, "lora_adapter.npz")
    head_path = os.path.join(epoch_dir, "viral_head.npz")
    
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint не найден: {lora_path}")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"Viral head checkpoint не найден: {head_path}")
    
    logger.info(f"Загрузка LoRA из: {lora_path}")
    load_lora_weights(model, lora_path)
    
    logger.info(f"Загрузка viral head из: {head_path}")
    load_viral_head(model, head_path)
    
    metadata = {
        "model_path": model_path,
        "checkpoint_dir": checkpoint_dir,
        "epoch": epoch,
        "lora_path": lora_path,
        "head_path": head_path
    }
    
    return model, metadata


def extract_hidden_states(
    model: ViralPredictorModel,
    input_ids: mx.array,
    attention_mask: mx.array,
    pixel_values_videos: Optional[mx.array] = None,
    video_grid_thw: Optional[mx.array] = None
) -> mx.array:
    """
    Извлекает скрытые состояния из модели (без прохода через viral_head).
    
    Returns:
        Скрытые состояния последнего слоя [batch, seq_len, hidden_size]
    """
    # 1. Получаем эмбеддинги (текст + видео)
    input_features = model.backbone.get_input_embeddings(
        input_ids=input_ids,
        pixel_values=pixel_values_videos,
        video_grid_thw=video_grid_thw
    )
    
    # 2. Рассчитываем position_ids
    position_ids, _ = model.backbone.language_model.get_rope_index(
        input_ids,
        image_grid_thw=None,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask
    )
    
    # 3. Получаем скрытые состояния из языковой модели
    h = model.backbone.language_model.model(
        inputs=None,
        inputs_embeds=input_features.inputs_embeds,
        position_ids=position_ids
    )
    
    return h


def compute_token_importance_gradients(
    model: ViralPredictorModel,
    input_ids: mx.array,
    attention_mask: mx.array,
    pixel_values_videos: Optional[mx.array] = None,
    video_grid_thw: Optional[mx.array] = None,
    labels: Optional[mx.array] = None
) -> mx.array:
    """
    Вычисляет важность токенов через градиенты по входным эмбеддингам.
    
    Returns:
        Важность каждого токена [seq_len]
    """
    def loss_fn(input_embeds):
        # Проходим через модель
        position_ids, _ = model.backbone.language_model.get_rope_index(
            input_ids,
            image_grid_thw=None,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask
        )
        
        h = model.backbone.language_model.model(
            inputs=None,
            inputs_embeds=input_embeds,
            position_ids=position_ids
        )
        
        batch_size = input_ids.shape[0]
        last_token_idx = mx.sum(attention_mask, axis=1).astype(mx.int32) - 1
        seq_len = h.shape[1]
        last_token_idx = mx.minimum(last_token_idx, seq_len - 1)
        h_final = h[mx.arange(batch_size), last_token_idx]
        
        score = model.viral_head(h_final)
        
        if labels is not None:
            loss = mx.mean((score.squeeze() - labels) ** 2)
        else:
            # Используем сам score как цель (максимизируем)
            loss = -mx.mean(score)
        
        return loss
    
    # Получаем входные эмбеддинги
    input_features = model.backbone.get_input_embeddings(
        input_ids=input_ids,
        pixel_values=pixel_values_videos,
        video_grid_thw=video_grid_thw
    )
    input_embeds = input_features.inputs_embeds
    
    # Вычисляем градиенты
    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(input_embeds)
    
    # Вычисляем норму градиентов для каждого токена
    token_importance = mx.sqrt(mx.sum(grads * grads, axis=-1))  # [batch, seq_len]
    token_importance = mx.mean(token_importance, axis=0)  # [seq_len]
    
    mx.eval(token_importance)
    return token_importance


def extract_video_token_importance(
    model: ViralPredictorModel,
    input_ids: mx.array,
    attention_mask: mx.array,
    pixel_values_videos: mx.array,
    video_grid_thw: mx.array,
    processor
) -> Tuple[mx.array, mx.array]:
    """
    Извлекает важность видео-токенов через градиенты.
    
    Returns:
        (video_token_importance, video_grid_thw)
    """
    from src.inference.pipeline import _extract_video_embeddings
    
    # Получаем скрытые состояния
    h = extract_hidden_states(
        model, input_ids, attention_mask, pixel_values_videos, video_grid_thw
    )
    
    # Извлекаем видео эмбеддинги
    video_embeddings = _extract_video_embeddings(h, input_ids, video_grid_thw, processor)
    
    # Вычисляем важность через градиенты по видео эмбеддингам
    def loss_fn(video_emb):
        # Усредняем по токенам кадра
        video_pooled = mx.mean(video_emb, axis=1)  # [batch, T, dim] -> [batch, dim]
        score = model.viral_head(video_pooled)
        return -mx.mean(score)  # Максимизируем score
    
    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(video_embeddings)
    
    # Вычисляем норму градиентов для каждого кадра
    frame_importance = mx.sqrt(mx.sum(grads * grads, axis=-1))  # [batch, T, dim] -> [batch, T]
    frame_importance = mx.mean(frame_importance, axis=0)  # [T]
    
    mx.eval(frame_importance)
    
    # Расширяем важность кадров на все токены кадра
    T, H_tokens, W_tokens = np.array(video_grid_thw)[0].astype(int)
    tokens_per_frame = (H_tokens * W_tokens) // 4
    
    video_token_importance_list = []
    for t in range(T):
        frame_imp = float(frame_importance[t])
        video_token_importance_list.extend([frame_imp] * tokens_per_frame)
    
    video_token_importance = mx.array(video_token_importance_list)
    
    return video_token_importance, video_grid_thw


def create_mock_attention_weights(seq_len: int = 20) -> np.ndarray:
    """
    Создает mock attention weights для тестирования.
    В реальной MLX модели извлечение attention weights сложнее.
    """
    attn = np.random.rand(seq_len, seq_len)
    attn = attn + np.eye(seq_len) * 2
    attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
    return attn


def visualize_qwen_features(
    checkpoint_path: str = "checkpoints",
    data_path: str = "dataset_train_ready.jsonl",
    output_dir: str = "visualizations/qwen_features",
    num_samples: int = 5,
    epoch: Optional[int] = None,
    model_path: str = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    max_frames: int = 6
):
    """
    Основная функция для визуализации признаков обученной модели.
    
    Args:
        checkpoint_path: Директория с checkpoints
        data_path: Путь к JSONL файлу с данными
        output_dir: Директория для сохранения визуализаций
        num_samples: Количество примеров для визуализации
        epoch: Номер эпохи (если None, берется последняя)
        model_path: Путь к базовой модели
        max_frames: Максимальное количество кадров для обработки
    """
    _lazy_import_models()
    
    from src.inference.preprocess import prepare_inputs, load_video_frames, build_prompt_text
    
    logger.info("="*60)
    logger.info("🚀 Запуск визуализации признаков Qwen модели")
    logger.info("="*60)
    
    # Загружаем модель
    model, metadata = load_model_with_checkpoint(model_path, checkpoint_path, epoch)
    logger.info(f"✅ Модель загружена (эпоха {metadata['epoch']})")
    
    # Загружаем датасет
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл данных не найден: {data_path}")
    
    dataset = ViralVideoDataset(data_path, model.processor, max_frames)
    
    # Ограничиваем количество примеров
    num_samples = min(num_samples, len(dataset))
    
    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"📊 Обработка {num_samples} примеров...")
    
    for sample_idx in range(num_samples):
        logger.info(f"\n--- Пример {sample_idx + 1}/{num_samples} ---")
        
        try:
            # Загружаем данные (это возвращает одномерные массивы для одного примера)
            batch = dataset._process_item(sample_idx)
            if batch is None:
                logger.warning(f"Пропуск примера {sample_idx} (не удалось загрузить)")
                continue
            
            # === ИСПРАВЛЕНИЕ: Добавляем измерение батча (N) -> (1, N) ===
            
            # 1. Input IDs
            input_ids = mx.array(batch["input_ids"])
            if input_ids.ndim == 1:
                input_ids = mx.expand_dims(input_ids, axis=0)
            
            # 2. Attention Mask
            attention_mask = mx.array(batch["attention_mask"])
            if attention_mask.ndim == 1:
                attention_mask = mx.expand_dims(attention_mask, axis=0)
            
            # 3. Pixel Values (Videos)
            pixel_values_videos = batch.get("pixel_values_videos")
            if pixel_values_videos is not None:
                pixel_values_videos = mx.array(pixel_values_videos)
                # Обычно pixel_values в Qwen плоские, но на всякий случай проверяем, 
                # если модель ожидает батч (в MLX реализации часто обрабатывается плоско, но оставим как есть)
            
            # 4. Video Grid THW (Критическое исправление для ошибки unpacking)
            video_grid_thw = batch.get("video_grid_thw")
            if video_grid_thw is not None:
                video_grid_thw = mx.array(video_grid_thw)
                # Превращаем [T, H, W] в [[T, H, W]] чтобы цикл for работал корректно
                if video_grid_thw.ndim == 1:
                    video_grid_thw = mx.expand_dims(video_grid_thw, axis=0)

            # ==========================================================

            # Загружаем оригинальные кадры для визуализации
            item = dataset.data[sample_idx]
            video_path = item.get("video_path")
            if video_path and os.path.exists(video_path):
                frames, _, _, _ = load_video_frames(video_path, max_frames)
            else:
                logger.warning(f"Видео не найдено для примера {sample_idx}, пропускаем визуализацию кадров")
                frames = None
            
            # Создаем директорию для этого примера
            sample_dir = os.path.join(output_dir, f"sample_{sample_idx + 1}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 1. Вычисляем важность токенов
            logger.info("Вычисление важности токенов...")
            token_importance = compute_token_importance_gradients(
                model, input_ids, attention_mask, pixel_values_videos, video_grid_thw
            )
            
            # 2. Создаем визуализацию важности токенов
            # input_ids[0] берем 0-й элемент, так как мы добавили батч
            token_importance_path = os.path.join(sample_dir, "token_importance.png")
            create_token_importance_plot(
                token_importance, input_ids[0], model.processor, token_importance_path
            )
            
            # 3. Если есть видео, вычисляем важность видео-токенов
            if pixel_values_videos is not None and video_grid_thw is not None:
                logger.info("Вычисление важности видео-токенов...")
                video_token_importance, video_grid_thw_used = extract_video_token_importance(
                    model, input_ids, attention_mask, pixel_values_videos, video_grid_thw, model.processor
                )
                
                # 4. Создаем визуализацию важности кадров
                frame_importance_list = map_tokens_to_frames(
                    video_token_importance, video_grid_thw_used
                )
                if frame_importance_list:
                    frame_importance = mx.array(frame_importance_list)
                    frame_importance_path = os.path.join(sample_dir, "frame_importance.png")
                    create_frame_importance_plot(frame_importance, frame_importance_path)
                
                # 5. Создаем тепловую карту на кадрах (если есть оригинальные кадры)
                if frames:
                    video_heatmap_path = os.path.join(sample_dir, "video_token_heatmap_on_frames.png")
                    create_video_token_heatmap_on_frames(
                        frames, video_token_importance, video_grid_thw_used, video_heatmap_path
                    )
            
            # 6. Создаем mock attention heatmap
            logger.info("Создание attention heatmap (mock)...")
            seq_len = input_ids.shape[-1]
            mock_attention = mx.array(create_mock_attention_weights(min(seq_len, 50)))
            attention_path = os.path.join(sample_dir, "attention_heatmap.png")
            # input_ids[0] - убираем батч для визуализации
            create_attention_heatmap(
                mock_attention, input_ids[0], model.processor, attention_path, max_tokens=50
            )
            
            logger.info(f"✅ Пример {sample_idx + 1} обработан: {sample_dir}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке примера {sample_idx}: {e}", exc_info=True)
            continue
    
    logger.info("\n" + "="*60)
    logger.info(f"✅ Визуализация завершена! Результаты в: {output_dir}")
    logger.info("="*60)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Визуализация признаков Qwen модели")
    parser.add_argument("--checkpoint_path", default="checkpoints", help="Директория с checkpoints")
    parser.add_argument("--data_path", default="dataset_train_ready.jsonl", help="Путь к JSONL файлу")
    parser.add_argument("--output_dir", default="visualizations/qwen_features", help="Директория для результатов")
    parser.add_argument("--num_samples", type=int, default=5, help="Количество примеров")
    parser.add_argument("--epoch", type=int, default=None, help="Номер эпохи (если None, берется последняя)")
    parser.add_argument("--model_path", default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit", help="Путь к базовой модели")
    parser.add_argument("--max_frames", type=int, default=6, help="Максимальное количество кадров")
    parser.add_argument("--test_mode", action="store_true", help="Тестовый режим без модели")
    
    args = parser.parse_args()
    
    if args.test_mode:
        # Тестовый режим уже обработан в начале файла
        pass
    else:
        visualize_qwen_features(
            checkpoint_path=args.checkpoint_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            epoch=args.epoch,
            model_path=args.model_path,
            max_frames=args.max_frames
        )
