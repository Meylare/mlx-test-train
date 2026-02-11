"""
Тестовый скрипт для проверки визуализаций без загрузки модели.
Создает mock данные и тестирует функции визуализации.
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys
import os

# Добавляем путь к модулям
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Импортируем из корня проекта (файл visualize_qwen_features.py находится в корне)
from visualize_qwen_features import (
    create_attention_heatmap,
    create_token_importance_plot,
    create_frame_importance_plot,
    create_video_token_heatmap_on_frames,
    map_tokens_to_frames,
    create_mock_attention_weights
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = "visualizations/test_output"


def create_mock_processor():
    """Создает mock processor для тестирования."""
    # Создаем простой mock processor
    class MockProcessor:
        class MockTokenizer:
            def convert_ids_to_tokens(self, ids):
                if isinstance(ids, mx.array):
                    ids = np.array(ids)
                elif isinstance(ids, list):
                    ids = np.array(ids)
                return [f"token_{i}" for i in ids]
        
        def __init__(self):
            self.tokenizer = self.MockTokenizer()
    
    return MockProcessor()


def create_mock_input_ids(seq_len=20):
    """Создает mock input_ids."""
    return mx.array(np.random.randint(1000, 2000, (seq_len,), dtype=np.int32))


def create_mock_frames(num_frames=10, height=224, width=224):
    """Создает mock кадры видео."""
    frames = []
    for i in range(num_frames):
        # Создаем разноцветные кадры для визуализации
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Каждый кадр имеет свой цветовой паттерн
        color_intensity = int(255 * (i / num_frames))
        frame[:, :, 0] = color_intensity  # Red channel
        frame[:, :, 1] = 255 - color_intensity  # Green channel
        frame[:, :, 2] = 128  # Blue channel
        frames.append(frame)
    return frames


def test_attention_heatmap():
    """Тестирует создание attention heatmap."""
    logger.info("🧪 Тест: Attention Heatmap")
    
    processor = create_mock_processor()
    attention_weights = mx.array(create_mock_attention_weights(seq_len=15))
    input_ids = create_mock_input_ids(seq_len=15)
    
    output_path = Path(OUTPUT_DIR) / "test_attention_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        create_attention_heatmap(
            attention_weights, input_ids, processor,
            str(output_path), max_tokens=15
        )
        logger.info(f"✅ Тест пройден: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Тест провален: {e}", exc_info=True)
        return False


def test_token_importance():
    """Тестирует создание token importance plot."""
    logger.info("🧪 Тест: Token Importance")
    
    processor = create_mock_processor()
    token_importance = mx.array(np.random.rand(20) * 0.5 + 0.1)  # Важность от 0.1 до 0.6
    input_ids = create_mock_input_ids(seq_len=20)
    
    output_path = Path(OUTPUT_DIR) / "test_token_importance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        create_token_importance_plot(
            token_importance, input_ids, processor,
            str(output_path), top_n=15
        )
        logger.info(f"✅ Тест пройден: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Тест провален: {e}", exc_info=True)
        return False


def test_frame_importance():
    """Тестирует создание frame importance plot."""
    logger.info("🧪 Тест: Frame Importance")
    
    frame_importance = mx.array(np.random.rand(10) * 0.5 + 0.1)
    
    output_path = Path(OUTPUT_DIR) / "test_frame_importance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        create_frame_importance_plot(frame_importance, str(output_path))
        logger.info(f"✅ Тест пройден: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Тест провален: {e}", exc_info=True)
        return False


def test_video_token_heatmap():
    """Тестирует создание video token heatmap на кадрах."""
    logger.info("🧪 Тест: Video Token Heatmap on Frames")
    
    frames = create_mock_frames(num_frames=5)
    
    # Создаем реалистичную пространственную важность токенов
    # Структура: T=5 кадров, H=14, W=14 токенов на кадр
    # Qwen2.5-VL объединяет 2x2 патча в 1 токен, поэтому делим на 4
    T, H_tokens, W_tokens = 5, 14, 14
    tokens_per_frame = (H_tokens * W_tokens) // 4
    num_tokens = T * tokens_per_frame
    
    # Создаем важность токенов с пространственными паттернами
    video_token_importance_list = []
    H_actual = int(np.sqrt(tokens_per_frame))
    W_actual = H_actual
    
    for t in range(T):
        for h in range(H_actual):
            for w in range(W_actual):
                # Создаем паттерн: центр кадра важнее, края менее важны
                center_h, center_w = H_actual // 2, W_actual // 2
                dist_from_center = np.sqrt((h - center_h)**2 + (w - center_w)**2)
                max_dist = np.sqrt(center_h**2 + center_w**2)
                importance = 0.3 + 0.7 * (1 - dist_from_center / (max_dist + 1e-8))
                # Добавляем случайность
                importance += np.random.rand() * 0.2 - 0.1
                importance = max(0.1, min(1.0, importance))
                video_token_importance_list.append(importance)
    
    video_token_importance = mx.array(video_token_importance_list)
    
    # Создаем video_grid_thw
    video_grid_thw = mx.array([[T, H_tokens, W_tokens]], dtype=mx.int32)
    
    output_path = Path(OUTPUT_DIR) / "test_video_token_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        create_video_token_heatmap_on_frames(
            frames, 
            video_token_importance, 
            video_grid_thw,
            str(output_path), 
            alpha=0.4
        )
        logger.info(f"✅ Тест пройден: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Тест провален: {e}", exc_info=True)
        return False


def test_map_tokens_to_frames():
    """Тестирует сопоставление токенов с кадрами."""
    logger.info("🧪 Тест: Map Tokens to Frames")
    
    # Создаем mock video_grid_thw: (T=5, H=14, W=14)
    # Qwen2.5-VL объединяет 2x2 патча в 1 токен, поэтому делим на 4
    video_grid_thw = mx.array([[5, 14, 14]], dtype=mx.int32)
    tokens_per_frame = (14 * 14) // 4
    num_tokens = 5 * tokens_per_frame
    video_token_importance = mx.array(np.random.rand(num_tokens) * 0.5 + 0.1)
    
    try:
        frame_importance_list = map_tokens_to_frames(
            video_token_importance, video_grid_thw, num_frames=5
        )
        
        assert len(frame_importance_list) == 5, f"Ожидалось 5 кадров, получено {len(frame_importance_list)}"
        assert all(0 <= imp <= 1 for imp in frame_importance_list), "Важность должна быть в [0, 1]"
        
        logger.info(f"✅ Тест пройден: {frame_importance_list}")
        return True
    except Exception as e:
        logger.error(f"❌ Тест провален: {e}", exc_info=True)
        return False


def run_all_tests():
    """Запускает все тесты."""
    logger.info("🚀 Запуск тестов визуализации...")
    logger.info(f"Результаты будут сохранены в: {OUTPUT_DIR}\n")
    
    results = {}
    
    results['attention_heatmap'] = test_attention_heatmap()
    results['token_importance'] = test_token_importance()
    results['frame_importance'] = test_frame_importance()
    results['video_token_heatmap'] = test_video_token_heatmap()
    results['map_tokens_to_frames'] = test_map_tokens_to_frames()
    
    # Итоги
    logger.info("\n" + "="*50)
    logger.info("📊 РЕЗУЛЬТАТЫ ТЕСТОВ:")
    logger.info("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("="*50)
    logger.info(f"Пройдено: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 Все тесты пройдены успешно!")
    else:
        logger.warning(f"⚠️ Провалено тестов: {total - passed}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
