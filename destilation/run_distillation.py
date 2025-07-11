import os
import torch
from test_distillation_wikitext import distill_gpt2_wikitext

# ============= НАСТРОЙКИ ДИСТИЛЛЯЦИИ =============
# Вы можете изменить эти параметры в соответствии с вашими потребностями

# === МОДЕЛИ ===
# Модель учителя (большая модель)
# Варианты: 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'Qwen/Qwen-1_8B' и т.д.
TEACHER_MODEL = 'gpt2-xl'

# Модель студента (меньшая модель, которую мы обучаем)
# Варианты: 'gpt2', 'gpt2-medium', 'Qwen/Qwen-1_8B-Chat' и т.д.
STUDENT_MODEL = 'gpt2'

# === ДАННЫЕ ===
# Датасет для обучения
# Варианты: 'wikitext' (стандартный), 'pokemon' (пример пользовательского), 
# или путь к вашему собственному файлу
DATASET = 'wikitext'

# Путь к пользовательскому датасету (если используется)
# Оставьте None, если используете стандартный датасет
DATASET_PATH = None

# Размер выборки из датасета (для ускорения обучения)
# Установите 0 для использования всего датасета
SAMPLE_SIZE = 0

# Максимальная длина последовательности
MAX_LENGTH = 256

# === ПАРАМЕТРЫ ДИСТИЛЛЯЦИИ ===
# Температура для дистилляции (влияет на "мягкость" распределения вероятностей)
# Более высокие значения делают распределение более плоским
TEMPERATURE = 2.0

# Вес для soft targets (KL-дивергенция между распределениями учителя и студента)
# Чем выше значение, тем больше студент имитирует распределение учителя
ALPHA = 0.7

# Вес для hard targets (обычная функция потерь на истинных метках)
# Чем выше значение, тем больше студент фокусируется на правильных ответах
BETA = 0.3

# === ПАРАМЕТРЫ ОБУЧЕНИЯ ===
# Количество эпох обучения
EPOCHS = 150

# Размер батча
BATCH_SIZE = 16

# Скорость обучения
LEARNING_RATE = 5e-5

# Не использовать CUDA (GPU)
# Установите True, если хотите использовать только CPU
NO_CUDA = False

# === СОХРАНЕНИЕ И ЛОГИРОВАНИЕ ===
# Директория для сохранения результатов
OUTPUT_DIR = './output'

# Название папки для сохранения результатов дистилляции
# Если None, будет создана папка с временной меткой
FOLDER_NAME = None

# Интервал сохранения чекпоинтов (в эпохах)
CHECKPOINT_INTERVAL = 50

# Сохранить метрики в JSON
SAVE_METRICS = True

# Сохранить график потерь
SAVE_PLOT = True

# Создать сравнительную таблицу до и после дистилляции
CREATE_COMPARISON_TABLE = False

# Сохранять лучшую модель по валидационной выборке
SAVE_BEST_MODEL = True

# Пропустить валидацию улучшения студента
SKIP_VALIDATION = False

# === ДОПОЛНИТЕЛЬНЫЕ ОПЦИИ ===
# Установить зависимости
INSTALL_DEPS = True

# Запустить тесты
RUN_TESTS = False

# Пропустить демонстрацию
SKIP_DEMO = True


def run_distillation():
    """Запускает процесс дистилляции с указанными параметрами"""
    # Запускаем дистилляцию
    print("\n=== ЗАПУСК ДИСТИЛЛЯЦИИ С НАСТРОЕННЫМИ ПАРАМЕТРАМИ ===\n")
    
    # Создаем объект args, имитирующий аргументы командной строки
    class Args:
        def __init__(self):
            self.teacher_model = TEACHER_MODEL
            self.student_model = STUDENT_MODEL
            self.dataset = DATASET
            self.dataset_path = DATASET_PATH
            self.temperature = TEMPERATURE
            self.alpha = ALPHA
            self.beta = BETA
            self.epochs = EPOCHS
            self.batch_size = BATCH_SIZE
            self.lr = LEARNING_RATE
            self.max_length = MAX_LENGTH
            self.sample_size = SAMPLE_SIZE
            self.save_metrics = SAVE_METRICS
            self.save_plot = SAVE_PLOT
            self.no_cuda = NO_CUDA
            self.output_dir = OUTPUT_DIR
            self.checkpoint_interval = CHECKPOINT_INTERVAL
            self.run_tests = RUN_TESTS
            self.skip_demo = SKIP_DEMO
            self.install_deps = INSTALL_DEPS
            self.skip_validation = SKIP_VALIDATION
            self.create_comparison_table = CREATE_COMPARISON_TABLE
            self.save_best_model = SAVE_BEST_MODEL
            self.folder_name = FOLDER_NAME

    args = Args()

    trained_student, metrics = distill_gpt2_wikitext(args)
    
    # Выводим итоговую информацию
    print("\n=== ДИСТИЛЛЯЦИЯ ЗАВЕРШЕНА ===\n")
    print(f"Коэффициент сжатия: {metrics['config']['compression_ratio']:.2f}x")
    print(f"Средняя потеря студента: {metrics['metrics']['avg_student_loss']:.4f}")
    print(f"Время выполнения: {metrics['metrics']['elapsed_time']:.2f} секунд")
    
    # Возвращаем обученную модель и метрики
    return trained_student, metrics


if __name__ == "__main__":
    run_distillation()