import os
import torch
from test_distillation_wikitext import distill_gpt2_wikitext
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

# ============= НАСТРОЙКИ ДИСТИЛЛЯЦИИ =============
# Вы можете изменить эти параметры в соответствии с вашими потребностями

# === МОДЕЛИ ===
# Модель учителя (большая модель)
# Варианты: 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'Qwen/Qwen-1_8B' и т.д.
TEACHER_MODEL = 'gpt2-xl'

# Модель студента (меньшая модель, которую мы обучаем)
# Для не предобученной модели установим значение 'random-gpt2'
STUDENT_MODEL = 'random-gpt2'

# Путь к локальной модели учителя (если None, будет использована модель из Hugging Face)
TEACHER_MODEL_PATH = None

# Путь к локальной модели студента (если None, будет использована модель из Hugging Face)
STUDENT_MODEL_PATH = None

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
# Увеличиваем размер выборки для лучшего обучения
SAMPLE_SIZE = 5000

# Максимальная длина последовательности
MAX_LENGTH = 256

# === ПАРАМЕТРЫ ДИСТИЛЛЯЦИИ ===
# Температура для дистилляции (влияет на "мягкость" распределения вероятностей)
# Снижаем температуру для уменьшения риска переобучения
TEMPERATURE = 1.5

# Вес для soft targets (KL-дивергенция между распределениями учителя и студента)
# Уменьшаем alpha для снижения влияния учителя
ALPHA = 0.5

# Вес для hard targets (обычная функция потерь на истинных метках)
# Увеличиваем beta для лучшего обучения на реальных данных
BETA = 0.5

# === ПАРАМЕТРЫ ОБУЧЕНИЯ ===
# Количество эпох обучения
# Увеличиваем количество эпох для лучшего обучения
EPOCHS = 5

# Размер батча
BATCH_SIZE = 8

# Скорость обучения
# Уменьшаем скорость обучения для более стабильного обучения
LEARNING_RATE = 1e-5

# Не использовать CUDA (GPU)
# Установите True, если хотите использовать только CPU
NO_CUDA = False

# === РЕГУЛЯРИЗАЦИЯ ===
# Вес L2-регуляризации (weight decay)
WEIGHT_DECAY = 0.01

# Вероятность dropout для слоев модели
DROPOUT_PROB = 0.1

# === СОХРАНЕНИЕ И ЛОГИРОВАНИЕ ===
# Директория для сохранения результатов
OUTPUT_DIR = './output'

# Название папки для сохранения результатов дистилляции
# Если None, будет создана папка с временной меткой
FOLDER_NAME = None

# Интервал сохранения чекпоинтов (в эпохах)
CHECKPOINT_INTERVAL = 1

# Сохранить метрики в JSON
SAVE_METRICS = True

# Сохранить график потерь
SAVE_PLOT = True

# Создать сравнительную таблицу до и после дистилляции
CREATE_COMPARISON_TABLE = True

# Сохранять лучшую модель по валидационной выборке
# Включаем для реализации early stopping
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

# Конфигурация дистилляции в формате словаря для использования в Jupyter ноутбуке
DISTILLATION_CONFIG = {
    'models': {
        'teacher_model_name': TEACHER_MODEL,
        'student_model_name': STUDENT_MODEL,
        'teacher_model_path': TEACHER_MODEL_PATH,
        'student_model_path': STUDENT_MODEL_PATH,
    },
    'dataset': {
        'dataset_name': DATASET,
        'dataset_path': DATASET_PATH,
        'max_length': MAX_LENGTH,
        'max_samples': SAMPLE_SIZE,
    },
    'distillation': {
        'temperature': TEMPERATURE,
        'alpha': ALPHA,
        'beta': BETA,
    },
    'training': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'no_cuda': NO_CUDA,
        'weight_decay': WEIGHT_DECAY,
        'dropout_prob': DROPOUT_PROB,
    },
    'output': {
        'output_dir': OUTPUT_DIR,
        'folder_name': FOLDER_NAME,
        'checkpoint_interval': CHECKPOINT_INTERVAL,
        'save_metrics': SAVE_METRICS,
        'save_plot': SAVE_PLOT,
        'create_comparison_table': CREATE_COMPARISON_TABLE,
        'save_best_model': SAVE_BEST_MODEL,
    },
    'misc': {
        'skip_validation': SKIP_VALIDATION,
        'install_deps': INSTALL_DEPS,
        'run_tests': RUN_TESTS,
        'skip_demo': SKIP_DEMO,
    }
}


def create_random_gpt2_model(config_name='gpt2', dropout_prob=0.1):
    """
    Создает модель GPT-2 с случайной инициализацией весов (не предобученную)
    
    Args:
        config_name: Имя конфигурации модели ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        dropout_prob: Вероятность dropout для слоев модели
    
    Returns:
        Модель GPT-2 со случайно инициализированными весами
    """
    # Загружаем конфигурацию модели
    if config_name == 'gpt2':
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            dropout=dropout_prob
        )
    elif config_name == 'gpt2-medium':
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            dropout=dropout_prob
        )
    elif config_name == 'gpt2-large':
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=1280,
            n_layer=36,
            n_head=20,
            dropout=dropout_prob
        )
    elif config_name == 'gpt2-xl':
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=1600,
            n_layer=48,
            n_head=25,
            dropout=dropout_prob
        )
    else:
        # По умолчанию используем конфигурацию gpt2
        config = GPT2Config(dropout=dropout_prob)
    
    # Создаем модель с случайной инициализацией весов
    model = GPT2LMHeadModel(config)
    
    # Инициализируем веса модели
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    model.apply(_init_weights)
    
    return model


def run_distillation(config=None):
    """
    Запускает процесс дистилляции с указанными параметрами
    
    Args:
        config: Словарь с конфигурацией дистилляции. Если None, используются глобальные параметры.
    """
    # Запускаем дистилляцию
    print("\n=== ЗАПУСК ДИСТИЛЛЯЦИИ С НАСТРОЕННЫМИ ПАРАМЕТРАМИ ===\n")
    
    # Используем переданную конфигурацию или глобальные параметры
    if config is None:
        # Используем глобальные параметры
        teacher_model = TEACHER_MODEL
        student_model = STUDENT_MODEL
        teacher_model_path = TEACHER_MODEL_PATH
        student_model_path = STUDENT_MODEL_PATH
        dataset = DATASET
        dataset_path = DATASET_PATH
        temperature = TEMPERATURE
        alpha = ALPHA
        beta = BETA
        epochs = EPOCHS
        batch_size = BATCH_SIZE
        learning_rate = LEARNING_RATE
        max_length = MAX_LENGTH
        sample_size = SAMPLE_SIZE
        save_metrics = SAVE_METRICS
        save_plot = SAVE_PLOT
        no_cuda = NO_CUDA
        output_dir = OUTPUT_DIR
        checkpoint_interval = CHECKPOINT_INTERVAL
        run_tests = RUN_TESTS
        skip_demo = SKIP_DEMO
        install_deps = INSTALL_DEPS
        skip_validation = SKIP_VALIDATION
        create_comparison_table = CREATE_COMPARISON_TABLE
        save_best_model = SAVE_BEST_MODEL
        folder_name = FOLDER_NAME
        weight_decay = WEIGHT_DECAY
        dropout_prob = DROPOUT_PROB
    else:
        # Используем переданную конфигурацию
        teacher_model = config['models']['teacher_model_name']
        student_model = config['models']['student_model_name']
        teacher_model_path = config['models'].get('teacher_model_path', None)
        student_model_path = config['models'].get('student_model_path', None)
        dataset = config['dataset']['dataset_name']
        dataset_path = config['dataset']['dataset_path']
        temperature = config['distillation']['temperature']
        alpha = config['distillation']['alpha']
        beta = config['distillation']['beta']
        epochs = config['training']['epochs']
        batch_size = config['training']['batch_size']
        learning_rate = config['training']['learning_rate']
        max_length = config['dataset']['max_length']
        sample_size = config['dataset']['max_samples']
        save_metrics = config['output']['save_metrics']
        save_plot = config['output']['save_plot']
        no_cuda = config['training']['no_cuda']
        output_dir = config['output']['output_dir']
        checkpoint_interval = config['output']['checkpoint_interval']
        run_tests = config['misc']['run_tests']
        skip_demo = config['misc']['skip_demo']
        install_deps = config['misc']['install_deps']
        skip_validation = config['misc']['skip_validation']
        create_comparison_table = config['output']['create_comparison_table']
        save_best_model = config['output']['save_best_model']
        folder_name = config['output']['folder_name']
        weight_decay = config['training']['weight_decay']
        dropout_prob = config['training']['dropout_prob']
    
    # Создаем объект args, имитирующий аргументы командной строки
    class Args:
        def __init__(self):
            self.teacher_model = teacher_model
            self.student_model = student_model
            self.teacher_model_path = teacher_model_path
            self.student_model_path = student_model_path
            self.dataset = dataset
            self.dataset_path = dataset_path
            self.temperature = temperature
            self.alpha = alpha
            self.beta = beta
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = learning_rate
            self.max_length = max_length
            self.sample_size = sample_size
            self.save_metrics = save_metrics
            self.save_plot = save_plot
            self.no_cuda = no_cuda
            self.output_dir = output_dir
            self.checkpoint_interval = checkpoint_interval
            self.run_tests = run_tests
            self.skip_demo = skip_demo
            self.install_deps = install_deps
            self.skip_validation = skip_validation
            self.create_comparison_table = create_comparison_table
            self.save_best_model = save_best_model
            self.folder_name = folder_name
            self.weight_decay = weight_decay
            self.dropout_prob = dropout_prob
            self.random_init = student_model.startswith('random-')
            self.random_model_size = student_model.replace('random-', '') if student_model.startswith('random-') else 'gpt2'

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