import os
import torch
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from run_distillation_0 import run_whitebox_distillation, run_blackbox_distillation

# ============= НАСТРОЙКИ СРАВНЕНИЯ ДИСТИЛЛЯЦИЙ =============

# === МОДЕЛИ ===
# Модель учителя (большая модель)
TEACHER_MODEL = 'gpt2-medium'

# Модель студента (меньшая модель, которую мы обучаем)
STUDENT_MODEL = 'gpt2'

# === ДАННЫЕ ===
# Датасет для обучения
DATASET = 'wikitext'

# Размер выборки из датасета (для ускорения обучения)
SAMPLE_SIZE = 1000

# Максимальная длина последовательности
MAX_LENGTH = 256

# === ПАРАМЕТРЫ ОБУЧЕНИЯ ===
# Количество эпох обучения
EPOCHS = 2

# Размер батча
BATCH_SIZE = 8

# Скорость обучения
LEARNING_RATE = 5e-5

# === СОХРАНЕНИЕ И ЛОГИРОВАНИЕ ===
# Директория для сохранения результатов
OUTPUT_DIR = './output/comparison'

# Создать сравнительную таблицу до и после дистилляции
CREATE_COMPARISON_TABLE = True

# Сохранять лучшую модель по валидационной выборке
SAVE_BEST_MODEL = True

# === ПАРАМЕТРЫ СРАВНЕНИЯ ===
# Запускать white-box дистилляцию
RUN_WHITEBOX = True

# Запускать black-box дистилляцию
RUN_BLACKBOX = True

# Параметры white-box дистилляции
WHITEBOX_CONFIG = {
    'models': {
        'teacher_model_name': TEACHER_MODEL,
        'student_model_name': STUDENT_MODEL,
        'box_type': 'white',
    },
    'dataset': {
        'dataset_name': DATASET,
        'max_length': MAX_LENGTH,
        'max_samples': SAMPLE_SIZE,
    },
    'distillation': {
        'temperature': 2.0,  # Обычно ниже для white-box
        'alpha': 0.7,
        'beta': 0.3,
    },
    'training': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'no_cuda': False,
    },
    'output': {
        'output_dir': f"{OUTPUT_DIR}/whitebox",
        'folder_name': f"whitebox_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'save_metrics': True,
        'save_plot': True,
        'create_comparison_table': CREATE_COMPARISON_TABLE,
        'save_best_model': SAVE_BEST_MODEL,
    },
    'misc': {
        'skip_validation': False,
        'install_deps': True,
        'run_tests': False,
        'skip_demo': True,
    }
}

# Параметры black-box дистилляции
BLACKBOX_CONFIG = {
    'models': {
        'teacher_model_name': TEACHER_MODEL,
        'student_model_name': STUDENT_MODEL,
        'box_type': 'black',
    },
    'dataset': {
        'dataset_name': DATASET,
        'max_length': MAX_LENGTH,
        'max_samples': SAMPLE_SIZE,
    },
    'distillation': {
        'temperature': 3.0,  # Обычно выше для black-box
        'alpha': 0.8,
        'beta': 0.2,
    },
    'training': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'no_cuda': False,
    },
    'output': {
        'output_dir': f"{OUTPUT_DIR}/blackbox",
        'folder_name': f"blackbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'save_metrics': True,
        'save_plot': True,
        'create_comparison_table': CREATE_COMPARISON_TABLE,
        'save_best_model': SAVE_BEST_MODEL,
    },
    'misc': {
        'skip_validation': False,
        'install_deps': True,
        'run_tests': False,
        'skip_demo': True,
    }
}


def compare_distillation_types():
    """Запускает оба типа дистилляции и сравнивает их результаты"""
    results = {}
    
    # Создаем директорию для результатов сравнения
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Запускаем white-box дистилляцию
    if RUN_WHITEBOX:
        print("\n" + "=" * 80)
        print("ЗАПУСК WHITE-BOX ДИСТИЛЛЯЦИИ")
        print("=" * 80 + "\n")
        
        _, whitebox_metrics = run_whitebox_distillation(WHITEBOX_CONFIG)
        results['whitebox'] = whitebox_metrics
    
    # Запускаем black-box дистилляцию
    if RUN_BLACKBOX:
        print("\n" + "=" * 80)
        print("ЗАПУСК BLACK-BOX ДИСТИЛЛЯЦИИ")
        print("=" * 80 + "\n")
        
        _, blackbox_metrics = run_blackbox_distillation(BLACKBOX_CONFIG)
        results['blackbox'] = blackbox_metrics
    
    # Сравниваем результаты
    if RUN_WHITEBOX and RUN_BLACKBOX:
        compare_and_visualize_results(results)
    
    return results


def compare_and_visualize_results(results):
    """Сравнивает и визуализирует результаты разных типов дистилляции"""
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ ДИСТИЛЛЯЦИИ")
    print("=" * 80 + "\n")
    
    # Создаем сравнительную таблицу
    comparison_data = {
        'Метрика': [
            'Коэффициент сжатия',
            'Средняя потеря студента',
            'Средняя потеря дистилляции',
            'Перплексия до дистилляции',
            'Перплексия после дистилляции',
            'Улучшение перплексии (%)',
            'Время выполнения (сек)',
        ]
    }
    
    # Добавляем данные для каждого типа дистилляции
    for dist_type, metrics in results.items():
        column_name = 'White-Box' if dist_type == 'whitebox' else 'Black-Box'
        
        # Извлекаем метрики
        compression_ratio = metrics['config']['compression_ratio']
        avg_student_loss = metrics['metrics']['avg_student_loss']
        avg_distill_loss = metrics['metrics']['avg_distillation_loss']
        initial_perplexity = metrics['metrics']['initial_perplexity']
        final_perplexity = metrics['metrics']['final_perplexity']
        perplexity_improvement = ((initial_perplexity - final_perplexity) / initial_perplexity) * 100
        elapsed_time = metrics['metrics']['elapsed_time']
        
        # Добавляем в таблицу
        comparison_data[column_name] = [
            f"{compression_ratio:.2f}x",
            f"{avg_student_loss:.4f}",
            f"{avg_distill_loss:.4f}",
            f"{initial_perplexity:.2f}",
            f"{final_perplexity:.2f}",
            f"{perplexity_improvement:.2f}%",
            f"{elapsed_time:.2f}",
        ]
    
    # Создаем DataFrame и выводим таблицу
    comparison_df = pd.DataFrame(comparison_data)
    print("\nСравнительная таблица результатов:")
    print(comparison_df.to_string(index=False))
    
    # Сохраняем таблицу в CSV
    comparison_csv_path = os.path.join(OUTPUT_DIR, 'distillation_comparison.csv')
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nТаблица сохранена в: {comparison_csv_path}")
    
    # Создаем графики для сравнения
    create_comparison_plots(results)
    
    # Сохраняем полные результаты в JSON
    results_json_path = os.path.join(OUTPUT_DIR, 'distillation_comparison_results.json')
    with open(results_json_path, 'w') as f:
        # Преобразуем тензоры в списки для сериализации JSON
        serializable_results = {}
        for dist_type, metrics in results.items():
            serializable_results[dist_type] = {}
            for category, values in metrics.items():
                serializable_results[dist_type][category] = {}
                for k, v in values.items():
                    if isinstance(v, torch.Tensor):
                        serializable_results[dist_type][category][k] = v.tolist()
                    else:
                        serializable_results[dist_type][category][k] = v
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"Полные результаты сохранены в: {results_json_path}")


def create_comparison_plots(results):
    """Создает графики для сравнения результатов разных типов дистилляции"""
    # Создаем директорию для графиков
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # График потерь
    plt.figure(figsize=(12, 6))
    for dist_type, metrics in results.items():
        label = 'White-Box' if dist_type == 'whitebox' else 'Black-Box'
        epochs = range(1, len(metrics['metrics']['student_losses']) + 1)
        plt.plot(epochs, metrics['metrics']['student_losses'], 
                 label=f"{label} - Потеря студента", 
                 marker='o' if dist_type == 'whitebox' else 's')
        plt.plot(epochs, metrics['metrics']['distillation_losses'], 
                 label=f"{label} - Потеря дистилляции", 
                 linestyle='--',
                 marker='o' if dist_type == 'whitebox' else 's')
    
    plt.title('Сравнение потерь при разных типах дистилляции')
    plt.xlabel('Эпоха')
    plt.ylabel('Потеря')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Сохраняем график
    losses_plot_path = os.path.join(plots_dir, 'losses_comparison.png')
    plt.savefig(losses_plot_path)
    plt.close()
    
    # График перплексии
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(2)
    
    initial_perplexities = []
    final_perplexities = []
    labels = []
    
    for dist_type, metrics in results.items():
        label = 'White-Box' if dist_type == 'whitebox' else 'Black-Box'
        labels.append(label)
        initial_perplexities.append(metrics['metrics']['initial_perplexity'])
        final_perplexities.append(metrics['metrics']['final_perplexity'])
    
    x = range(len(labels))
    
    plt.bar([i - bar_width/2 for i in x], initial_perplexities, bar_width, label='До дистилляции')
    plt.bar([i + bar_width/2 for i in x], final_perplexities, bar_width, label='После дистилляции')
    
    plt.xlabel('Тип дистилляции')
    plt.ylabel('Перплексия (ниже = лучше)')
    plt.title('Сравнение перплексии до и после дистилляции')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Сохраняем график
    perplexity_plot_path = os.path.join(plots_dir, 'perplexity_comparison.png')
    plt.savefig(perplexity_plot_path)
    plt.close()
    
    # График времени выполнения
    plt.figure(figsize=(8, 5))
    times = [metrics['metrics']['elapsed_time'] for _, metrics in results.items()]
    plt.bar(labels, times, color=['blue', 'orange'])
    plt.xlabel('Тип дистилляции')
    plt.ylabel('Время выполнения (секунды)')
    plt.title('Сравнение времени выполнения')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Сохраняем график
    time_plot_path = os.path.join(plots_dir, 'time_comparison.png')
    plt.savefig(time_plot_path)
    plt.close()
    
    print(f"\nГрафики сравнения сохранены в директории: {plots_dir}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ WHITE-BOX И BLACK-BOX ДИСТИЛЛЯЦИИ")
    print("=" * 80 + "\n")
    
    print("Этот скрипт запускает оба типа дистилляции с одинаковыми моделями и данными,")
    print("но с разными параметрами, оптимизированными для каждого типа.")
    print("После завершения будет создана сравнительная таблица и графики.")
    
    # Запрашиваем подтверждение перед запуском
    confirm = input("\nЗапустить сравнение? Это может занять значительное время. (y/n): ")
    
    if confirm.lower() in ['y', 'yes', 'да']:
        results = compare_distillation_types()
        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ ЗАВЕРШЕНО")
        print("=" * 80 + "\n")
        print(f"Результаты сохранены в директории: {OUTPUT_DIR}")
    else:
        print("Сравнение отменено.")