#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для автоматизации процесса дистилляции моделей с различными параметрами.
Позволяет запускать серию экспериментов с разными конфигурациями и сравнивать результаты.
"""

import os
import json
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Импортируем функции для запуска дистилляции
from run_whitebox_distillation import run_whitebox_distillation
from run_blackbox_distillation import run_blackbox_distillation


def create_experiment_config(base_config, experiment_params):
    """
    Создает конфигурацию для эксперимента на основе базовой конфигурации
    и параметров эксперимента.
    
    Args:
        base_config (dict): Базовая конфигурация
        experiment_params (dict): Параметры эксперимента для замены в базовой конфигурации
        
    Returns:
        dict: Конфигурация для эксперимента
    """
    # Создаем глубокую копию базовой конфигурации
    config = json.loads(json.dumps(base_config))
    
    # Обновляем конфигурацию параметрами эксперимента
    for section, params in experiment_params.items():
        if section in config:
            config[section].update(params)
        else:
            config[section] = params
            
    return config


def run_experiment(config, experiment_name):
    """
    Запускает эксперимент с заданной конфигурацией.
    
    Args:
        config (dict): Конфигурация эксперимента
        experiment_name (str): Название эксперимента
        
    Returns:
        tuple: (trained_student, metrics) - обученная модель и метрики
    """
    print(f"\n{'='*50}")
    print(f"Запуск эксперимента: {experiment_name}")
    print(f"{'='*50}\n")
    
    # Обновляем имя папки для сохранения результатов
    config['output']['folder_name'] = experiment_name
    
    # Запускаем дистилляцию в зависимости от типа
    box_type = config['models']['box_type']
    if box_type == 'white':
        return run_whitebox_distillation(config)
    elif box_type == 'black':
        return run_blackbox_distillation(config)
    else:
        raise ValueError(f"Неизвестный тип дистилляции: {box_type}")


def save_experiment_results(results, output_dir):
    """
    Сохраняет результаты экспериментов в CSV файл и создает визуализации.
    
    Args:
        results (list): Список результатов экспериментов
        output_dir (str): Директория для сохранения результатов
    """
    # Создаем директорию для результатов, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    
    # Сохраняем результаты в CSV
    csv_path = os.path.join(output_dir, 'experiment_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nРезультаты сохранены в {csv_path}")
    
    # Создаем визуализации
    create_visualizations(results_df, output_dir)


def create_visualizations(results_df, output_dir):
    """
    Создает визуализации результатов экспериментов.
    
    Args:
        results_df (DataFrame): DataFrame с результатами экспериментов
        output_dir (str): Директория для сохранения визуализаций
    """
    # Создаем директорию для визуализаций
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Сравнение перплексии до и после дистилляции
    plt.figure(figsize=(12, 6))
    x = range(len(results_df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], results_df['initial_perplexity'], width, label='До дистилляции')
    plt.bar([i + width/2 for i in x], results_df['final_perplexity'], width, label='После дистилляции')
    
    plt.xlabel('Эксперимент')
    plt.ylabel('Перплексия (ниже = лучше)')
    plt.title('Сравнение перплексии до и после дистилляции')
    plt.xticks(x, results_df['experiment_name'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'perplexity_comparison.png'))
    plt.close()
    
    # 2. Сравнение времени выполнения
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['experiment_name'], results_df['elapsed_time'], color='skyblue')
    plt.xlabel('Эксперимент')
    plt.ylabel('Время выполнения (секунды)')
    plt.title('Время выполнения экспериментов')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'execution_time.png'))
    plt.close()
    
    # 3. Улучшение перплексии (в процентах)
    results_df['perplexity_improvement'] = ((results_df['initial_perplexity'] - results_df['final_perplexity']) / 
                                           results_df['initial_perplexity'] * 100)
    
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['experiment_name'], results_df['perplexity_improvement'], color='lightgreen')
    plt.xlabel('Эксперимент')
    plt.ylabel('Улучшение перплексии (%)')
    plt.title('Процентное улучшение перплексии')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'perplexity_improvement.png'))
    plt.close()
    
    # 4. Сравнение потерь
    if 'avg_student_loss' in results_df.columns and 'avg_distillation_loss' in results_df.columns:
        plt.figure(figsize=(12, 6))
        x = range(len(results_df))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], results_df['avg_student_loss'], width, label='Потеря студента')
        plt.bar([i + width/2 for i in x], results_df['avg_distillation_loss'], width, label='Потеря дистилляции')
        
        plt.xlabel('Эксперимент')
        plt.ylabel('Средняя потеря')
        plt.title('Сравнение потерь')
        plt.xticks(x, results_df['experiment_name'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'loss_comparison.png'))
        plt.close()
    
    print(f"Визуализации сохранены в {viz_dir}")


def main():
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description='Автоматизация экспериментов с дистилляцией')
    parser.add_argument('--output_dir', type=str, default='./output/experiments',
                        help='Директория для сохранения результатов экспериментов')
    args = parser.parse_args()
    
    # Создаем уникальную директорию для текущей серии экспериментов
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'experiment_series_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Базовая конфигурация для экспериментов
    base_config = {
        'models': {
            'teacher_model_name': 'gpt2-medium',
            'student_model_name': 'gpt2',
            'teacher_model_path': None,
            'student_model_path': None,
            'box_type': 'white',  # По умолчанию white-box
        },
        'dataset': {
            'dataset_name': 'wikitext',
            'dataset_path': None,
            'max_length': 256,
            'max_samples': 1000,  # Ограничиваем размер датасета для ускорения экспериментов
        },
        'distillation': {
            'temperature': 2.0,
            'alpha': 0.5,
            'beta': 0.5,
        },
        'training': {
            'epochs': 3,
            'batch_size': 8,
            'learning_rate': 5e-5,
            'no_cuda': False,
        },
        'output': {
            'output_dir': output_dir,
            'folder_name': 'default',  # Будет переопределено для каждого эксперимента
            'checkpoint_interval': 1,
            'save_metrics': True,
            'save_plot': True,
            'create_comparison_table': True,
            'save_best_model': True,
        },
        'misc': {
            'skip_validation': False,
            'install_deps': True,
            'run_tests': False,
            'skip_demo': True,
        }
    }
    
    # Определяем эксперименты
    experiments = [
        # 1. Базовый эксперимент с White-Box дистилляцией
        {
            'name': 'whitebox_baseline',
            'params': {
                'models': {'box_type': 'white'},
                'distillation': {'temperature': 2.0, 'alpha': 0.5, 'beta': 0.5}
            }
        },
        # 2. White-Box с высокой температурой
        {
            'name': 'whitebox_high_temp',
            'params': {
                'models': {'box_type': 'white'},
                'distillation': {'temperature': 4.0, 'alpha': 0.5, 'beta': 0.5}
            }
        },
        # 3. White-Box с акцентом на дистилляцию
        {
            'name': 'whitebox_high_alpha',
            'params': {
                'models': {'box_type': 'white'},
                'distillation': {'temperature': 2.0, 'alpha': 0.8, 'beta': 0.2}
            }
        },
        # 4. Базовый эксперимент с Black-Box дистилляцией
        {
            'name': 'blackbox_baseline',
            'params': {
                'models': {'box_type': 'black'},
                'distillation': {'temperature': 3.0, 'alpha': 0.7, 'beta': 0.3}
            }
        },
        # 5. Black-Box с высокой температурой
        {
            'name': 'blackbox_high_temp',
            'params': {
                'models': {'box_type': 'black'},
                'distillation': {'temperature': 5.0, 'alpha': 0.7, 'beta': 0.3}
            }
        },
        # 6. Black-Box с акцентом на дистилляцию
        {
            'name': 'blackbox_high_alpha',
            'params': {
                'models': {'box_type': 'black'},
                'distillation': {'temperature': 3.0, 'alpha': 0.9, 'beta': 0.1}
            }
        },
    ]
    
    # Запускаем эксперименты и собираем результаты
    results = []
    
    for experiment in experiments:
        # Создаем конфигурацию для эксперимента
        config = create_experiment_config(base_config, experiment['params'])
        
        # Запускаем эксперимент
        start_time = time.time()
        _, metrics = run_experiment(config, experiment['name'])
        elapsed_time = time.time() - start_time
        
        # Собираем результаты
        result = {
            'experiment_name': experiment['name'],
            'box_type': config['models']['box_type'],
            'temperature': config['distillation']['temperature'],
            'alpha': config['distillation']['alpha'],
            'beta': config['distillation']['beta'],
            'initial_perplexity': metrics['metrics']['initial_perplexity'],
            'final_perplexity': metrics['metrics']['final_perplexity'],
            'avg_student_loss': metrics['metrics']['avg_student_loss'],
            'avg_distillation_loss': metrics['metrics']['avg_distillation_loss'],
            'compression_ratio': metrics['config']['compression_ratio'],
            'elapsed_time': metrics['metrics']['elapsed_time'],
        }
        
        results.append(result)
        
        # Сохраняем промежуточные результаты
        with open(os.path.join(output_dir, f"{experiment['name']}_results.json"), 'w') as f:
            json.dump(result, f, indent=2)
    
    # Сохраняем и визуализируем результаты
    save_experiment_results(results, output_dir)
    
    # Выводим сводную таблицу результатов
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 80)
    
    # Определяем лучший эксперимент по перплексии
    best_experiment = results_df.loc[results_df['final_perplexity'].idxmin()]
    print(f"\nЛучший эксперимент: {best_experiment['experiment_name']}")
    print(f"Тип дистилляции: {best_experiment['box_type']}-box")
    print(f"Температура: {best_experiment['temperature']}")
    print(f"Альфа: {best_experiment['alpha']}")
    print(f"Бета: {best_experiment['beta']}")
    print(f"Финальная перплексия: {best_experiment['final_perplexity']:.2f}")
    print(f"Улучшение перплексии: {((best_experiment['initial_perplexity'] - best_experiment['final_perplexity']) / best_experiment['initial_perplexity'] * 100):.2f}%")
    
    print(f"\nВсе результаты и визуализации сохранены в {output_dir}")


if __name__ == "__main__":
    main()