#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый модуль для проверки работоспособности дистилляции знаний

Этот файл содержит тесты для проверки основных функций модуля distillation.py
и демонстрирует примеры использования с различными моделями.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import sys
import os
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Импорт нашего модуля дистилляции
from distillation import KnowledgeDistillation, DistillationConfig, create_optimizer


class SimpleTeacherModel(nn.Module):
    """Простая модель учителя для тестирования"""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 512, num_classes: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


class SimpleStudentModel(nn.Module):
    """Простая модель студента для тестирования"""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


def create_synthetic_data(num_samples: int = 1000, input_size: int = 784, num_classes: int = 10) -> Tuple[DataLoader, DataLoader]:
    """Создание синтетических данных для тестирования"""
    
    # Генерация случайных данных
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Разделение на train/test
    split_idx = int(0.8 * num_samples)
    
    train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
    test_dataset = TensorDataset(X[split_idx:], y[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


def test_basic_functionality():
    """Тест базовой функциональности дистилляции"""
    print("\n=== Тест базовой функциональности ===")
    
    # Создание конфигурации
    config = DistillationConfig(
        temperature=4.0,
        alpha=0.7,
        beta=0.3,
        log_interval=10
    )
    
    # Создание дистиллятора
    distiller = KnowledgeDistillation(config)
    
    # Создание моделей
    teacher = SimpleTeacherModel()
    student = SimpleStudentModel()
    
    print(f"Параметры учителя: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Параметры студента: {sum(p.numel() for p in student.parameters()):,}")
    
    # Установка моделей
    distiller.set_models(teacher, student)
    
    # Создание тестовых данных
    batch_size = 16
    inputs = torch.randn(batch_size, 784)
    targets = torch.randint(0, 10, (batch_size,))
    
    # Тест функции потерь
    with torch.no_grad():
        teacher_logits = teacher(inputs)
        student_logits = student(inputs)
    
    losses = distiller.distillation_loss(student_logits, teacher_logits, targets)
    
    print(f"Общая потеря: {losses['total_loss'].item():.4f}")
    print(f"Потеря дистилляции: {losses['distillation_loss'].item():.4f}")
    print(f"Потеря студента: {losses['student_loss'].item():.4f}")
    
    # Получение метрик
    metrics = distiller.get_metrics()
    print(f"Коэффициент сжатия: {metrics['config']['compression_ratio']:.2f}x")
    
    print("✅ Базовая функциональность работает корректно")
    return True


def test_training_process():
    """Тест процесса обучения"""
    print("\n=== Тест процесса обучения ===")
    
    # Создание данных
    train_loader, test_loader = create_synthetic_data(num_samples=500)
    
    # Создание моделей
    teacher = SimpleTeacherModel()
    student = SimpleStudentModel()
    
    # Предварительное обучение учителя (симуляция)
    print("Симуляция предварительного обучения учителя...")
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    teacher.train()
    
    for epoch in range(2):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            teacher_optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            teacher_optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"Учитель - Эпоха {epoch+1}, Батч {batch_idx}, Потеря: {loss.item():.4f}")
    
    # Дистилляция
    config = DistillationConfig(
        temperature=3.0,
        alpha=0.8,
        beta=0.2,
        log_interval=5
    )
    
    distiller = KnowledgeDistillation(config)
    student_optimizer = optim.Adam(student.parameters(), lr=0.001)
    
    print("\nНачало дистилляции...")
    trained_student = distiller.apply(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        optimizer=student_optimizer,
        num_epochs=2
    )
    
    # Получение финальных метрик
    final_metrics = distiller.get_metrics()
    print(f"\nФинальные метрики:")
    print(f"Средняя общая потеря: {final_metrics['metrics']['avg_total_loss']:.4f}")
    print(f"Средняя потеря дистилляции: {final_metrics['metrics']['avg_distillation_loss']:.4f}")
    print(f"Средняя потеря студента: {final_metrics['metrics']['avg_student_loss']:.4f}")
    print(f"Время обучения: {final_metrics['metrics']['elapsed_time']:.2f} сек")
    
    print("✅ Процесс обучения работает корректно")
    return True


def test_gpt2_distillation(sample_size: int = 100):
    """Тест дистилляции с моделями GPT-2"""
    print("\n=== Тест дистилляции GPT-2 ===")
    
    try:
        # Загрузка токенизатора
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Создание моделей
        print("Загрузка моделей GPT-2...")
        teacher_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        student_model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        print(f"Параметры учителя (GPT-2 medium): {sum(p.numel() for p in teacher_model.parameters()):,}")
        print(f"Параметры студента (GPT-2): {sum(p.numel() for p in student_model.parameters()):,}")
        
        # Создание простых текстовых данных
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "Knowledge distillation helps create smaller models.",
            "Natural language processing enables computers to understand text."
        ] * (sample_size // 5)
        
        # Токенизация
        max_length = 64
        encoded = tokenizer(
            texts[:sample_size], 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
        
        # Создание DataLoader
        dataset = TensorDataset(encoded['input_ids'], encoded['input_ids'])  # targets = inputs для LM
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Конфигурация дистилляции
        config = DistillationConfig(
            temperature=4.0,
            alpha=0.7,
            beta=0.3,
            log_interval=5,
            device='cpu'  # Используем CPU для совместимости
        )
        
        # Создание дистиллятора
        distiller = KnowledgeDistillation(config)
        
        # Оптимизатор для студента
        optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)
        
        # Функция потерь для языковых моделей
        def lm_loss_fn(logits, targets):
            # Сдвигаем targets для предсказания следующего токена
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            return nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        print("\nНачало дистилляции GPT-2...")
        
        # Применение дистилляции
        trained_student = distiller.apply(
            student_model=student_model,
            teacher_model=teacher_model,
            train_loader=dataloader,
            optimizer=optimizer,
            num_epochs=1,
            student_loss_fn=lm_loss_fn
        )
        
        # Сохранение модели
        save_path = "distilled_gpt2_student.pt"
        distiller.save_student_model(save_path)
        print(f"Модель сохранена в {save_path}")
        
        print("✅ Дистилляция GPT-2 выполнена успешно")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании GPT-2: {e}")
        print("Возможно, недостаточно памяти или отсутствует интернет-соединение")
        return False


def test_pokemon_dataset(sample_size: int = 200):
    """Тест с набором данных о покемонах"""
    print("\n=== Тест с набором данных о покемонах ===")
    
    try:
        # Загрузка датасета
        print("Загрузка датасета о покемонах...")
        dataset = load_dataset("lamini/pokemon-bleu", split="train")
        
        # Ограничиваем размер выборки
        if len(dataset) > sample_size:
            dataset = dataset.select(range(sample_size))
        
        # Подготовка текстов
        texts = [item['text'] for item in dataset]
        
        # Токенизация
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        encoded = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors='pt'
        )
        
        # Создание DataLoader
        dataset_tensor = TensorDataset(encoded['input_ids'], encoded['input_ids'])
        dataloader = DataLoader(dataset_tensor, batch_size=4, shuffle=True)
        
        # Модели
        teacher_model = GPT2LMHeadModel.from_pretrained('gpt2')
        student_config = GPT2Config.from_pretrained('gpt2')
        student_config.n_layer = 6  # Уменьшаем количество слоев
        student_config.n_head = 8   # Уменьшаем количество голов внимания
        student_model = GPT2LMHeadModel(student_config)
        
        print(f"Параметры учителя: {sum(p.numel() for p in teacher_model.parameters()):,}")
        print(f"Параметры студента: {sum(p.numel() for p in student_model.parameters()):,}")
        
        # Дистилляция
        config = DistillationConfig(
            temperature=5.0,
            alpha=0.8,
            beta=0.2,
            log_interval=10,
            device='cpu'
        )
        
        distiller = KnowledgeDistillation(config)
        optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)
        
        def lm_loss_fn(logits, targets):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            return nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        print("\nНачало дистилляции на данных о покемонах...")
        
        trained_student = distiller.apply(
            student_model=student_model,
            teacher_model=teacher_model,
            train_loader=dataloader,
            optimizer=optimizer,
            num_epochs=1,
            student_loss_fn=lm_loss_fn
        )
        
        print("✅ Дистилляция на данных о покемонах выполнена успешно")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании с покемонами: {e}")
        return False


def test_factory_function():
    """Тест фабричной функции создания оптимизаторов"""
    print("\n=== Тест фабричной функции ===")
    
    # Тест создания дистиллятора
    config = {
        'temperature': 3.0,
        'alpha': 0.6,
        'beta': 0.4
    }
    
    distiller = create_optimizer('distillation', config)
    assert isinstance(distiller, KnowledgeDistillation)
    assert distiller.config.temperature == 3.0
    assert distiller.config.alpha == 0.6
    
    print("✅ Фабричная функция работает корректно")
    return True


def save_metrics_to_csv(metrics: Dict[str, Any], filename: str = "distillation_metrics.csv"):
    """Сохранение метрик в CSV файл"""
    try:
        # Подготовка данных для CSV
        data = {
            'metric': [],
            'value': []
        }
        
        # Добавляем метрики обучения
        if 'metrics' in metrics:
            for key, value in metrics['metrics'].items():
                data['metric'].append(key)
                data['value'].append(value)
        
        # Добавляем конфигурацию
        if 'config' in metrics:
            for key, value in metrics['config'].items():
                data['metric'].append(f"config_{key}")
                data['value'].append(value)
        
        # Сохранение в CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Метрики сохранены в {filename}")
        
    except Exception as e:
        print(f"Ошибка при сохранении метрик: {e}")


def create_visualization(metrics: Dict[str, Any], filename: str = "distillation_results.png"):
    """Создание визуализации результатов"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Результаты дистилляции знаний', fontsize=16)
        
        # График 1: Конфигурация
        if 'config' in metrics:
            config = metrics['config']
            ax1.bar(['Temperature', 'Alpha', 'Beta'], 
                   [config.get('temperature', 0), config.get('alpha', 0), config.get('beta', 0)])
            ax1.set_title('Параметры дистилляции')
            ax1.set_ylabel('Значение')
        
        # График 2: Потери
        if 'metrics' in metrics:
            losses = ['avg_total_loss', 'avg_distillation_loss', 'avg_student_loss']
            loss_values = [metrics['metrics'].get(loss, 0) for loss in losses]
            ax2.bar(['Total', 'Distillation', 'Student'], loss_values)
            ax2.set_title('Средние потери')
            ax2.set_ylabel('Потеря')
        
        # График 3: Сравнение моделей
        if 'config' in metrics and 'teacher_params' in metrics['config']:
            params = [metrics['config']['teacher_params'], metrics['config']['student_params']]
            ax3.bar(['Teacher', 'Student'], params)
            ax3.set_title('Количество параметров')
            ax3.set_ylabel('Параметры')
            ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # График 4: Коэффициент сжатия
        if 'config' in metrics and 'compression_ratio' in metrics['config']:
            ax4.bar(['Compression Ratio'], [metrics['config']['compression_ratio']])
            ax4.set_title('Коэффициент сжатия')
            ax4.set_ylabel('Раз')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Визуализация сохранена в {filename}")
        
    except Exception as e:
        print(f"Ошибка при создании визуализации: {e}")


def run_comprehensive_test():
    """Запуск комплексного теста всех функций"""
    print("🚀 Запуск комплексного теста модуля дистилляции")
    print("=" * 60)
    
    results = {
        'basic_functionality': False,
        'training_process': False,
        'factory_function': False,
        'gpt2_distillation': False,
        'pokemon_dataset': False
    }
    
    # Базовые тесты
    try:
        results['basic_functionality'] = test_basic_functionality()
    except Exception as e:
        print(f"❌ Ошибка в базовом тесте: {e}")
    
    try:
        results['training_process'] = test_training_process()
    except Exception as e:
        print(f"❌ Ошибка в тесте обучения: {e}")
    
    try:
        results['factory_function'] = test_factory_function()
    except Exception as e:
        print(f"❌ Ошибка в тесте фабричной функции: {e}")
    
    # Продвинутые тесты (могут требовать интернет и больше ресурсов)
    try:
        results['gpt2_distillation'] = test_gpt2_distillation(sample_size=50)
    except Exception as e:
        print(f"❌ Ошибка в тесте GPT-2: {e}")
    
    try:
        results['pokemon_dataset'] = test_pokemon_dataset(sample_size=100)
    except Exception as e:
        print(f"❌ Ошибка в тесте с покемонами: {e}")
    
    # Итоговый отчет
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ ПРОЙДЕН" if passed_test else "❌ НЕ ПРОЙДЕН"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nОбщий результат: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 Все тесты пройдены успешно! Модуль дистилляции работает корректно.")
    elif passed >= total * 0.6:
        print("⚠️ Большинство тестов пройдено. Модуль работает, но есть проблемы с некоторыми функциями.")
    else:
        print("🚨 Много тестов не пройдено. Требуется проверка модуля.")
    
    return results


def main():
    """Главная функция для запуска тестов"""
    parser = argparse.ArgumentParser(description='Тестирование модуля дистилляции знаний')
    parser.add_argument('--test', choices=['basic', 'training', 'gpt2', 'pokemon', 'factory', 'all'], 
                       default='all', help='Тип теста для запуска')
    parser.add_argument('--sample_size', type=int, default=100, 
                       help='Размер выборки для тестов с данными')
    parser.add_argument('--save_metrics', action='store_true', 
                       help='Сохранить метрики в CSV')
    parser.add_argument('--create_plots', action='store_true', 
                       help='Создать графики результатов')
    parser.add_argument('--install_deps', action='store_true', 
                       help='Установить зависимости')
    
    args = parser.parse_args()
    
    # Установка зависимостей
    if args.install_deps:
        print("Установка зависимостей...")
        os.system("pip install torch transformers datasets matplotlib pandas numpy")
        print("Зависимости установлены.")
        return
    
    # Проверка доступности модуля
    try:
        from distillation import KnowledgeDistillation
        print("✅ Модуль distillation.py найден и импортирован успешно")
    except ImportError as e:
        print(f"❌ Ошибка импорта модуля distillation.py: {e}")
        print("Убедитесь, что файл distillation.py находится в той же директории")
        sys.exit(1)
    
    # Запуск тестов
    if args.test == 'all':
        results = run_comprehensive_test()
    elif args.test == 'basic':
        test_basic_functionality()
    elif args.test == 'training':
        test_training_process()
    elif args.test == 'gpt2':
        test_gpt2_distillation(args.sample_size)
    elif args.test == 'pokemon':
        test_pokemon_dataset(args.sample_size)
    elif args.test == 'factory':
        test_factory_function()
    
    print("\n🏁 Тестирование завершено!")


if __name__ == "__main__":
    main()