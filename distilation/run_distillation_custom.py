import os
import torch
import copy
import sys
import re
import pandas as pd
from distillation_logic import distill_gpt2_wikitext
from ghf.simple_parser import collect_model_data, get_model_info
from transformers import AutoModel, AutoConfig

# Функция для извлечения количества параметров из модели
def get_model_parameters(model_name):
    try:
        # Попытка загрузить конфигурацию модели
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, 'num_parameters'):
            return config.num_parameters / 1000000  # Конвертируем в миллионы
        
        # Если в конфигурации нет информации о параметрах, пробуем загрузить модель
        try:
            model = AutoModel.from_pretrained(model_name)
            num_params = sum(p.numel() for p in model.parameters()) / 1000000  # Конвертируем в миллионы
            del model  # Освобождаем память
            return num_params
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            
        # Если не удалось получить параметры из модели, используем API Hugging Face
        model_info = get_model_info(model_name)
        if model_info and 'parameters_numeric' in model_info and model_info['parameters_numeric']:
            return model_info['parameters_numeric']
            
        # Пытаемся извлечь из названия модели
        match = re.search(r'(\d+(\.\d+)?)([BbMm])', model_name)
        if match:
            value = float(match.group(1))
            unit = match.group(3).upper()
            if unit == 'B':
                return value * 1000  # Конвертируем миллиарды в миллионы
            elif unit == 'M':
                return value
    except Exception as e:
        print(f"Ошибка при определении параметров модели: {e}")
    
    # Если не удалось определить количество параметров
    return None

# Функция для запуска дистилляции с пользовательскими параметрами
def run_custom_distillation():
    print("\n=== НАСТРОЙКА ПОЛЬЗОВАТЕЛЬСКОЙ ДИСТИЛЛЯЦИИ ===\n")
    
    # Запрос модели учителя
    teacher_model = input("Введите название модели учителя (например, 'gpt2-xl', 'Qwen/Qwen-1_8B'): ")
    if not teacher_model:
        teacher_model = 'gpt2-xl'
        print(f"Используется модель учителя по умолчанию: {teacher_model}")
    
    # Получаем количество параметров модели учителя
    print(f"\nПолучение информации о модели учителя {teacher_model}...")
    teacher_params = get_model_parameters(teacher_model)
    
    if teacher_params is None:
        print("Не удалось определить количество параметров модели учителя.")
        teacher_params = float(input("Введите количество параметров модели учителя в миллионах (например, 1500 для 1.5B): "))
    else:
        print(f"Модель учителя имеет {teacher_params:.2f} миллионов параметров")
    
    # Запрос коэффициента сжатия
    compression_ratio = float(input("\nВведите коэффициент сжатия (например, 2 для уменьшения в 2 раза): "))
    if compression_ratio <= 0:
        compression_ratio = 2
        print(f"Используется коэффициент сжатия по умолчанию: {compression_ratio}")
    
    # Расчет целевого количества параметров
    target_params = teacher_params / compression_ratio
    print(f"Целевое количество параметров: {target_params:.2f} миллионов")
    
    # Запрос разброса параметров
    div = float(input("\nВведите допустимый разброс параметров в миллионах (например, 30): "))
    if div <= 0:
        div = target_params * 0.2  # По умолчанию 20% от целевого значения
        print(f"Используется разброс по умолчанию: {div:.2f} миллионов")
    
    # Поиск подходящих моделей
    print("\nПоиск подходящих моделей на Hugging Face...")
    min_models = 5  # Минимальное количество моделей для выбора
    
    # Собираем данные о моделях с заданными параметрами
    models_data = collect_model_data(
        min_likes=50,  # Минимальное количество лайков
        max_models=100,  # Максимальное количество моделей для анализа
        min_models=min_models,  # Минимальное количество моделей после фильтрации
        sort_by='likes',  # Сортировка по лайкам
        num_params=target_params,  # Целевое количество параметров
        div=div  # Разброс параметров
    )
    
    # Проверяем, что нашли достаточно моделей
    if models_data is None or len(models_data) < min_models:
        print(f"Не удалось найти достаточно моделей. Расширяем диапазон поиска...")
        div = div * 2  # Увеличиваем разброс в 2 раза
        models_data = collect_model_data(
            min_likes=20,  # Снижаем требования к лайкам
            max_models=200,  # Увеличиваем максимальное количество моделей
            min_models=min_models,
            sort_by='likes',
            num_params=target_params,
            div=div
        )
    
    # Если все еще не нашли достаточно моделей
    if models_data is None or len(models_data) < min_models:
        print("Не удалось найти подходящие модели. Пожалуйста, измените параметры поиска.")
        return
    
    # Отображаем топ-5 моделей (или меньше, если нашли меньше)
    top_models = min(5, len(models_data))
    print(f"\nНайдено {len(models_data)} подходящих моделей. Топ-{top_models}:")
    
    # Создаем список моделей для выбора
    model_options = []
    for i in range(top_models):
        model = models_data.iloc[i]
        model_name = model['id']
        params = model['parameters_numeric'] if 'parameters_numeric' in model and model['parameters_numeric'] else 'Неизвестно'
        likes = model['likes'] if 'likes' in model else 'Неизвестно'
        
        # Добавляем модель в список опций
        model_options.append(model_name)
        
        # Выводим информацию о модели
        print(f"{i+1}. {model_name}")
        print(f"   Параметры: {params} млн")
        print(f"   Лайки: {likes}")
        if 'downloads' in model:
            print(f"   Загрузки: {model['downloads']}")
        print()
    
    # Запрос выбора модели студента
    while True:
        try:
            choice = int(input(f"Выберите модель студента (1-{top_models}): "))
            if 1 <= choice <= top_models:
                student_model = model_options[choice-1]
                break
            else:
                print(f"Пожалуйста, введите число от 1 до {top_models}.")
        except ValueError:
            print("Пожалуйста, введите корректное число.")
    
    print(f"\nВыбрана модель студента: {student_model}")
    
    # Запрос параметров дистилляции
    print("\n=== НАСТРОЙКА ПАРАМЕТРОВ ДИСТИЛЛЯЦИИ ===\n")
    
    # Тип дистилляции
    box_type = input("Выберите тип дистилляции (white/black, по умолчанию white): ").lower()
    if box_type not in ['white', 'black']:
        box_type = 'white'
        print(f"Используется тип дистилляции по умолчанию: {box_type}")
    
    # Датасет
    dataset = input("Выберите датасет (wikitext/pokemon/custom, по умолчанию wikitext): ").lower()
    if dataset not in ['wikitext', 'pokemon', 'custom']:
        dataset = 'wikitext'
        print(f"Используется датасет по умолчанию: {dataset}")
    
    dataset_path = None
    if dataset == 'custom':
        dataset_path = input("Введите путь к пользовательскому датасету: ")
        if not os.path.exists(dataset_path):
            print(f"Путь {dataset_path} не существует. Используется датасет wikitext.")
            dataset = 'wikitext'
            dataset_path = None
    
    # Размер выборки
    try:
        sample_size = int(input("Введите размер выборки (по умолчанию 1000, 0 для всего датасета): "))
        if sample_size < 0:
            sample_size = 1000
            print(f"Используется размер выборки по умолчанию: {sample_size}")
    except ValueError:
        sample_size = 1000
        print(f"Используется размер выборки по умолчанию: {sample_size}")
    
    # Параметры обучения
    try:
        epochs = int(input("Введите количество эпох (по умолчанию 2): "))
        if epochs <= 0:
            epochs = 2
            print(f"Используется количество эпох по умолчанию: {epochs}")
    except ValueError:
        epochs = 2
        print(f"Используется количество эпох по умолчанию: {epochs}")
    
    try:
        batch_size = int(input("Введите размер батча (по умолчанию 8): "))
        if batch_size <= 0:
            batch_size = 8
            print(f"Используется размер батча по умолчанию: {batch_size}")
    except ValueError:
        batch_size = 8
        print(f"Используется размер батча по умолчанию: {batch_size}")
    
    try:
        learning_rate = float(input("Введите скорость обучения (по умолчанию 5e-5): "))
        if learning_rate <= 0:
            learning_rate = 5e-5
            print(f"Используется скорость обучения по умолчанию: {learning_rate}")
    except ValueError:
        learning_rate = 5e-5
        print(f"Используется скорость обучения по умолчанию: {learning_rate}")
    
    # Параметры дистилляции
    try:
        temperature = float(input("Введите температуру дистилляции (по умолчанию 2.0): "))
        if temperature <= 0:
            temperature = 2.0
            print(f"Используется температура по умолчанию: {temperature}")
    except ValueError:
        temperature = 2.0
        print(f"Используется температура по умолчанию: {temperature}")
    
    try:
        alpha = float(input("Введите вес для soft targets (по умолчанию 0.7): "))
        if alpha < 0 or alpha > 1:
            alpha = 0.7
            print(f"Используется вес для soft targets по умолчанию: {alpha}")
    except ValueError:
        alpha = 0.7
        print(f"Используется вес для soft targets по умолчанию: {alpha}")
    
    try:
        beta = float(input("Введите вес для hard targets (по умолчанию 0.3): "))
        if beta < 0 or beta > 1:
            beta = 0.3
            print(f"Используется вес для hard targets по умолчанию: {beta}")
    except ValueError:
        beta = 0.3
        print(f"Используется вес для hard targets по умолчанию: {beta}")
    
    # Создаем объект args для передачи в функцию дистилляции
    class Args:
        def __init__(self):
            self.teacher_model = teacher_model
            self.student_model = student_model
            self.dataset = dataset
            self.dataset_path = dataset_path
            self.temperature = temperature
            self.alpha = alpha
            self.beta = beta
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = learning_rate
            self.max_length = 256  # Стандартное значение
            self.sample_size = sample_size
            self.save_metrics = True
            self.save_plot = True
            self.no_cuda = not torch.cuda.is_available()
            self.output_dir = './output'
            self.checkpoint_interval = 1
            self.run_tests = False
            self.skip_demo = True
            self.install_deps = True
            self.skip_validation = False
            self.create_comparison_table = True
            self.save_best_model = True
            self.folder_name = None
            self.box = box_type
    
    args = Args()
    
    # Запускаем дистилляцию
    print("\n=== ЗАПУСК ДИСТИЛЛЯЦИИ С НАСТРОЕННЫМИ ПАРАМЕТРАМИ ===\n")
    print(f"Модель учителя: {teacher_model} ({teacher_params:.2f} млн параметров)")
    print(f"Модель студента: {student_model}")
    print(f"Коэффициент сжатия: {compression_ratio:.2f}x")
    print(f"Тип дистилляции: {box_type}")
    print(f"Датасет: {dataset}" + (f" ({dataset_path})" if dataset_path else ""))
    print(f"Размер выборки: {sample_size}")
    print(f"Эпохи: {epochs}, Батч: {batch_size}, Скорость обучения: {learning_rate}")
    print(f"Температура: {temperature}, Alpha: {alpha}, Beta: {beta}")
    
    # Запускаем дистилляцию
    trained_student, metrics = distill_gpt2_wikitext(args)
    
    # Выводим итоговую информацию
    print("\n=== ДИСТИЛЛЯЦИЯ ЗАВЕРШЕНА ===\n")
    print(f"Коэффициент сжатия: {metrics['config']['compression_ratio']:.2f}x")
    print(f"Средняя потеря студента: {metrics['metrics']['avg_student_loss']:.4f}")
    print(f"Время выполнения: {metrics['metrics']['elapsed_time']:.2f} секунд")
    
    # Возвращаем обученную модель и метрики
    return trained_student, metrics


if __name__ == "__main__":
    run_custom_distillation()