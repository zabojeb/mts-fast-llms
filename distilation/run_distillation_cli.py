import os
import torch
import argparse
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

# Функция для поиска и выбора модели студента
def find_student_model(teacher_params, compression_ratio, div=None, auto_select=False, min_likes=50):
    # Расчет целевого количества параметров
    target_params = teacher_params / compression_ratio
    print(f"Целевое количество параметров: {target_params:.2f} миллионов")
    
    # Если разброс не указан, используем 20% от целевого значения
    if div is None or div <= 0:
        div = target_params * 0.2
        print(f"Используется разброс по умолчанию: {div:.2f} миллионов")
    
    # Поиск подходящих моделей
    print("\nПоиск подходящих моделей на Hugging Face...")
    min_models = 5  # Минимальное количество моделей для выбора
    
    # Собираем данные о моделях с заданными параметрами
    models_data = collect_model_data(
        min_likes=min_likes,  # Минимальное количество лайков
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
            min_likes=min_likes // 2,  # Снижаем требования к лайкам
            max_models=200,  # Увеличиваем максимальное количество моделей
            min_models=min_models,
            sort_by='likes',
            num_params=target_params,
            div=div
        )
    
    # Если все еще не нашли достаточно моделей
    if models_data is None or len(models_data) < min_models:
        print("Не удалось найти подходящие модели. Пожалуйста, измените параметры поиска.")
        return None
    
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
    
    # Если указан автоматический выбор, берем первую модель
    if auto_select:
        student_model = model_options[0]
        print(f"Автоматически выбрана модель студента: {student_model}")
        return student_model
    
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
    return student_model

# Функция для запуска дистилляции с аргументами командной строки
def main():
    parser = argparse.ArgumentParser(description="Запуск дистилляции с автоматическим подбором модели студента")
    
    # Основные параметры
    parser.add_argument("--teacher", type=str, default="gpt2-xl", help="Название модели учителя (например, 'gpt2-xl')")
    parser.add_argument("--compression", type=float, default=2.0, help="Коэффициент сжатия (например, 2 для уменьшения в 2 раза)")
    parser.add_argument("--div", type=float, default=None, help="Допустимый разброс параметров в миллионах")
    parser.add_argument("--student", type=str, default=None, help="Название модели студента (если не указано, будет выполнен автоматический поиск)")
    parser.add_argument("--auto-select", action="store_true", help="Автоматически выбрать первую найденную модель студента")
    parser.add_argument("--min-likes", type=int, default=50, help="Минимальное количество лайков для моделей при поиске")
    
    # Параметры дистилляции
    parser.add_argument("--box-type", type=str, default="white", choices=["white", "black"], help="Тип дистилляции (white/black)")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "pokemon", "custom"], help="Датасет для обучения")
    parser.add_argument("--dataset-path", type=str, default=None, help="Путь к пользовательскому датасету")
    parser.add_argument("--sample-size", type=int, default=1000, help="Размер выборки из датасета (0 для всего датасета)")
    parser.add_argument("--epochs", type=int, default=2, help="Количество эпох обучения")
    parser.add_argument("--batch-size", type=int, default=8, help="Размер батча")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Скорость обучения")
    parser.add_argument("--temperature", type=float, default=2.0, help="Температура для дистилляции")
    parser.add_argument("--alpha", type=float, default=0.7, help="Вес для soft targets (KL-дивергенция)")
    parser.add_argument("--beta", type=float, default=0.3, help="Вес для hard targets (обычная функция потерь)")
    parser.add_argument("--max-length", type=int, default=256, help="Максимальная длина последовательности")
    parser.add_argument("--output-dir", type=str, default="./output", help="Директория для сохранения результатов")
    parser.add_argument("--no-cuda", action="store_true", help="Не использовать CUDA (GPU)")
    parser.add_argument("--skip-validation", action="store_true", help="Пропустить валидацию улучшения студента")
    
    args = parser.parse_args()
    
    # Получаем количество параметров модели учителя
    print(f"\nПолучение информации о модели учителя {args.teacher}...")
    teacher_params = get_model_parameters(args.teacher)
    
    if teacher_params is None:
        print("Не удалось определить количество параметров модели учителя.")
        teacher_params = float(input("Введите количество параметров модели учителя в миллионах (например, 1500 для 1.5B): "))
    else:
        print(f"Модель учителя имеет {teacher_params:.2f} миллионов параметров")
    
    # Если модель студента не указана, выполняем поиск
    student_model = args.student
    if student_model is None:
        student_model = find_student_model(
            teacher_params=teacher_params,
            compression_ratio=args.compression,
            div=args.div,
            auto_select=args.auto_select,
            min_likes=args.min_likes
        )
        
        if student_model is None:
            print("Не удалось найти подходящую модель студента. Завершение работы.")
            return
    
    # Создаем объект args для передачи в функцию дистилляции
    class DistillationArgs:
        def __init__(self):
            self.teacher_model = args.teacher
            self.student_model = student_model
            self.dataset = args.dataset
            self.dataset_path = args.dataset_path
            self.temperature = args.temperature
            self.alpha = args.alpha
            self.beta = args.beta
            self.epochs = args.epochs
            self.batch_size = args.batch_size
            self.lr = args.learning_rate
            self.max_length = args.max_length
            self.sample_size = args.sample_size
            self.save_metrics = True
            self.save_plot = True
            self.no_cuda = args.no_cuda if args.no_cuda else not torch.cuda.is_available()
            self.output_dir = args.output_dir
            self.checkpoint_interval = 1
            self.run_tests = False
            self.skip_demo = True
            self.install_deps = True
            self.skip_validation = args.skip_validation
            self.create_comparison_table = True
            self.save_best_model = True
            self.folder_name = None
            self.box = args.box_type
    
    distillation_args = DistillationArgs()
    
    # Запускаем дистилляцию
    print("\n=== ЗАПУСК ДИСТИЛЛЯЦИИ С НАСТРОЕННЫМИ ПАРАМЕТРАМИ ===\n")
    print(f"Модель учителя: {args.teacher} ({teacher_params:.2f} млн параметров)")
    print(f"Модель студента: {student_model}")
    print(f"Коэффициент сжатия: {args.compression:.2f}x")
    print(f"Тип дистилляции: {args.box_type}")
    print(f"Датасет: {args.dataset}" + (f" ({args.dataset_path})" if args.dataset_path else ""))
    print(f"Размер выборки: {args.sample_size}")
    print(f"Эпохи: {args.epochs}, Батч: {args.batch_size}, Скорость обучения: {args.learning_rate}")
    print(f"Температура: {args.temperature}, Alpha: {args.alpha}, Beta: {args.beta}")
    
    # Запускаем дистилляцию
    trained_student, metrics = distill_gpt2_wikitext(distillation_args)
    
    # Выводим итоговую информацию
    print("\n=== ДИСТИЛЛЯЦИЯ ЗАВЕРШЕНА ===\n")
    print(f"Коэффициент сжатия: {metrics['config']['compression_ratio']:.2f}x")
    print(f"Средняя потеря студента: {metrics['metrics']['avg_student_loss']:.4f}")
    print(f"Время выполнения: {metrics['metrics']['elapsed_time']:.2f} секунд")
    
    # Возвращаем обученную модель и метрики
    return trained_student, metrics


if __name__ == "__main__":
    main()