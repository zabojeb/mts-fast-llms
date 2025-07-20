import os
import torch
import time
from distillation_logic import distill_gpt2_wikitext
from ghf.simple_parser import collect_model_data, get_model_info
from transformers import AutoModel, AutoConfig
import re

# Функция для извлечения количества параметров из модели
def get_model_parameters(model_name):
    try:
        # Попытка загрузить конфигурацию модели
        print(f"Загрузка конфигурации модели {model_name}...")
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, 'num_parameters'):
            return config.num_parameters / 1000000  # Конвертируем в миллионы
        
        # Если в конфигурации нет информации о параметрах, пробуем загрузить модель
        try:
            print("Загрузка модели для подсчета параметров...")
            model = AutoModel.from_pretrained(model_name)
            num_params = sum(p.numel() for p in model.parameters()) / 1000000  # Конвертируем в миллионы
            del model  # Освобождаем память
            return num_params
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            
        # Если не удалось получить параметры из модели, используем API Hugging Face
        print("Получение информации через API Hugging Face...")
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

# Функция для запуска интерактивной дистилляции
def run_interactive_distillation():
    print("\n=== ИНТЕРАКТИВНАЯ ДИСТИЛЛЯЦИЯ МОДЕЛЕЙ ===\n")
    print("Этот скрипт поможет вам настроить и запустить процесс дистилляции модели.")
    print("Вы укажете модель учителя и желаемый коэффициент сжатия, а система подберет подходящие модели студентов.\n")
    
    # Запрос модели учителя
    teacher_model = input("Введите название модели учителя (например, 'gpt2-xl', 'facebook/opt-1.3b'): ").strip()
    if not teacher_model:
        teacher_model = "gpt2-xl"
        print(f"Используется модель учителя по умолчанию: {teacher_model}")
    
    # Получение количества параметров модели учителя
    print(f"\nПолучение информации о модели учителя {teacher_model}...")
    teacher_params = get_model_parameters(teacher_model)
    
    if teacher_params is None:
        print("Не удалось автоматически определить количество параметров модели учителя.")
        while True:
            try:
                teacher_params = float(input("Введите количество параметров модели учителя в миллионах (например, 1500 для 1.5B): "))
                break
            except ValueError:
                print("Пожалуйста, введите корректное число.")
    else:
        print(f"Модель учителя имеет {teacher_params:.2f} миллионов параметров")
    
    # Запрос коэффициента сжатия
    while True:
        try:
            compression_ratio = float(input(f"\nВведите коэффициент сжатия (например, 2 для уменьшения в 2 раза): "))
            if compression_ratio <= 0:
                print("Коэффициент сжатия должен быть положительным числом.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите корректное число.")
    
    # Расчет целевого количества параметров
    target_params = teacher_params / compression_ratio
    print(f"Целевое количество параметров: {target_params:.2f} миллионов")
    
    # Запрос разброса параметров
    while True:
        div_input = input(f"\nВведите допустимый разброс параметров в миллионах (нажмите Enter для значения по умолчанию {target_params * 0.2:.2f}): ").strip()
        if not div_input:
            div = target_params * 0.2
            print(f"Используется разброс по умолчанию: {div:.2f} миллионов")
            break
        try:
            div = float(div_input)
            if div < 0:
                print("Разброс должен быть неотрицательным числом.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите корректное число.")
    
    # Запрос минимального количества лайков
    while True:
        likes_input = input("\nВведите минимальное количество лайков для моделей (нажмите Enter для значения по умолчанию 50): ").strip()
        if not likes_input:
            min_likes = 50
            print(f"Используется минимальное количество лайков по умолчанию: {min_likes}")
            break
        try:
            min_likes = int(likes_input)
            if min_likes < 0:
                print("Количество лайков должно быть неотрицательным числом.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите корректное целое число.")
    
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
    
    # Запрос типа дистилляции
    while True:
        box_type = input("\nВыберите тип дистилляции (white/black, по умолчанию white): ").strip().lower()
        if not box_type:
            box_type = "white"
            print(f"Используется тип дистилляции по умолчанию: {box_type}")
            break
        if box_type in ["white", "black"]:
            break
        print("Пожалуйста, введите 'white' или 'black'.")
    
    # Запрос датасета
    while True:
        dataset = input("\nВыберите датасет (wikitext/pokemon/custom, по умолчанию wikitext): ").strip().lower()
        if not dataset:
            dataset = "wikitext"
            print(f"Используется датасет по умолчанию: {dataset}")
            break
        if dataset in ["wikitext", "pokemon", "custom"]:
            break
        print("Пожалуйста, введите 'wikitext', 'pokemon' или 'custom'.")
    
    # Запрос пути к пользовательскому датасету, если выбран custom
    dataset_path = None
    if dataset == "custom":
        dataset_path = input("Введите путь к пользовательскому датасету: ").strip()
        while not os.path.exists(dataset_path):
            print("Указанный путь не существует.")
            dataset_path = input("Введите корректный путь к датасету или оставьте пустым для отмены: ").strip()
            if not dataset_path:
                dataset = "wikitext"
                print(f"Используется датасет по умолчанию: {dataset}")
                break
    
    # Запрос размера выборки
    while True:
        sample_size_input = input("\nВведите размер выборки из датасета (0 для всего датасета, по умолчанию 1000): ").strip()
        if not sample_size_input:
            sample_size = 1000
            print(f"Используется размер выборки по умолчанию: {sample_size}")
            break
        try:
            sample_size = int(sample_size_input)
            if sample_size < 0:
                print("Размер выборки должен быть неотрицательным числом.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите корректное целое число.")
    
    # Запрос количества эпох
    while True:
        epochs_input = input("\nВведите количество эпох обучения (по умолчанию 2): ").strip()
        if not epochs_input:
            epochs = 2
            print(f"Используется количество эпох по умолчанию: {epochs}")
            break
        try:
            epochs = int(epochs_input)
            if epochs <= 0:
                print("Количество эпох должно быть положительным числом.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите корректное целое число.")
    
    # Запрос размера батча
    while True:
        batch_size_input = input("\nВведите размер батча (по умолчанию 8): ").strip()
        if not batch_size_input:
            batch_size = 8
            print(f"Используется размер батча по умолчанию: {batch_size}")
            break
        try:
            batch_size = int(batch_size_input)
            if batch_size <= 0:
                print("Размер батча должен быть положительным числом.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите корректное целое число.")
    
    # Запрос скорости обучения
    while True:
        lr_input = input("\nВведите скорость обучения (по умолчанию 5e-5): ").strip()
        if not lr_input:
            learning_rate = 5e-5
            print(f"Используется скорость обучения по умолчанию: {learning_rate}")
            break
        try:
            learning_rate = float(lr_input)
            if learning_rate <= 0:
                print("Скорость обучения должна быть положительным числом.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите корректное число.")
    
    # Запрос температуры
    while True:
        temp_input = input("\nВведите температуру для дистилляции (по умолчанию 2.0): ").strip()
        if not temp_input:
            temperature = 2.0
            print(f"Используется температура по умолчанию: {temperature}")
            break
        try:
            temperature = float(temp_input)
            if temperature <= 0:
                print("Температура должна быть положительным числом.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите корректное число.")
    
    # Запрос alpha и beta
    while True:
        alpha_input = input("\nВведите вес для soft targets (alpha, по умолчанию 0.7): ").strip()
        if not alpha_input:
            alpha = 0.7
            print(f"Используется alpha по умолчанию: {alpha}")
            break
        try:
            alpha = float(alpha_input)
            if alpha < 0 or alpha > 1:
                print("Alpha должна быть в диапазоне от 0 до 1.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите корректное число.")
    
    while True:
        beta_input = input(f"\nВведите вес для hard targets (beta, по умолчанию {1-alpha:.1f}): ").strip()
        if not beta_input:
            beta = 1 - alpha
            print(f"Используется beta по умолчанию: {beta}")
            break
        try:
            beta = float(beta_input)
            if beta < 0 or beta > 1:
                print("Beta должна быть в диапазоне от 0 до 1.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите корректное число.")
    
    # Запрос максимальной длины последовательности
    while True:
        max_length_input = input("\nВведите максимальную длину последовательности (по умолчанию 256): ").strip()
        if not max_length_input:
            max_length = 256
            print(f"Используется максимальная длина последовательности по умолчанию: {max_length}")
            break
        try:
            max_length = int(max_length_input)
            if max_length <= 0:
                print("Максимальная длина последовательности должна быть положительным числом.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите корректное целое число.")
    
    # Запрос директории для сохранения результатов
    output_dir = input("\nВведите директорию для сохранения результатов (по умолчанию ./output): ").strip()
    if not output_dir:
        output_dir = "./output"
        print(f"Используется директория по умолчанию: {output_dir}")
    
    # Запрос использования CUDA
    no_cuda = False
    if torch.cuda.is_available():
        cuda_input = input("\nИспользовать GPU для обучения? (y/n, по умолчанию y): ").strip().lower()
        if cuda_input == "n":
            no_cuda = True
            print("Будет использоваться CPU для обучения.")
        else:
            print("Будет использоваться GPU для обучения.")
    else:
        print("GPU не доступен, будет использоваться CPU для обучения.")
        no_cuda = True
    
    # Запрос пропуска валидации
    skip_validation_input = input("\nПропустить валидацию улучшения студента? (y/n, по умолчанию n): ").strip().lower()
    skip_validation = skip_validation_input == "y"
    if skip_validation:
        print("Валидация улучшения студента будет пропущена.")
    else:
        print("Будет выполнена валидация улучшения студента.")
    
    # Создаем объект args для передачи в функцию дистилляции
    class DistillationArgs:
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
            self.max_length = max_length
            self.sample_size = sample_size
            self.save_metrics = True
            self.save_plot = True
            self.no_cuda = no_cuda
            self.output_dir = output_dir
            self.checkpoint_interval = 1
            self.run_tests = False
            self.skip_demo = True
            self.install_deps = True
            self.skip_validation = skip_validation
            self.create_comparison_table = True
            self.save_best_model = True
            self.folder_name = None
            self.box = box_type
    
    distillation_args = DistillationArgs()
    
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
    
    # Запрос подтверждения запуска
    confirm = input("\nНачать дистилляцию? (y/n): ").strip().lower()
    if confirm != "y":
        print("Дистилляция отменена.")
        return
    
    # Запускаем дистилляцию
    start_time = time.time()
    trained_student, metrics = distill_gpt2_wikitext(distillation_args)
    elapsed_time = time.time() - start_time
    
    # Выводим итоговую информацию
    print("\n=== ДИСТИЛЛЯЦИЯ ЗАВЕРШЕНА ===\n")
    print(f"Коэффициент сжатия: {metrics['config']['compression_ratio']:.2f}x")
    print(f"Средняя потеря студента: {metrics['metrics']['avg_student_loss']:.4f}")
    print(f"Время выполнения: {metrics['metrics']['elapsed_time']:.2f} секунд")
    print(f"Результаты сохранены в: {metrics['config']['output_dir']}")
    
    # Возвращаем обученную модель и метрики
    return trained_student, metrics


if __name__ == "__main__":
    run_interactive_distillation()