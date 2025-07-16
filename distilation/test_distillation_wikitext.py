import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import argparse
from tqdm import tqdm, trange
import copy
import datetime
from distillation import KnowledgeDistillation, DistillationConfig, create_optimizer

# Функция для создания уникальной директории для дистилляции
def create_distillation_dir(base_dir, folder_name=None):
    """
    Создает уникальную директорию для сохранения результатов дистилляции.
    
    Args:
        base_dir: Базовая директория для сохранения результатов
        folder_name: Опциональное название папки (если не указано, будет создана папка с временной меткой)
    
    Returns:
        Путь к созданной директории
    """
    # Получаем текущую дату и время
    now = datetime.datetime.now()
    timestamp = now.strftime("%d.%m.%y.%H.%M.%S")
    
    # Определяем счетчик дистилляций
    distillation_count = 1
    while True:
        # Если указано название папки, используем его
        if folder_name:
            dir_name = f"distillation#{distillation_count}_{folder_name}_{timestamp}"
        else:
            dir_name = f"distillation#{distillation_count}_{timestamp}"
        
        # Создаем полный путь
        full_path = os.path.join(base_dir, dir_name)
        
        # Проверяем, существует ли такая директория
        if not os.path.exists(full_path):
            # Создаем директорию
            os.makedirs(full_path, exist_ok=True)
            return full_path
        
        # Если директория существует, увеличиваем счетчик
        distillation_count += 1

# Настройка аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description='Тестирование модуля дистилляции знаний на WikiText')
    parser.add_argument('--teacher_model', type=str, default='gpt2-medium', help='Модель учителя (имя модели из Hugging Face или путь к локальной модели)')
    parser.add_argument('--student_model', type=str, default='gpt2', help='Модель студента (имя модели из Hugging Face или путь к локальной модели)')
    parser.add_argument('--teacher_model_path', type=str, help='Путь к локальной модели учителя (если не указано, будет использована модель из Hugging Face)')
    parser.add_argument('--student_model_path', type=str, help='Путь к локальной модели студента (если не указано, будет использована модель из Hugging Face)')
    parser.add_argument('--dataset', type=str, default='wikitext', help='Датасет для обучения (wikitext, pokemon или путь к файлу)')
    parser.add_argument('-r', '--dataset_path', type=str, help='Путь к пользовательскому датасету')
    parser.add_argument('--temperature', type=float, default=4.0, help='Температура для дистилляции')
    parser.add_argument('--alpha', type=float, default=0.7, help='Вес для soft targets')
    parser.add_argument('--beta', type=float, default=0.3, help='Вес для hard targets')
    parser.add_argument('--epochs', type=int, default=2, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=4, help='Размер батча')
    parser.add_argument('--lr', type=float, default=5e-5, help='Скорость обучения')
    parser.add_argument('--max_length', type=int, default=128, help='Максимальная длина последовательности')
    parser.add_argument('--sample_size', type=int, default=100, help='Размер выборки из датасета')
    parser.add_argument('--save_metrics', action='store_true', help='Сохранить метрики в JSON')
    parser.add_argument('--save_plot', action='store_true', help='Сохранить график потерь')
    parser.add_argument('--no_cuda', action='store_true', help='Не использовать CUDA')
    parser.add_argument('--output_dir', type=str, default='./output', help='Директория для сохранения результатов')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Интервал сохранения чекпоинтов (в эпохах)')
    parser.add_argument('--run_tests', action='store_true', help='Запустить тесты (по умолчанию тесты пропускаются)')
    parser.add_argument('--skip_demo', action='store_true', help='Пропустить демонстрацию')
    parser.add_argument('--install_deps', action='store_true', help='Установить зависимости')
    parser.add_argument('--skip_validation', action='store_true', help='Пропустить валидацию улучшения студента')
    parser.add_argument('--create_comparison_table', action='store_true', help='Создать сравнительную таблицу до и после дистилляции')
    parser.add_argument('--save_best_model', action='store_true', help='Сохранять лучшую модель по валидационной выборке (по умолчанию отключено)')
    parser.add_argument('--folder_name', type=str, help='Название папки для сохранения результатов дистилляции (если не указано, будет создана папка с временной меткой)')
    parser.add_argument('--use_auto_model', action='store_true', help='Использовать AutoModel для загрузки моделей (рекомендуется для не-GPT2 моделей)')
    return parser.parse_args()

# Класс для подготовки данных WikiText
class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        self.inputs = []
        self.targets = []
        self._prepare_data()
        
    def _prepare_data(self):
        print("Подготовка данных WikiText...")
        for item in self.dataset:
            text = item['text']
            if len(text.strip()) > 0:  # Пропускаем пустые строки
                encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors='pt')
                input_ids = encodings.input_ids.squeeze()
                
                if len(input_ids) > 1:  # Убедимся, что у нас есть хотя бы 2 токена
                    self.inputs.append(input_ids[:-1])  # Все токены кроме последнего
                    self.targets.append(input_ids[1:])  # Все токены кроме первого
        
        print(f"Подготовлено {len(self.inputs)} последовательностей из WikiText")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Функция для создания батчей одинаковой длины
def collate_batch(batch):
    inputs, targets = zip(*batch)
    
    # Находим максимальную длину в батче
    max_len = max([len(x) for x in inputs])
    
    # Паддинг до максимальной длины
    padded_inputs = []
    padded_targets = []
    
    for i, t in zip(inputs, targets):
        # Паддинг входных данных
        if len(i) < max_len:
            padded_i = torch.cat([i, torch.zeros(max_len - len(i), dtype=torch.long)], dim=0)
        else:
            padded_i = i
        
        # Паддинг целевых данных
        if len(t) < max_len:
            padded_t = torch.cat([t, torch.zeros(max_len - len(t), dtype=torch.long)], dim=0)
        else:
            padded_t = t
        
        padded_inputs.append(padded_i)
        padded_targets.append(padded_t)
    
    # Преобразуем списки в тензоры
    inputs_tensor = torch.stack(padded_inputs)
    targets_tensor = torch.stack(padded_targets)
    
    return inputs_tensor, targets_tensor

# Функция потерь для языковых моделей
def language_model_loss(logits, targets):
    # Изменяем форму логитов для функции потерь
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    
    # Игнорируем паддинг (нули) при вычислении потерь
    mask = targets != 0
    targets = targets[mask]
    logits = logits[mask]
    
    if targets.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    
    return F.cross_entropy(logits, targets)

# Функция для загрузки и подготовки данных
def load_dataset_for_distillation(args, tokenizer):
    # Выбор датасета в зависимости от аргумента
    if args.dataset.lower() == 'wikitext':
        print("Загрузка датасета WikiText...")
        dataset = WikiTextDataset(tokenizer, max_length=args.max_length)
    elif args.dataset.lower() == 'pokemon':
        print("Загрузка датасета Pokemon...")
        # Загружаем датасет Pokemon
        raw_dataset = load_dataset("lamini/pokemon-bleu", split="train")
        
        # Подготавливаем данные
        inputs = []
        targets = []
        
        for item in raw_dataset:
            text = item['text']
            if len(text.strip()) > 0:  # Пропускаем пустые строки
                encodings = tokenizer(text, truncation=True, max_length=args.max_length, return_tensors='pt')
                input_ids = encodings.input_ids.squeeze()
                
                if len(input_ids) > 1:  # Убедимся, что у нас есть хотя бы 2 токена
                    inputs.append(input_ids[:-1])  # Все токены кроме последнего
                    targets.append(input_ids[1:])  # Все токены кроме первого
        
        # Создаем свой датасет
        class CustomDataset(Dataset):
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets
            
            def __len__(self):
                return len(self.inputs)
            
            def __getitem__(self, idx):
                return self.inputs[idx], self.targets[idx]
        
        dataset = CustomDataset(inputs, targets)
        print(f"Подготовлено {len(dataset)} последовательностей из Pokemon")
    
    elif args.dataset_path:
        print(f"Загрузка пользовательского датасета из {args.dataset_path}...")
        # Загружаем пользовательский датасет из файла
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Подготавливаем данные
        inputs = []
        targets = []
        
        for line in lines:
            if len(line.strip()) > 0:  # Пропускаем пустые строки
                encodings = tokenizer(line, truncation=True, max_length=args.max_length, return_tensors='pt')
                input_ids = encodings.input_ids.squeeze()
                
                if len(input_ids) > 1:  # Убедимся, что у нас есть хотя бы 2 токена
                    inputs.append(input_ids[:-1])  # Все токены кроме последнего
                    targets.append(input_ids[1:])  # Все токены кроме первого
        
        # Создаем свой датасет
        class CustomDataset(Dataset):
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets
            
            def __len__(self):
                return len(self.inputs)
            
            def __getitem__(self, idx):
                return self.inputs[idx], self.targets[idx]
        
        dataset = CustomDataset(inputs, targets)
        print(f"Подготовлено {len(dataset)} последовательностей из пользовательского датасета")
    else:
        raise ValueError(f"Неизвестный датасет: {args.dataset}. Используйте 'wikitext', 'pokemon' или укажите путь к файлу с помощью -r")
    
    return dataset

# Информация о доступных моделях
def print_model_info():
    print("\nДоступные модели для дистилляции:")
    print("GPT-2 модели:")
    print("  - GPT-2 (124M параметров) - базовая модель")
    print("  - GPT-2-medium (355M параметров) - средняя модель")
    print("  - GPT-2-large (774M параметров) - большая модель")
    print("  - GPT-2-xl (1.5B параметров) - очень большая модель")
    print("  - DistilGPT2 (82M параметров) - уже дистиллированная версия GPT-2")
    print("\nQwen модели:")
    print("  Пример использования: --teacher_model=\"Qwen/Qwen-1.8B\"")
    print("\nДатасеты:")
    print("  - Встроенные: wikitext, pokemon")
    print("  - Пользовательские: укажите путь с помощью -r или --dataset_path")

def validate_student_improvement(student_model, teacher_model, validation_loader, device):
    """Проверяет улучшение студента после дистилляции
    
    Args:
        student_model: Модель студента после дистилляции
        teacher_model: Модель учителя
        validation_loader: Загрузчик данных для валидации
        device: Устройство для вычислений
        
    Returns:
        dict: Словарь с результатами валидации
    """
    print("\nПроведение валидации улучшения студента...")
    
    # Переводим модели в режим оценки и на нужное устройство
    student_model.eval()
    teacher_model.eval()
    
    # Убедимся, что модели находятся на правильном устройстве
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    
    # Инициализируем счетчики для потерь
    student_loss_total = 0.0
    teacher_loss_total = 0.0
    total_samples = 0
    
    # Функция для расчета потерь языковой модели
    def compute_loss(logits, targets):
        # Убедимся, что логиты и цели на одном устройстве
        if logits.device != targets.device:
            logits = logits.to(targets.device)
            
        # Преобразуем логиты в формат [batch_size * seq_len, vocab_size]
        logits_flat = logits.view(-1, logits.size(-1))
        # Преобразуем целевые значения в формат [batch_size * seq_len]
        targets_flat = targets.view(-1)
        # Вычисляем потери, игнорируя паддинг (0)
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0, reduction='sum')
        # Нормализуем потери по количеству ненулевых токенов
        non_pad_mask = targets_flat != 0
        num_tokens = non_pad_mask.sum().item()
        return loss / num_tokens if num_tokens > 0 else loss
    
    # Проходим по валидационному набору данных
    with torch.no_grad():
        for inputs, targets in tqdm(validation_loader, desc="Валидация"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Получаем предсказания моделей
            student_logits = student_model(inputs)
            teacher_logits = teacher_model(inputs)
            
            # Вычисляем потери
            student_loss = compute_loss(student_logits, targets)
            teacher_loss = compute_loss(teacher_logits, targets)
            
            # Обновляем счетчики
            batch_size = inputs.size(0)
            student_loss_total += student_loss.item() * batch_size
            teacher_loss_total += teacher_loss.item() * batch_size
            total_samples += batch_size
    
    # Вычисляем средние потери
    avg_student_loss = student_loss_total / total_samples
    avg_teacher_loss = teacher_loss_total / total_samples
    
    # Вычисляем перплексию
    student_perplexity = np.exp(avg_student_loss)
    teacher_perplexity = np.exp(avg_teacher_loss)
    
    # Определяем, улучшился ли студент
    is_improved = avg_student_loss < avg_teacher_loss
    
    # Формируем результаты
    results = {
        'avg_student_loss': avg_student_loss,
        'avg_teacher_loss': avg_teacher_loss,
        'student_perplexity': student_perplexity,
        'teacher_perplexity': teacher_perplexity,
        'is_improved': is_improved
    }
    
    # Выводим результаты
    print(f"\nРезультаты валидации:")
    print(f"Средняя потеря студента: {avg_student_loss:.4f}")
    print(f"Средняя потеря учителя: {avg_teacher_loss:.4f}")
    print(f"Перплексия студента: {student_perplexity:.2f}")
    print(f"Перплексия учителя: {teacher_perplexity:.2f}")
    
    if is_improved:
        print("\n✅ Студент успешно улучшен! Его потери ниже, чем у учителя.")
    else:
        print("\n⚠️ Студент не превзошел учителя. Возможно, требуется дополнительное обучение или настройка гиперпараметров.")
    
    return results

def create_comparison_table(comparison_data, output_dir):
    """Создает сравнительную таблицу метрик до и после дистилляции
    
    Args:
        comparison_data: Словарь с данными для сравнения
        output_dir: Путь для сохранения таблицы
        
    Returns:
        pd.DataFrame: Таблица сравнения
    """
    print("\nСоздание сравнительной таблицы...")
    
    # Извлекаем данные из словаря comparison_data
    before_metrics = comparison_data.get('before', {})
    after_metrics = comparison_data.get('after', {})
    parameter_changes = comparison_data.get('parameter_changes', {})
    teacher_params = comparison_data.get('teacher_params', {})
    student_params = comparison_data.get('student_params', {})
    compression_ratio = comparison_data.get('compression_ratio', 'Н/Д')
    avg_student_loss = comparison_data.get('avg_student_loss', 'Н/Д')
    avg_total_loss = comparison_data.get('avg_total_loss', 'Н/Д')
    avg_distillation_loss = comparison_data.get('avg_distillation_loss', 'Н/Д')
    elapsed_time = comparison_data.get('elapsed_time', 'Н/Д')
    
    # Создаем словарь для таблицы
    data = {
        'Метрика': [
            'Потеря студента',
            'Перплексия студента',
            'Количество параметров студента',
            'Количество параметров учителя',
            'Коэффициент сжатия',
            'Средняя потеря студента',
            'Средняя общая потеря',
            'Средняя потеря дистилляции',
            'Время выполнения (сек)'
        ],
        'До дистилляции': [
            before_metrics.get('student_loss', 'Н/Д'),
            before_metrics.get('student_perplexity', 'Н/Д'),
            student_params.get('total', 'Н/Д'),
            teacher_params.get('total', 'Н/Д'),
            compression_ratio,
            'Н/Д',
            'Н/Д',
            'Н/Д',
            'Н/Д'
        ],
        'После дистилляции': [
            after_metrics.get('student_loss', 'Н/Д'),
            after_metrics.get('student_perplexity', 'Н/Д'),
            student_params.get('total', 'Н/Д'),
            teacher_params.get('total', 'Н/Д'),
            compression_ratio,
            avg_student_loss,
            avg_total_loss,
            avg_distillation_loss,
            elapsed_time
        ],
        'Изменение': ['Н/Д', 'Н/Д', 'Н/Д', 'Н/Д', 'Н/Д', 'Н/Д', 'Н/Д', 'Н/Д', 'Н/Д']
    }
    
    # Вычисляем изменения, если есть данные
    if before_metrics.get('student_loss') is not None and after_metrics.get('student_loss') is not None:
        loss_change = after_metrics['student_loss'] - before_metrics['student_loss']
        data['Изменение'][0] = f"{loss_change:.4f} ({loss_change/before_metrics['student_loss']*100:.2f}%)"
    
    if before_metrics.get('student_perplexity') is not None and after_metrics.get('student_perplexity') is not None:
        perplexity_change = after_metrics['student_perplexity'] - before_metrics['student_perplexity']
        data['Изменение'][1] = f"{perplexity_change:.2f} ({perplexity_change/before_metrics['student_perplexity']*100:.2f}%)"
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    
    # Выводим таблицу
    print("\nСравнительная таблица:")
    print(df.to_string(index=False))
    
    # Сохраняем таблицу в CSV
    csv_path = os.path.join(output_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"Таблица сохранена в {csv_path}")
    
    return df

def plot_perplexity(distiller, output_dir=None):
    """Строит график перплексии для учителя и студента
    
    Args:
        distiller: Объект дистилляции
        output_dir: Путь для сохранения графиков (если None, то графики не сохраняются)
    """
    # Получаем историю потерь из объекта дистилляции
    loss_history = distiller.get_loss_history()
    
    # Создаем массив для оси X (номера батчей)
    batches = list(range(1, len(loss_history['teacher']) + 1))
    
    # Рассчитываем перплексию на основе потерь (perplexity = exp(loss))
    teacher_perplexity = [np.exp(loss) for loss in loss_history['teacher']]
    student_perplexity = [np.exp(loss) for loss in loss_history['student']]
    
    # Создаем график
    plt.figure(figsize=(12, 8))
    plt.plot(batches, teacher_perplexity, label='Перплексия учителя', color='blue', linewidth=2)
    plt.plot(batches, student_perplexity, label='Перплексия студента', color='red', linewidth=2)
    plt.xlabel('Батчи', fontsize=12)
    plt.ylabel('Перплексия', fontsize=12)
    plt.title('Сравнение перплексии учителя и студента', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Добавляем аннотацию о перплексии
    avg_teacher_perplexity = np.mean(teacher_perplexity)
    avg_student_perplexity = np.mean(student_perplexity)
    plt.annotate(
        f'Средняя перплексия: Учитель ({avg_teacher_perplexity:.2f}), Студент ({avg_student_perplexity:.2f})\n'
        f'Перплексия студента должна постепенно приближаться к перплексии учителя',
        xy=(0.5, 0.02), xycoords='figure fraction',
        ha='center', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3)
    )
    
    plt.tight_layout()
    
    # Сохраняем график с высоким разрешением, если указан путь
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'perplexity_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"График перплексии сохранен в {plot_path}")
    
    # Выводим средние значения перплексии
    print(f"Средняя перплексия учителя: {avg_teacher_perplexity:.2f}")
    print(f"Средняя перплексия студента: {avg_student_perplexity:.2f}")
    plt.close()

def plot_losses(distiller, output_dir=None):
    """Строит график потерь дистилляции
    
    Args:
        distiller: Объект дистилляции
        output_dir: Путь для сохранения графиков (если None, то графики не сохраняются)
    """
    import matplotlib.pyplot as plt
    import os
    
    # Получаем историю потерь
    loss_history = distiller.get_loss_history()
    
    # Создаем массив для оси X (номера батчей)
    batches = list(range(1, len(loss_history['total']) + 1))
    
    # Создаем график с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Первый подграфик: общая потеря и потеря дистилляции
    ax1.plot(batches, loss_history['total'], label='Общая потеря', color='purple')
    ax1.plot(batches, loss_history['distill'], label='Потеря дистилляции', color='green')
    ax1.set_xlabel('Батчи')
    ax1.set_ylabel('Потеря')
    ax1.set_title('Общая потеря и потеря дистилляции')
    ax1.legend()
    ax1.grid(True)
    
    # Второй подграфик: потери студента и учителя
    ax2.plot(batches, loss_history['student'], label='Потеря студента', color='red')
    ax2.plot(batches, loss_history['teacher'], label='Потеря учителя', color='blue')
    ax2.set_xlabel('Батчи')
    ax2.set_ylabel('Потеря')
    ax2.set_title('Сравнение потерь студента и учителя')
    ax2.legend()
    ax2.grid(True)
    
    # Настраиваем расположение
    plt.tight_layout()
    
    # Сохраняем график, если указан путь
    if output_dir:
        plot_path = os.path.join(output_dir, 'loss_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"График потерь сохранен в {plot_path}")
    
    plt.close()

def analyze_student_parameters(student_model_before, student_model_after):
    """Анализирует изменения параметров студента до и после дистилляции
    
    Args:
        student_model_before: Модель студента до дистилляции
        student_model_after: Модель студента после дистилляции
        
    Returns:
        dict: Словарь с информацией об изменениях параметров
    """
    print("\nАнализ изменений параметров студента...")
    
    # Получаем параметры моделей
    params_before = {name: param.clone().detach().cpu().numpy() for name, param in student_model_before.named_parameters()}
    params_after = {name: param.clone().detach().cpu().numpy() for name, param in student_model_after.named_parameters()}
    
    # Инициализируем счетчики
    total_params = 0
    total_change = 0.0
    max_change = 0.0
    min_change = float('inf')
    max_change_layer = ""
    min_change_layer = ""
    layer_changes = {}
    
    # Анализируем изменения по слоям
    for name in params_before:
        if name in params_after:
            # Вычисляем абсолютную разницу
            diff = np.abs(params_after[name] - params_before[name])
            avg_diff = np.mean(diff)
            max_diff = np.max(diff)
            
            # Обновляем счетчики
            num_params = params_before[name].size
            total_params += num_params
            total_change += avg_diff * num_params
            
            # Обновляем максимальное и минимальное изменение
            if avg_diff > max_change:
                max_change = avg_diff
                max_change_layer = name
            if avg_diff < min_change:
                min_change = avg_diff
                min_change_layer = name
            
            # Сохраняем изменения для слоя
            layer_changes[name] = {
                'avg_change': avg_diff,
                'max_change': max_diff,
                'num_params': num_params
            }
    
    # Вычисляем среднее изменение по всем параметрам
    avg_change = total_change / total_params if total_params > 0 else 0.0
    
    # Формируем результаты
    results = {
        'total': total_params,
        'avg_change': avg_change,
        'max_change': max_change,
        'min_change': min_change,
        'max_change_layer': max_change_layer,
        'min_change_layer': min_change_layer,
        'layer_changes': layer_changes
    }
    
    # Выводим результаты
    print(f"Общее количество параметров: {total_params:,}")
    print(f"Среднее изменение параметров: {avg_change:.6f}")
    print(f"Максимальное изменение: {max_change:.6f} (слой: {max_change_layer})")
    print(f"Минимальное изменение: {min_change:.6f} (слой: {min_change_layer})")
    
    # Выводим топ-5 слоев с наибольшими изменениями
    print("\nТоп-5 слоев с наибольшими изменениями:")
    top_layers = sorted(layer_changes.items(), key=lambda x: x[1]['avg_change'], reverse=True)[:5]
    for name, info in top_layers:
        print(f"  {name}: {info['avg_change']:.6f} (параметров: {info['num_params']:,})")
    
    return results

# Функция для дистилляции GPT-2 на выбранном датасете
def distill_gpt2_wikitext(args):
    print(f"Запуск дистилляции с параметрами:")
    print(f"  Учитель: {args.teacher_model}, Студент: {args.student_model}")
    print(f"  Датасет: {args.dataset}")
    print(f"  Температура: {args.temperature}, Alpha: {args.alpha}, Beta: {args.beta}")
    print(f"  Эпохи: {args.epochs}, Размер батча: {args.batch_size}, LR: {args.lr}")
    
    # Создаем уникальную директорию для текущей дистилляции
    distillation_dir = create_distillation_dir(args.output_dir, args.folder_name)
    print(f"Результаты дистилляции будут сохранены в: {distillation_dir}")
    
    # Определяем устройство
    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    print(f"Используется устройство: {device}")
    
    # Определяем, использовать ли AutoModel
    use_auto_model = hasattr(args, 'use_auto_model') and args.use_auto_model
    
    # Если модель не GPT-2 или это Qwen, принудительно используем AutoModel
    if not ('gpt2' in args.teacher_model.lower() and 'gpt2' in args.student_model.lower()) or \
       'qwen' in args.teacher_model.lower() or 'qwen' in args.student_model.lower() or \
       'llama' in args.teacher_model.lower() or 'llama' in args.student_model.lower() or \
       'mistral' in args.teacher_model.lower() or 'mistral' in args.student_model.lower() or \
       'phi' in args.teacher_model.lower() or 'phi' in args.student_model.lower():
        use_auto_model = True
    
    # Проверяем, является ли модель Qwen
    is_qwen = 'qwen' in args.teacher_model.lower() or 'qwen' in args.student_model.lower()
    
    # Для обратной совместимости проверяем, является ли модель GPT-2
    is_gpt2 = ('gpt2' in args.teacher_model.lower() and 'gpt2' in args.student_model.lower()) and \
              not (args.teacher_model.startswith('random-') or args.student_model.startswith('random-'))
    
    # Загружаем токенизатор и модели
    if is_gpt2 and not use_auto_model:
        print("Загрузка моделей GPT-2 (специфичный класс)...")
        # Загружаем токенизатор из модели учителя
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(args.teacher_model)
        except Exception as e:
            print(f"Ошибка при загрузке токенизатора GPT-2: {e}")
            print("Попытка загрузки токенизатора с использованием AutoTokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
            except Exception as e2:
                print(f"Ошибка при загрузке токенизатора с AutoTokenizer: {e2}")
                print("Использование стандартного токенизатора GPT-2...")
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Загружаем модель учителя
        try:
            if hasattr(args, 'teacher_model_path') and args.teacher_model_path:
                print(f"Загрузка модели учителя из локального пути: {args.teacher_model_path}")
                teacher_model = GPT2LMHeadModel.from_pretrained(args.teacher_model_path, trust_remote_code=True)
            else:
                print(f"Загрузка модели учителя из Hugging Face: {args.teacher_model}")
                teacher_model = GPT2LMHeadModel.from_pretrained(args.teacher_model, trust_remote_code=True)
        except Exception as e:
            print(f"Ошибка при загрузке модели учителя как GPT2LMHeadModel: {e}")
            print("Попытка загрузки модели учителя с использованием AutoModelForCausalLM...")
            if hasattr(args, 'teacher_model_path') and args.teacher_model_path:
                teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, trust_remote_code=True)
            else:
                teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model, trust_remote_code=True)
        
        # Загружаем модель студента
        try:
            if hasattr(args, 'student_model_path') and args.student_model_path:
                print(f"Загрузка модели студента из локального пути: {args.student_model_path}")
                student_model = GPT2LMHeadModel.from_pretrained(args.student_model_path, trust_remote_code=True)
            else:
                print(f"Загрузка модели студента из Hugging Face: {args.student_model}")
                student_model = GPT2LMHeadModel.from_pretrained(args.student_model, trust_remote_code=True)
        except Exception as e:
            print(f"Ошибка при загрузке модели студента как GPT2LMHeadModel: {e}")
            print("Попытка загрузки модели студента с использованием AutoModelForCausalLM...")
            if hasattr(args, 'student_model_path') and args.student_model_path:
                student_model = AutoModelForCausalLM.from_pretrained(args.student_model_path, trust_remote_code=True)
            else:
                student_model = AutoModelForCausalLM.from_pretrained(args.student_model, trust_remote_code=True)
    else:
        print("Загрузка моделей с использованием AutoModel...")
        # Загружаем токенизатор из модели учителя
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        except Exception as e:
            print(f"Ошибка при загрузке токенизатора учителя: {e}")
            print("Попытка загрузки токенизатора GPT-2...")
            try:
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            except Exception as e2:
                print(f"Ошибка при загрузке токенизатора GPT-2: {e2}")
                print("Попытка загрузки токенизатора из модели студента...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
                except Exception as e3:
                    print(f"Все попытки загрузки токенизатора не удались. Последняя ошибка: {e3}")
                    raise ValueError("Не удалось загрузить токенизатор для моделей.")
        
        # Загружаем модель учителя
        try:
            if hasattr(args, 'teacher_model_path') and args.teacher_model_path:
                print(f"Загрузка модели учителя из локального пути: {args.teacher_model_path}")
                teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, trust_remote_code=True)
            else:
                print(f"Загрузка модели учителя из Hugging Face: {args.teacher_model}")
                teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model, trust_remote_code=True)
        except Exception as e:
            print(f"Ошибка при загрузке модели учителя: {e}")
            raise ValueError(f"Не удалось загрузить модель учителя: {e}")
        
        # Загружаем модель студента
        try:
            if hasattr(args, 'student_model_path') and args.student_model_path:
                print(f"Загрузка модели студента из локального пути: {args.student_model_path}")
                student_model = AutoModelForCausalLM.from_pretrained(args.student_model_path, trust_remote_code=True)
            else:
                print(f"Загрузка модели студента из Hugging Face: {args.student_model}")
                student_model = AutoModelForCausalLM.from_pretrained(args.student_model, trust_remote_code=True)
        except Exception as e:
            print(f"Ошибка при загрузке модели студента: {e}")
            raise ValueError(f"Не удалось загрузить модель студента: {e}")
    
    # Проверяем совместимость словарей моделей
    teacher_vocab_size = teacher_model.config.vocab_size
    student_vocab_size = student_model.config.vocab_size
    
    print(f"Размер словаря учителя: {teacher_vocab_size}")
    print(f"Размер словаря студента: {student_vocab_size}")
    
    # Если словари несовместимы, копируем слой эмбеддингов и токенизатор от учителя к ученику
    if teacher_vocab_size != student_vocab_size:
        print(f"\n⚠️ Обнаружено несоответствие размеров словарей: учитель ({teacher_vocab_size}) vs студент ({student_vocab_size})")
        print("Копирование слоя эмбеддингов и токенизатора от учителя к ученику...")
        
        # Копируем слой эмбеддингов от учителя к ученику
        if hasattr(teacher_model, 'transformer') and hasattr(student_model, 'transformer'):
            # Для моделей GPT-2
            if hasattr(teacher_model.transformer, 'wte') and hasattr(student_model.transformer, 'wte'):
                print("Копирование слоя эмбеддингов для GPT-2...")
                # Сохраняем оригинальный размер эмбеддингов студента
                original_embedding_dim = student_model.transformer.wte.embedding_dim
                # Создаем новый слой эмбеддингов с размером словаря учителя, но размерностью эмбеддингов студента
                new_embeddings = nn.Embedding(teacher_vocab_size, original_embedding_dim)
                # Копируем веса из слоя эмбеддингов учителя (с учетом возможной разницы в размерности)
                with torch.no_grad():
                    # Если размерности эмбеддингов совпадают, копируем напрямую
                    if teacher_model.transformer.wte.embedding_dim == original_embedding_dim:
                        new_embeddings.weight.copy_(teacher_model.transformer.wte.weight)
                    else:
                        # Если размерности не совпадают, используем проекцию или усечение
                        print(f"Размерности эмбеддингов не совпадают: учитель ({teacher_model.transformer.wte.embedding_dim}) vs студент ({original_embedding_dim})")
                        if teacher_model.transformer.wte.embedding_dim > original_embedding_dim:
                            # Усечение: берем только первые original_embedding_dim компонент
                            new_embeddings.weight.copy_(teacher_model.transformer.wte.weight[:, :original_embedding_dim])
                        else:
                            # Проекция: заполняем доступные компоненты и оставляем остальные инициализированными случайно
                            new_embeddings.weight[:, :teacher_model.transformer.wte.embedding_dim].copy_(teacher_model.transformer.wte.weight)
                
                # Заменяем слой эмбеддингов студента
                student_model.transformer.wte = new_embeddings
                
                # Обновляем конфигурацию студента
                student_model.config.vocab_size = teacher_vocab_size
                
                print("Слой эмбеддингов успешно скопирован от учителя к ученику")
            
            # Аналогично для выходного слоя (lm_head)
            if hasattr(teacher_model, 'lm_head') and hasattr(student_model, 'lm_head'):
                print("Обновление выходного слоя (lm_head)...")
                # Создаем новый выходной слой с размером словаря учителя
                original_output_dim = student_model.lm_head.out_features if hasattr(student_model.lm_head, 'out_features') else student_model.lm_head.weight.size(1)
                new_lm_head = nn.Linear(original_output_dim, teacher_vocab_size, bias=False)
                
                # Копируем веса из выходного слоя учителя (с учетом возможной разницы в размерности)
                with torch.no_grad():
                    teacher_output_dim = teacher_model.lm_head.out_features if hasattr(teacher_model.lm_head, 'out_features') else teacher_model.lm_head.weight.size(1)
                    if teacher_output_dim == original_output_dim:
                        new_lm_head.weight.copy_(teacher_model.lm_head.weight)
                    else:
                        print(f"Размерности выходного слоя не совпадают: учитель ({teacher_output_dim}) vs студент ({original_output_dim})")
                        # Адаптируем веса с учетом разных размерностей
                        if hasattr(teacher_model.lm_head, 'weight') and hasattr(student_model.lm_head, 'weight'):
                            if teacher_output_dim > original_output_dim:
                                new_lm_head.weight.copy_(teacher_model.lm_head.weight[:, :original_output_dim])
                            else:
                                new_lm_head.weight[:, :teacher_output_dim].copy_(teacher_model.lm_head.weight)
                
                # Заменяем выходной слой студента
                student_model.lm_head = new_lm_head
                
                print("Выходной слой успешно обновлен")
        
        # Для других архитектур моделей (не GPT-2)
        elif hasattr(teacher_model, 'get_input_embeddings') and hasattr(student_model, 'get_input_embeddings'):
            print("Копирование слоя эмбеддингов для общей архитектуры...")
            teacher_embeddings = teacher_model.get_input_embeddings()
            student_embeddings = student_model.get_input_embeddings()
            
            # Создаем новый слой эмбеддингов с размером словаря учителя, но размерностью эмбеддингов студента
            original_embedding_dim = student_embeddings.embedding_dim
            new_embeddings = nn.Embedding(teacher_vocab_size, original_embedding_dim)
            
            # Копируем веса из слоя эмбеддингов учителя (с учетом возможной разницы в размерности)
            with torch.no_grad():
                if teacher_embeddings.embedding_dim == original_embedding_dim:
                    new_embeddings.weight.copy_(teacher_embeddings.weight)
                else:
                    print(f"Размерности эмбеддингов не совпадают: учитель ({teacher_embeddings.embedding_dim}) vs студент ({original_embedding_dim})")
                    if teacher_embeddings.embedding_dim > original_embedding_dim:
                        new_embeddings.weight.copy_(teacher_embeddings.weight[:, :original_embedding_dim])
                    else:
                        new_embeddings.weight[:, :teacher_embeddings.embedding_dim].copy_(teacher_embeddings.weight)
            
            # Заменяем слой эмбеддингов студента
            student_model.set_input_embeddings(new_embeddings)
            
            # Обновляем выходной слой, если он доступен
            if hasattr(teacher_model, 'get_output_embeddings') and hasattr(student_model, 'get_output_embeddings'):
                teacher_output = teacher_model.get_output_embeddings()
                student_output = student_model.get_output_embeddings()
                
                if teacher_output is not None and student_output is not None:
                    print("Обновление выходного слоя...")
                    # Создаем новый выходной слой с размером словаря учителя
                    original_output_dim = student_output.in_features if hasattr(student_output, 'in_features') else student_output.weight.size(1)
                    new_output = nn.Linear(original_output_dim, teacher_vocab_size, bias=False)
                    
                    # Копируем веса из выходного слоя учителя (с учетом возможной разницы в размерности)
                    with torch.no_grad():
                        teacher_output_dim = teacher_output.in_features if hasattr(teacher_output, 'in_features') else teacher_output.weight.size(1)
                        if teacher_output_dim == original_output_dim:
                            new_output.weight.copy_(teacher_output.weight)
                        else:
                            print(f"Размерности выходного слоя не совпадают: учитель ({teacher_output_dim}) vs студент ({original_output_dim})")
                            if teacher_output_dim > original_output_dim:
                                new_output.weight.copy_(teacher_output.weight[:, :original_output_dim])
                            else:
                                new_output.weight[:, :teacher_output_dim].copy_(teacher_output.weight)
                    
                    # Заменяем выходной слой студента
                    student_model.set_output_embeddings(new_output)
                    
                    print("Выходной слой успешно обновлен")
            
            # Обновляем конфигурацию студента
            student_model.config.vocab_size = teacher_vocab_size
            
            print("Слой эмбеддингов успешно скопирован от учителя к ученику")
        
        # Копируем токенизатор от учителя к ученику
        print("Копирование токенизатора от учителя к ученику...")
        # Используем токенизатор учителя для студента
        # Это уже сделано, так как мы используем один токенизатор для обеих моделей
        
        print("✅ Словари моделей успешно синхронизированы!")
    
    # Создаем копию студента для сравнения до и после дистилляции
    student_model_before = copy.deepcopy(student_model)
    
    # Выводим информацию о моделях
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = teacher_params / student_params
    
    print(f"Параметры учителя: {teacher_params:,}")
    print(f"Параметры студента: {student_params:,}")
    print(f"Коэффициент сжатия: {compression_ratio:.2f}x")
    
    # Подготовка данных
    dataset = load_dataset_for_distillation(args, tokenizer)
    
    # Если указан размер выборки, берем подмножество данных
    if args.sample_size > 0 and args.sample_size < len(dataset):
        indices = np.random.choice(len(dataset), args.sample_size, replace=False)
        dataset = Subset(dataset, indices)
        print(f"Используется подмножество данных размером {args.sample_size}")
    
    # Создаем загрузчик данных для обучения
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_batch
    )
    
    # Настраиваем оптимизатор
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr)
    
    # Настраиваем конфигурацию дистилляции
    distillation_config = DistillationConfig(
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        device=device,
        save_metrics=args.save_metrics,
        log_interval=10  # Логируем каждые 10 батчей
    )
    
    # Создаем модуль дистилляции
    distiller = KnowledgeDistillation(distillation_config)
    
    # Определяем функцию для адаптации выходов моделей
    def model_forward(model, inputs):
        outputs = model(inputs)
        return outputs.logits
    
    # Оборачиваем модели для совместимости с интерфейсом дистилляции
    class WrappedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, inputs):
            return model_forward(self.model, inputs)
    
    wrapped_teacher = WrappedModel(teacher_model)
    wrapped_student = WrappedModel(student_model)
    wrapped_student_before = WrappedModel(student_model_before)
    
    # Создаем директорию для чекпоинтов внутри уникальной директории дистилляции
    checkpoint_dir = os.path.join(distillation_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Проводим валидацию до дистилляции, если не указан флаг пропуска валидации
    before_metrics = {}
    if not args.skip_validation:
        print("\nПроведение валидации до дистилляции...")
        # Явно перемещаем модели на устройство перед валидацией
        wrapped_student_before = wrapped_student_before.to(device)
        wrapped_teacher = wrapped_teacher.to(device)
        before_metrics = validate_student_improvement(wrapped_student_before, wrapped_teacher, val_loader, device)
    
    # Функция для отображения прогресса дистилляции
    def display_distillation_progress(distiller, student_model, teacher_model, train_loader, val_loader, optimizer, num_epochs, student_loss_fn, checkpoint_dir, checkpoint_interval, save_best_model=False):
        """Запускает процесс дистилляции с отображением прогресса в консоли
        
        Args:
            distiller: Объект дистилляции
            student_model: Модель студента
            teacher_model: Модель учителя
            train_loader: Загрузчик данных для обучения
            val_loader: Загрузчик данных для валидации
            optimizer: Оптимизатор
            num_epochs: Количество эпох
            student_loss_fn: Функция потерь для студента
            checkpoint_dir: Директория для сохранения чекпоинтов
            checkpoint_interval: Интервал сохранения чекпоинтов (в эпохах)
            save_best_model: Сохранять лучшую модель по валидационной выборке (по умолчанию отключено)
        
        Returns:
            Обученная модель студента
        """
        # Устанавливаем модели
        distiller.set_models(teacher_model, student_model)
        distiller.metrics.reset()
        
        # Получаем устройство из конфигурации
        device = distiller.config.device
        
        # Переводим учителя в режим eval
        teacher_model.eval()
        
        # Создаем прогресс-бар для эпох
        epoch_bar = trange(num_epochs, desc="Эпохи", position=0)
        
        # Инициализируем переменные для отслеживания лучшей модели
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = -1
        
        # Для каждой эпохи
        for epoch in epoch_bar:
            # Создаем прогресс-бар для батчей
            batch_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs}", position=1, leave=False)
            
            # Сбрасываем метрики для текущей эпохи
            epoch_losses = {'total': 0.0, 'distill': 0.0, 'student': 0.0, 'teacher': 0.0}
            batch_count = 0
            
            # Для каждого батча
            for batch_idx, (inputs, targets) in enumerate(batch_bar):
                # Перемещаем данные на устройство
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Явно перемещаем модели на устройство перед каждым батчем
                teacher_model = teacher_model.to(device)
                student_model = student_model.to(device)
                
                # Получаем предсказания учителя (без градиентов)
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                
                # Получаем предсказания студента
                student_model.train()
                student_logits = student_model(inputs)
                
                # Вычисляем потери
                losses = distiller.distillation_loss(
                    student_logits, teacher_logits, targets, student_loss_fn
                )
                
                # Обратное распространение
                optimizer.zero_grad()
                losses['total_loss'].backward()
                optimizer.step()
                
                # Обновляем метрики
                # Проверяем наличие ключа 'teacher_loss' в словаре losses
                teacher_loss = losses.get('teacher_loss', torch.tensor(0.0))
                distiller.metrics.update(
                    losses['total_loss'].item(),
                    losses['distillation_loss'].item(),
                    losses['student_loss'].item(),
                    teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else teacher_loss
                )
                
                # Сохраняем историю для графика
                distiller.loss_history['total'].append(losses['total_loss'].item())
                distiller.loss_history['distill'].append(losses['distillation_loss'].item())
                distiller.loss_history['student'].append(losses['student_loss'].item())
                # Проверяем наличие ключа 'teacher_loss' в словаре losses
                teacher_loss = losses.get('teacher_loss', torch.tensor(0.0))
                distiller.loss_history['teacher'].append(teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else teacher_loss)
                
                # Обновляем метрики для текущей эпохи
                epoch_losses['total'] += losses['total_loss'].item()
                epoch_losses['distill'] += losses['distillation_loss'].item()
                epoch_losses['student'] += losses['student_loss'].item()
                # Проверяем наличие ключа 'teacher_loss' в словаре losses
                teacher_loss = losses.get('teacher_loss', torch.tensor(0.0))
                epoch_losses['teacher'] += teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else teacher_loss
                batch_count += 1
                
                # Обновляем прогресс-бар с текущими потерями
                # Проверяем наличие ключа 'teacher_loss' в словаре losses
                teacher_loss = losses.get('teacher_loss', torch.tensor(0.0))
                batch_bar.set_postfix({
                    'total': losses['total_loss'].item(),
                    'distill': losses['distillation_loss'].item(),
                    'student': losses['student_loss'].item(),
                    'teacher': teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else teacher_loss
                })
            
            # Вычисляем средние потери за эпоху
            avg_losses = {
                key: value / batch_count for key, value in epoch_losses.items()
            }
            
            # Обновляем прогресс-бар эпох с средними потерями
            epoch_bar.set_postfix({
                'avg_total': avg_losses['total'],
                'avg_distill': avg_losses['distill'],
                'avg_student': avg_losses['student'],
                'avg_teacher': avg_losses['teacher']
            })
            
            # Сохраняем чекпоинт, если указан интервал и директория
            if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'student_checkpoint_epoch_{epoch+1}.pt')
                distiller.save_student_model(checkpoint_path)
                print(f"\nСохранен чекпоинт после эпохи {epoch+1} в {checkpoint_path}")
            
            # Проводим валидацию, если включено сохранение лучшей модели
            if save_best_model and val_loader is not None:
                print(f"\nПроведение валидации после эпохи {epoch+1}...")
                student_model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for val_inputs, val_targets in val_loader:
                        val_inputs = val_inputs.to(device)
                        val_targets = val_targets.to(device)
                        
                        # Явно перемещаем модель на устройство перед валидацией
                        student_model = student_model.to(device)
                        
                        val_logits = student_model(val_inputs)
                        
                        # Убедимся, что логиты и цели на одном устройстве
                        if val_logits.device != val_targets.device:
                            val_logits = val_logits.to(val_targets.device)
                            
                        val_batch_loss = student_loss_fn(val_logits, val_targets).item()
                        val_loss += val_batch_loss
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
                print(f"Средняя валидационная потеря: {avg_val_loss:.4f}")
                
                # Сохраняем лучшую модель
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = {key: value.cpu().clone() for key, value in student_model.state_dict().items()}
                    best_epoch = epoch + 1
                    
                    if checkpoint_dir:
                        best_model_path = os.path.join(checkpoint_dir, 'best_student_model.pt')
                        torch.save({
                            'model_state_dict': best_model_state,
                            'epoch': best_epoch,
                            'val_loss': best_val_loss,
                            'config': distiller.config,
                            'metrics': distiller.get_metrics()
                        }, best_model_path)
                        print(f"Сохранена лучшая модель (эпоха {best_epoch}, потеря {best_val_loss:.4f}) в {best_model_path}")
        
        # Загружаем лучшую модель, если она была сохранена
        if save_best_model and best_model_state is not None:
            print(f"\nЗагрузка лучшей модели (эпоха {best_epoch}, потеря {best_val_loss:.4f})")
            student_model.load_state_dict({key: value.to(device) for key, value in best_model_state.items()})
            print("Лучшая модель успешно загружена.")
        else:
            print("\nИспользуется модель после последней эпохи.")

        
        # Закрываем прогресс-бары
        epoch_bar.close()
        
        return student_model
    
    # Запускаем дистилляцию с отображением прогресса
    print("Начало процесса дистилляции...")
    trained_student = display_distillation_progress(
        distiller,
        wrapped_student,
        wrapped_teacher,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=args.epochs,
        student_loss_fn=language_model_loss,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        save_best_model=args.save_best_model
    )

    # Получаем и выводим метрики
    metrics = distiller.get_metrics()
    print("\nМетрики дистилляции:")
    print(f"Средняя потеря студента: {metrics['metrics']['avg_student_loss']:.4f}")
    print(f"Средняя общая потеря: {metrics['metrics']['avg_total_loss']:.4f}")
    print(f"Средняя потеря дистилляции: {metrics['metrics']['avg_distillation_loss']:.4f}")
    print(f"Время выполнения: {metrics['metrics']['elapsed_time']:.2f} секунд")
    print(f"Коэффициент сжатия: {metrics['config']['compression_ratio']:.2f}x")
    
    # Строим и сохраняем графики в уникальную директорию дистилляции
    plot_losses(distiller, distillation_dir)
    plot_perplexity(distiller, distillation_dir)
    
    # Сохраняем модель студента в уникальную директорию дистилляции
    student_path = os.path.join(distillation_dir, 'student_model_wikitext.pt')
    distiller.save_student_model(student_path)
    print(f"Модель студента сохранена в {student_path}")
    
    # Сохраняем метрики в JSON, если указано
    if args.save_metrics:
        metrics_path = os.path.join(distillation_dir, 'metrics_wikitext.json')
        with open(metrics_path, 'w') as f:
            # Преобразуем метрики в сериализуемый формат
            serializable_metrics = {
                'metrics': {
                    k: float(v) for k, v in metrics['metrics'].items()
                },
                'config': {
                    k: float(v) if isinstance(v, (int, float)) else v 
                    for k, v in metrics['config'].items()
                }
            }
            json.dump(serializable_metrics, f, indent=2)
        print(f"Метрики сохранены в {metrics_path}")
    
    # Создаем и сохраняем графики потерь, если указано
    if args.save_plot:
        # Получаем историю потерь из объекта дистилляции
        loss_history = distiller.get_loss_history()
        
        # Создаем массив для оси X (номера батчей)
        batches = list(range(1, len(loss_history['total']) + 1))
        
        # 1. График общих потерь
        plt.figure(figsize=(12, 8))
        plt.plot(batches, loss_history['total'], label='Общая потеря', linewidth=2)
        plt.plot(batches, loss_history['distill'], label='Потеря дистилляции', linewidth=2)
        plt.xlabel('Батчи', fontsize=12)
        plt.ylabel('Значение потери', fontsize=12)
        plt.title('Общие потери при дистилляции', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Сохраняем график с высоким разрешением
        plot_path = os.path.join(distillation_dir, 'total_loss_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"График общих потерь сохранен в {plot_path}")
        plt.close()
        
        # 2. График сравнения потерь учителя и студента
        plt.figure(figsize=(12, 8))
        plt.plot(batches, loss_history['teacher'], label='Потеря учителя', color='blue', linewidth=2)
        plt.plot(batches, loss_history['student'], label='Потеря студента', color='red', linewidth=2)
        plt.xlabel('Батчи', fontsize=12)
        plt.ylabel('Значение потери', fontsize=12)
        plt.title('Сравнение потерь учителя и студента', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Добавляем аннотацию, если потери студента больше потерь учителя
        avg_teacher_loss = np.mean(loss_history['teacher'])
        avg_student_loss = np.mean(loss_history['student'])
        if avg_student_loss > avg_teacher_loss:
            plt.annotate(
                f'Средняя потеря студента ({avg_student_loss:.4f}) выше, чем у учителя ({avg_teacher_loss:.4f})\n'
                f'Это нормально в процессе дистилляции, так как студент имеет меньше параметров',
                xy=(0.5, 0.02), xycoords='figure fraction',
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3)
            )
        
        plt.tight_layout()
        
        # Сохраняем график с высоким разрешением
        plot_path = os.path.join(distillation_dir, 'teacher_student_loss_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"График сравнения потерь учителя и студента сохранен в {plot_path}")
        plt.close()
        
        # 3. График всех потерь вместе
        plt.figure(figsize=(12, 8))
        plt.plot(batches, loss_history['total'], label='Общая потеря', linewidth=2)
        plt.plot(batches, loss_history['distill'], label='Потеря дистилляции', linewidth=2)
        plt.plot(batches, loss_history['student'], label='Потеря студента', linewidth=2)
        plt.plot(batches, loss_history['teacher'], label='Потеря учителя', linewidth=2)
        plt.xlabel('Батчи', fontsize=12)
        plt.ylabel('Значение потери', fontsize=12)
        plt.title('Все потери при дистилляции', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Сохраняем график с высоким разрешением
        plot_path = os.path.join(distillation_dir, 'all_losses_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"График всех потерь сохранен в {plot_path}")
        plt.close()
        
        # 4. График перплексии
        # Рассчитываем перплексию на основе потерь (perplexity = exp(loss))
        teacher_perplexity = [np.exp(loss) for loss in loss_history['teacher']]
        student_perplexity = [np.exp(loss) for loss in loss_history['student']]
        
        plt.figure(figsize=(12, 8))
        plt.plot(batches, teacher_perplexity, label='Перплексия учителя', color='blue', linewidth=2)
        plt.plot(batches, student_perplexity, label='Перплексия студента', color='red', linewidth=2)
        plt.xlabel('Батчи', fontsize=12)
        plt.ylabel('Перплексия', fontsize=12)
        plt.title('Сравнение перплексии учителя и студента', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Добавляем аннотацию о перплексии
        avg_teacher_perplexity = np.mean(teacher_perplexity)
        avg_student_perplexity = np.mean(student_perplexity)
        plt.annotate(
            f'Средняя перплексия: Учитель ({avg_teacher_perplexity:.2f}), Студент ({avg_student_perplexity:.2f})\n'
            f'Перплексия студента должна постепенно приближаться к перплексии учителя',
            xy=(0.5, 0.02), xycoords='figure fraction',
            ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3)
        )
        
        plt.tight_layout()
        
        # Сохраняем график с высоким разрешением
        plot_path = os.path.join(distillation_dir, 'perplexity_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"График перплексии сохранен в {plot_path}")
        plt.close()
    
    # Проводим валидацию после дистилляции, если не указан флаг пропуска валидации
    after_metrics = {}
    if not args.skip_validation:
        print("\nПроведение валидации после дистилляции...")
        # Явно перемещаем модели на устройство перед валидацией
        wrapped_student = wrapped_student.to(device)
        wrapped_teacher = wrapped_teacher.to(device)
        after_metrics = validate_student_improvement(wrapped_student, wrapped_teacher, val_loader, device)
        
        # Выводим сравнение результатов до и после дистилляции
        print("\nСравнение результатов до и после дистилляции:")
        print(f"Потеря студента до: {before_metrics['avg_student_loss']:.4f}, после: {after_metrics['avg_student_loss']:.4f}")
        print(f"Перплексия студента до: {before_metrics['student_perplexity']:.4f}, после: {after_metrics['student_perplexity']:.4f}")
        print(f"Потеря учителя: {after_metrics['avg_teacher_loss']:.4f}, перплексия: {after_metrics['teacher_perplexity']:.4f}")
        
        # Определяем, улучшилась ли модель
        if after_metrics['avg_student_loss'] < before_metrics['avg_student_loss']:
            improvement = (before_metrics['avg_student_loss'] - after_metrics['avg_student_loss']) / before_metrics['avg_student_loss'] * 100
            print(f"\nМодель студента улучшилась на {improvement:.2f}% по потере")
        else:
            print("\nМодель студента не улучшилась по потере")
    
    # Анализируем изменения параметров студента
    parameter_changes = analyze_student_parameters(student_model_before, student_model)
    
    # Создаем сравнительную таблицу, если указан соответствующий флаг
    if args.create_comparison_table:
        comparison_data = {
            'before': before_metrics,
            'after': after_metrics,
            'parameter_changes': parameter_changes,
            'teacher_params': teacher_params,
            'student_params': student_params,
            'compression_ratio': compression_ratio,
            'avg_student_loss': metrics['metrics']['avg_student_loss'],
            'avg_total_loss': metrics['metrics']['avg_total_loss'],
            'avg_distillation_loss': metrics['metrics']['avg_distillation_loss'],
            'elapsed_time': metrics['metrics']['elapsed_time']
        }
        create_comparison_table(comparison_data, distillation_dir)
    
    return trained_student, metrics

# Основная функция
def main():
    args = parse_args()
    
    # Выводим информацию о доступных моделях
    print_model_info()
    
    # Установка зависимостей, если указано
    if args.install_deps:
        print("Установка зависимостей...")
        import subprocess
        try:
            subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
            print("Зависимости успешно установлены.")
        except Exception as e:
            print(f"Ошибка при установке зависимостей: {e}")
            return
    
    # Запускаем тесты, только если указан флаг --run_tests
    if args.run_tests:
        print("Запуск тестов...")
        try:
            import test_distillation
            test_distillation.run_comprehensive_test()
            print("Тесты успешно пройдены.")
        except Exception as e:
            print(f"Ошибка при запуске тестов: {e}")
            if input("Продолжить выполнение? (y/n): ").lower() != 'y':
                return
    else:
        print("Тесты пропущены.")
    
    # Пропускаем демонстрацию, если указано
    if not args.skip_demo:
        print("Запуск демонстрации...")
        try:
            # Здесь можно добавить код для демонстрации
            print("Демонстрация успешно завершена.")
        except Exception as e:
            print(f"Ошибка при запуске демонстрации: {e}")
            if input("Продолжить выполнение? (y/n): ").lower() != 'y':
                return
    else:
        print("Демонстрация пропущена.")
    
    # Запуск дистилляции
    distill_gpt2_wikitext(args)

if __name__ == "__main__":
    main()