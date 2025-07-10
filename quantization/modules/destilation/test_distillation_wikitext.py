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
from distillation import KnowledgeDistillation, DistillationConfig, create_optimizer

# Настройка аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description='Тестирование модуля дистилляции знаний на WikiText')
    parser.add_argument('--teacher_model', type=str, default='gpt2-medium', help='Модель учителя')
    parser.add_argument('--student_model', type=str, default='gpt2', help='Модель студента')
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

def plot_perplexity(distiller):
    """Строит график перплексии для учителя и студента"""
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
    
    # Сохраняем график с высоким разрешением
    os.makedirs('output', exist_ok=True)
    plot_path = os.path.join('output', 'perplexity_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"График перплексии сохранен в {plot_path}")
    
    # Выводим средние значения перплексии
    print(f"Средняя перплексия учителя: {avg_teacher_perplexity:.2f}")
    print(f"Средняя перплексия студента: {avg_student_perplexity:.2f}")
    plt.close()

def plot_losses(distiller):
    """Строит график потерь дистилляции"""
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
    
    # Сохраняем график
    os.makedirs('./output', exist_ok=True)
    plt.savefig('./output/loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"График потерь сохранен в ./output/loss_plot.png")

# Функция для дистилляции GPT-2 на выбранном датасете
def distill_gpt2_wikitext(args):
    print(f"Запуск дистилляции с параметрами:")
    print(f"  Учитель: {args.teacher_model}, Студент: {args.student_model}")
    print(f"  Датасет: {args.dataset}")
    print(f"  Температура: {args.temperature}, Alpha: {args.alpha}, Beta: {args.beta}")
    print(f"  Эпохи: {args.epochs}, Размер батча: {args.batch_size}, LR: {args.lr}")
    
    # Определяем устройство
    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    print(f"Используется устройство: {device}")
    
    # Проверяем, является ли модель Qwen
    is_qwen = 'qwen' in args.teacher_model.lower() or 'qwen' in args.student_model.lower()
    
    # Загружаем токенизатор и модели
    if is_qwen:
        print("Загрузка моделей Qwen...")
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model)
        student_model = AutoModelForCausalLM.from_pretrained(args.student_model)
    else:
        print("Загрузка моделей GPT-2...")
        tokenizer = GPT2Tokenizer.from_pretrained(args.teacher_model)
        teacher_model = GPT2LMHeadModel.from_pretrained(args.teacher_model)
        student_model = GPT2LMHeadModel.from_pretrained(args.student_model)
    
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
    
    # Создаем загрузчик данных
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
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
    
    # Создаем директорию для чекпоинтов
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Функция для отображения прогресса дистилляции
    def display_distillation_progress(distiller, student_model, teacher_model, train_loader, optimizer, num_epochs, student_loss_fn, checkpoint_dir, checkpoint_interval):
        """Запускает процесс дистилляции с отображением прогресса в консоли
        
        Args:
            distiller: Объект дистилляции
            student_model: Модель студента
            teacher_model: Модель учителя
            train_loader: Загрузчик данных для обучения
            optimizer: Оптимизатор
            num_epochs: Количество эпох
            student_loss_fn: Функция потерь для студента
            checkpoint_dir: Директория для сохранения чекпоинтов
            checkpoint_interval: Интервал сохранения чекпоинтов (в эпохах)
        
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
                distiller.metrics.update(
                    losses['total_loss'].item(),
                    losses['distillation_loss'].item(),
                    losses['student_loss'].item(),
                    losses['teacher_loss'].item()
                )
                
                # Сохраняем историю для графика
                distiller.loss_history['total'].append(losses['total_loss'].item())
                distiller.loss_history['distill'].append(losses['distillation_loss'].item())
                distiller.loss_history['student'].append(losses['student_loss'].item())
                distiller.loss_history['teacher'].append(losses['teacher_loss'].item())
                
                # Обновляем метрики для текущей эпохи
                epoch_losses['total'] += losses['total_loss'].item()
                epoch_losses['distill'] += losses['distillation_loss'].item()
                epoch_losses['student'] += losses['student_loss'].item()
                epoch_losses['teacher'] += losses['teacher_loss'].item()
                batch_count += 1
                
                # Обновляем прогресс-бар с текущими потерями
                batch_bar.set_postfix({
                    'total': losses['total_loss'].item(),
                    'distill': losses['distillation_loss'].item(),
                    'student': losses['student_loss'].item(),
                    'teacher': losses['teacher_loss'].item()
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
        optimizer,
        num_epochs=args.epochs,
        student_loss_fn=language_model_loss,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval
    )

    # Получаем и выводим метрики
    metrics = distiller.get_metrics()
    print("\nМетрики дистилляции:")
    print(f"Средняя потеря студента: {metrics['metrics']['avg_student_loss']:.4f}")
    print(f"Средняя общая потеря: {metrics['metrics']['avg_total_loss']:.4f}")
    print(f"Средняя потеря дистилляции: {metrics['metrics']['avg_distillation_loss']:.4f}")
    print(f"Время выполнения: {metrics['metrics']['elapsed_time']:.2f} секунд")
    print(f"Коэффициент сжатия: {metrics['config']['compression_ratio']:.2f}x")
    
    # Строим и сохраняем графики
    plot_losses(distiller)
    plot_perplexity(distiller)
    
    # Создаем директорию для вывода, если она не существует
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Сохраняем модель студента
    student_path = os.path.join(args.output_dir, 'student_model_wikitext.pt')
    distiller.save_student_model(student_path)
    print(f"Модель студента сохранена в {student_path}")
    
    # Сохраняем метрики в JSON, если указано
    if args.save_metrics:
        metrics_path = os.path.join(args.output_dir, 'metrics_wikitext.json')
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
        plot_path = os.path.join(args.output_dir, 'total_loss_plot.png')
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
        plot_path = os.path.join(args.output_dir, 'teacher_student_loss_plot.png')
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
        plot_path = os.path.join(args.output_dir, 'all_losses_plot.png')
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
        plot_path = os.path.join(args.output_dir, 'perplexity_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"График перплексии сохранен в {plot_path}")
        plt.close()
    
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