import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Union, Dict, List, Optional, Tuple
import gc
import os
import copy


def calculate_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Вычисляет размер модели в параметрах и мегабайтах.
    """
    total_params = 0
    total_size_mb = 0
    non_zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        non_zero_params += (param != 0).sum().item()
        # Размер в байтах (float32 = 4 байта, float16 = 2 байта)
        param_size = param.numel() * param.element_size()
        total_size_mb += param_size / (1024 * 1024)
    
    # Реальный размер с учетом разреженности (приблизительный)
    sparsity = 1 - (non_zero_params / total_params)
    effective_size_mb = total_size_mb * (1 - sparsity * 0.5)  # Sparse storage saves ~50% for sparse matrices
    
    return {
        'total_params': total_params,
        'non_zero_params': non_zero_params,
        'total_params_millions': total_params / 1e6,
        'size_mb': total_size_mb,
        'effective_size_mb': effective_size_mb,
        'size_gb': total_size_mb / 1024,
        'sparsity': sparsity * 100
    }


def get_prunable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Получает список слоев, которые можно прунить.
    Исключаем embeddings и последний слой для сохранения функциональности.
    """
    prunable_layers = []
    
    # Получаем все Linear слои
    all_linear_layers = [(name, module) for name, module in model.named_modules() 
                         if isinstance(module, nn.Linear)]
    
    # Исключаем критически важные слои
    excluded_keywords = ['embed', 'lm_head', 'output_proj', 'final']
    
    for name, module in all_linear_layers:
        # Проверяем, не является ли слой критически важным
        if not any(keyword in name.lower() for keyword in excluded_keywords):
            prunable_layers.append((name, module))
    
    return prunable_layers


def apply_magnitude_pruning(model: nn.Module, sparsity: float = 0.3) -> nn.Module:
    """
    Применяет magnitude-based прунинг напрямую к весам.
    """
    pruned_count = 0
    total_params = 0
    
    # Получаем слои для прунинга
    prunable_layers = get_prunable_layers(model)
    
    if not prunable_layers:
        print("Предупреждение: не найдено слоев для прунинга!")
        return model
    
    print(f"Найдено {len(prunable_layers)} слоев для прунинга")
    
    for layer_name, layer in prunable_layers:
        if hasattr(layer, 'weight'):
            weight = layer.weight.data
            
            # Вычисляем порог для прунинга
            weight_abs = torch.abs(weight)
            threshold = torch.quantile(weight_abs.flatten(), sparsity)
            
            # Создаем маску
            mask = weight_abs > threshold
            
            # Применяем маску
            layer.weight.data = weight * mask.float()
            
            # Подсчитываем статистику
            pruned = (~mask).sum().item()
            total = mask.numel()
            pruned_count += pruned
            total_params += total
            
            layer_sparsity = pruned / total * 100
            print(f"  {layer_name}: {layer_sparsity:.1f}% прунинговано")
    
    overall_sparsity = pruned_count / total_params * 100 if total_params > 0 else 0
    print(f"\nОбщая разреженность прунингованных слоев: {overall_sparsity:.1f}%")
    
    return model


def apply_structured_pruning(model: nn.Module, sparsity: float = 0.3, dim: int = 0) -> nn.Module:
    """
    Применяет структурированный прунинг (удаление целых строк/столбцов).
    """
    prunable_layers = get_prunable_layers(model)
    
    for layer_name, layer in prunable_layers:
        if hasattr(layer, 'weight'):
            weight = layer.weight.data
            
            # Вычисляем L2 норму по указанной размерности
            if dim == 0:  # Прунинг выходных нейронов
                norms = torch.norm(weight, p=2, dim=1)
            else:  # Прунинг входных нейронов
                norms = torch.norm(weight, p=2, dim=0)
            
            # Находим индексы для прунинга
            num_prune = int(norms.size(0) * sparsity)
            if num_prune > 0:
                _, indices = torch.topk(norms, num_prune, largest=False)
                
                # Обнуляем соответствующие строки или столбцы
                if dim == 0:
                    layer.weight.data[indices] = 0
                    if layer.bias is not None:
                        layer.bias.data[indices] = 0
                else:
                    layer.weight.data[:, indices] = 0
    
    return model


def apply_random_pruning(model: nn.Module, sparsity: float = 0.3) -> nn.Module:
    """
    Применяет случайный прунинг.
    """
    prunable_layers = get_prunable_layers(model)
    
    for layer_name, layer in prunable_layers:
        if hasattr(layer, 'weight'):
            weight = layer.weight.data
            
            # Создаем случайную маску
            mask = torch.rand_like(weight) > sparsity
            
            # Применяем маску
            layer.weight.data = weight * mask.float()
    
    return model


def apply_iterative_pruning(model: nn.Module, 
                          method: str = 'magnitude',
                          final_sparsity: float = 0.3,
                          iterations: int = 5) -> nn.Module:
    """
    Применяет итеративный прунинг для более плавного удаления весов.
    """
    # Вычисляем sparsity для каждой итерации
    sparsities = np.linspace(0, final_sparsity, iterations + 1)[1:]
    
    for i, sparsity in enumerate(sparsities):
        print(f"\nИтерация {i+1}/{iterations}, целевая разреженность: {sparsity*100:.1f}%")
        
        if method == 'magnitude':
            model = apply_magnitude_pruning(model, sparsity)
        elif method == 'random':
            model = apply_random_pruning(model, sparsity)
        elif method == 'structured':
            model = apply_structured_pruning(model, sparsity)
    
    return model


def calculate_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Вычисляет процент нулевых весов в модели.
    """
    total_params = 0
    zero_params = 0
    layer_sparsities = {}
    
    for name, param in model.named_parameters():
        if param.dim() > 1:  # Только веса, не bias
            layer_total = param.numel()
            layer_zeros = (param == 0).sum().item()
            total_params += layer_total
            zero_params += layer_zeros
            
            if layer_total > 0:
                layer_sparsity = layer_zeros / layer_total * 100
                if layer_sparsity > 0:
                    layer_sparsities[name] = layer_sparsity
    
    overall_sparsity = zero_params / total_params * 100 if total_params > 0 else 0
    
    return {
        'total_params': total_params,
        'zero_params': zero_params,
        'sparsity_percentage': overall_sparsity,
        'remaining_params': total_params - zero_params,
        'remaining_percentage': 100 - overall_sparsity,
        'layer_sparsities': layer_sparsities
    }


def finetune_after_pruning(model: nn.Module, 
                         tokenizer,
                         num_samples: int = 100,
                         device: str = 'cuda') -> nn.Module:
    """
    Быстрая донастройка модели после прунинга для восстановления качества.
    """
    model.eval()
    
    # Простые промпты для калибровки
    calibration_texts = [
        "The weather today is",
        "Artificial intelligence can",
        "In the future, we will",
        "The most important thing is",
        "Technology helps us to",
        "People around the world",
        "The best way to learn",
        "Science has shown that",
        "History teaches us",
        "The meaning of life"
    ]
    
    print("\nКалибровка модели после прунинга...")
    
    with torch.no_grad():
        for text in calibration_texts[:num_samples]:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            try:
                _ = model(**inputs)
            except:
                pass
    
    return model


def prune_llm_model(
    model: Union[str, nn.Module],
    pruning_config: Dict = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    return_tokenizer: bool = False
) -> Union[nn.Module, Tuple[nn.Module, any]]:
    """
    Универсальная функция для прунинга LLM моделей.
    
    Args:
        model: Путь к модели на HuggingFace или объект модели
        pruning_config: Конфигурация прунинга
        device: Устройство для вычислений
        return_tokenizer: Возвращать ли токенайзер вместе с моделью
    
    Returns:
        Прунингованная модель (и токенайзер, если return_tokenizer=True)
    """
    
    # Дефолтная конфигурация
    default_config = {
        'method': 'magnitude',  # 'magnitude', 'random', 'structured'
        'sparsity': 0.3,       # Процент весов для удаления
        'structured': False,    # Структурированный прунинг
        'dim': 0,              # Размерность для структурированного прунинга
        'iterative': False,    # Итеративный прунинг
        'iterations': 5,       # Количество итераций
        'calibrate': True      # Калибровка после прунинга
    }
    
    if pruning_config:
        default_config.update(pruning_config)
    config = default_config
    
    # Загрузка модели
    if isinstance(model, str):
        print(f"Загрузка модели {model}...")
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        
        # Загружаем в float32 для стабильного прунинга
        model = AutoModelForCausalLM.from_pretrained(
            model, 
            torch_dtype=torch.float32,  # Важно: используем float32
            device_map=device,
            trust_remote_code=True
        )
        
        # Добавляем pad token если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = None
        model = model.to(device)
    
    # Переводим модель в режим оценки
    model.eval()
    
    # Информация о модели до прунинга
    print("\n=== Информация о модели ДО прунинга ===")
    size_before = calculate_model_size(model)
    print(f"Всего параметров: {size_before['total_params_millions']:.2f}M")
    print(f"Размер модели: {size_before['size_mb']:.2f} MB ({size_before['size_gb']:.2f} GB)")
    
    # Применение прунинга
    print(f"\nПрименение {config['method']} прунинга со sparsity={config['sparsity']}...")
    
    if config['iterative']:
        model = apply_iterative_pruning(
            model,
            method=config['method'],
            final_sparsity=config['sparsity'],
            iterations=config['iterations']
        )
    else:
        if config['method'] == 'magnitude':
            model = apply_magnitude_pruning(model, config['sparsity'])
        elif config['method'] == 'random':
            model = apply_random_pruning(model, config['sparsity'])
        elif config['method'] == 'structured':
            model = apply_structured_pruning(model, config['sparsity'], config['dim'])
        else:
            raise ValueError(f"Неизвестный метод прунинга: {config['method']}")
    
    # Вычисление sparsity
    sparsity_info = calculate_sparsity(model)
    print(f"\nДостигнутая разреженность: {sparsity_info['sparsity_percentage']:.2f}%")
    print(f"Осталось активных параметров: {sparsity_info['remaining_params']:,} ({sparsity_info['remaining_percentage']:.2f}%)")
    
    # Показываем разреженность по слоям
    if sparsity_info['layer_sparsities']:
        print("\nРазреженность по слоям:")
        for layer_name, layer_sparsity in list(sparsity_info['layer_sparsities'].items())[:10]:
            if layer_sparsity > 0:
                print(f"  {layer_name}: {layer_sparsity:.1f}%")
    
    # Калибровка модели
    if config['calibrate'] and tokenizer is not None:
        model = finetune_after_pruning(model, tokenizer, device=device)
    
    # Информация после прунинга
    print("\n=== Информация о модели ПОСЛЕ прунинга ===")
    size_after = calculate_model_size(model)
    print(f"Всего параметров: {size_after['total_params_millions']:.2f}M")
    print(f"Ненулевых параметров: {size_after['non_zero_params']/1e6:.2f}M")
    print(f"Размер модели: {size_after['size_mb']:.2f} MB")
    print(f"Эффективный размер (sparse): {size_after['effective_size_mb']:.2f} MB")
    
    # Сравнение
    print("\n=== Сравнение ===")
    print(f"Достигнутая разреженность: {size_after['sparsity']:.2f}%")
    print(f"Уменьшение активных параметров: {(1 - size_after['non_zero_params']/size_before['total_params'])*100:.2f}%")
    print(f"Потенциальное уменьшение размера: {(size_before['size_mb'] - size_after['effective_size_mb']):.2f} MB")
    
    # Очистка памяти
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if return_tokenizer and tokenizer is not None:
        return model, tokenizer
    return model


def test_pruned_model(model, tokenizer, test_prompt: str = "Hello, how are you?", max_length: int = 50):
    """
    Простой тест генерации текста для проверки работоспособности модели.
    """
    model.eval()
    
    # Подготовка входных данных
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Генерация с безопасными параметрами
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            min_new_tokens=10,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def save_pruned_model(model, tokenizer, save_path: str, save_sparse: bool = True):
    """
    Сохраняет прунингованную модель.
    """
    print(f"\nСохранение модели в {save_path}...")
    
    if save_sparse:
        # Конвертируем в sparse формат для экономии места
        state_dict = model.state_dict()
        sparse_state_dict = {}
        
        for name, tensor in state_dict.items():
            if tensor.dim() > 1 and (tensor == 0).sum() > tensor.numel() * 0.1:
                # Сохраняем как sparse tensor
                sparse_state_dict[name] = tensor.to_sparse()
            else:
                sparse_state_dict[name] = tensor
        
        torch.save(sparse_state_dict, os.path.join(save_path, 'pytorch_model_sparse.bin'))
    
    # Обычное сохранение
    model.save_pretrained(save_path)
    if tokenizer:
        tokenizer.save_pretrained(save_path)
    
    print("Модель сохранена!")


# Пример использования
if __name__ == "__main__":
    input_model_name=input()
    # Конфигурация прунинга
    pruning_config = {
        'method': 'magnitude',    # Используем magnitude вместо random
        'sparsity': 0.25,         # 25% прунинг
        'iterative': True,        # Итеративный прунинг для лучшего качества
        'iterations': 3,          # 3 итерации
        'calibrate': True         # Калибровка после прунинга
    }
    
    # Прунинг модели
    model_name = input_model_name
    
    try:
        # Применение прунинга
        pruned_model, tokenizer = prune_llm_model(
            model=model_name,
            pruning_config=pruning_config,
            return_tokenizer=True
        )
        
        # Тест генерации
        print("\n=== Тест генерации ===")
        test_prompts = [
            "The future of artificial intelligence is",
            "Python programming language is",
            "Machine learning helps us"
        ]
        
        for prompt in test_prompts:
            generated = test_pruned_model(pruned_model, tokenizer, prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated}")
        
        # Сохранение модели (опционально)
        save_pruned_model(pruned_model, tokenizer, "/kaggle/working/")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()