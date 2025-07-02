# Практические примеры оптимизации инференса LLM и VLM моделей

В данном документе представлены практические примеры оптимизации инференса для больших языковых моделей (LLM) и мультимодальных моделей (VLM) с использованием различных библиотек и фреймворков.

## Содержание

- [Основы оптимизации инференса](#основы-оптимизации-инференса)
- [KV-кэширование](#kv-кэширование)
  - [Реализация в PyTorch](#реализация-kv-кэширования-в-pytorch)
  - [Реализация в Hugging Face Transformers](#реализация-kv-кэширования-в-hugging-face-transformers)
- [Оптимизация генерации текста](#оптимизация-генерации-текста)
  - [Beam Search](#beam-search)
  - [Nucleus Sampling](#nucleus-sampling)
  - [Speculative Decoding](#speculative-decoding)
- [Оптимизация с помощью специализированных библиотек](#оптимизация-с-помощью-специализированных-библиотек)
  - [vLLM](#vllm)
  - [TensorRT-LLM](#tensorrt-llm)
  - [CTranslate2](#ctranslate2)
- [Оптимизация для различных аппаратных платформ](#оптимизация-для-различных-аппаратных-платформ)
  - [CUDA оптимизации](#cuda-оптимизации)
  - [CPU оптимизации](#cpu-оптимизации)
  - [Оптимизации для мобильных устройств](#оптимизации-для-мобильных-устройств)
- [Сравнение методов оптимизации инференса](#сравнение-методов-оптимизации-инференса)

## Основы оптимизации инференса

Оптимизация инференса LLM и VLM моделей направлена на решение следующих задач:

1. **Уменьшение задержки (latency)** — время, необходимое для получения ответа от модели
2. **Увеличение пропускной способности (throughput)** — количество запросов, обрабатываемых в единицу времени
3. **Снижение потребления памяти** — особенно важно для больших моделей и длинных контекстов
4. **Эффективное использование аппаратных ресурсов** — оптимальное распределение вычислений между CPU, GPU и другими ускорителями

Основные подходы к оптимизации инференса:

- **KV-кэширование** — сохранение ключей и значений для уже обработанных токенов
- **Оптимизация алгоритмов генерации** — beam search, nucleus sampling, speculative decoding
- **Использование специализированных библиотек** — vLLM, TensorRT-LLM, CTranslate2
- **Аппаратные оптимизации** — CUDA-оптимизации, тензорные ядра, оптимизации для CPU

## KV-кэширование

KV-кэширование (Key-Value caching) — это техника, которая позволяет избежать повторных вычислений для уже обработанных токенов при автореггрессивной генерации. Это особенно важно для моделей на основе трансформеров, где каждый новый токен требует внимания ко всем предыдущим токенам.

### Реализация KV-кэширования в PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                self_attn_kv_cache=None, cross_attn_kv_cache=None):
        
        tgt2 = self.norm1(tgt)
        
        # Самовнимание с KV-кэшированием
        if self_attn_kv_cache is not None:
            # Используем кэш для ключей и значений
            k_self, v_self = self_attn_kv_cache
            # Вычисляем только для нового токена
            q_self = tgt2[-1:]
            # Получаем новые ключи и значения только для нового токена
            new_k_self, new_v_self = self.self_attn._get_key_value(q_self)
            # Объединяем с кэшем
            k_self = torch.cat([k_self, new_k_self], dim=0)
            v_self = torch.cat([v_self, new_v_self], dim=0)
            # Обновляем кэш
            self_attn_kv_cache = (k_self, v_self)
            # Выполняем внимание с кэшированными ключами и значениями
            self_attn_output, _ = self.self_attn(q_self, k_self, v_self, attn_mask=tgt_mask,
                                                key_padding_mask=tgt_key_padding_mask)
            # Объединяем с предыдущим выходом (для совместимости с интерфейсом)
            self_attn_output = torch.cat([tgt[:-1], self_attn_output], dim=0)
        else:
            # Обычное самовнимание без кэширования
            self_attn_output, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask,
                                                key_padding_mask=tgt_key_padding_mask)
            # Инициализируем кэш, если его еще нет
            k_self, v_self = self.self_attn._get_key_value(tgt2)
            self_attn_kv_cache = (k_self, v_self)
        
        tgt = tgt + self.dropout1(self_attn_output)
        tgt2 = self.norm2(tgt)
        
        # Кросс-внимание с KV-кэшированием
        if cross_attn_kv_cache is not None:
            # Используем кэш для ключей и значений энкодера
            k_cross, v_cross = cross_attn_kv_cache
            # Вычисляем только для нового токена
            q_cross = tgt2[-1:]
            # Выполняем внимание с кэшированными ключами и значениями
            cross_attn_output, _ = self.multihead_attn(q_cross, k_cross, v_cross, attn_mask=memory_mask,
                                                     key_padding_mask=memory_key_padding_mask)
            # Объединяем с предыдущим выходом (для совместимости с интерфейсом)
            cross_attn_output = torch.cat([tgt[:-1], cross_attn_output], dim=0)
        else:
            # Обычное кросс-внимание без кэширования
            cross_attn_output, _ = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask,
                                                     key_padding_mask=memory_key_padding_mask)
            # Инициализируем кэш, если его еще нет
            k_cross, v_cross = self.multihead_attn._get_key_value(memory)
            cross_attn_kv_cache = (k_cross, v_cross)
        
        tgt = tgt + self.dropout2(cross_attn_output)
        tgt2 = self.norm3(tgt)
        
        # FFN (без кэширования)
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(ffn_output)
        
        return tgt, self_attn_kv_cache, cross_attn_kv_cache

# Пример использования KV-кэширования при генерации
def generate_with_kv_cache(model, input_ids, max_length=50):
    # Инициализация кэшей для всех слоев
    num_layers = len(model.decoder.layers)
    self_attn_kv_caches = [None] * num_layers
    cross_attn_kv_caches = [None] * num_layers
    
    # Кодирование входной последовательности
    encoder_output = model.encode(input_ids)
    
    # Начальный токен для декодера
    decoder_input = torch.tensor([[model.bos_token_id]], device=input_ids.device)
    
    # Генерация токенов
    generated_ids = [model.bos_token_id]
    
    for _ in range(max_length):
        # Декодирование с использованием KV-кэша
        decoder_output = decoder_input
        for i, layer in enumerate(model.decoder.layers):
            decoder_output, self_attn_kv_caches[i], cross_attn_kv_caches[i] = layer(
                decoder_output, encoder_output,
                self_attn_kv_cache=self_attn_kv_caches[i],
                cross_attn_kv_cache=cross_attn_kv_caches[i]
            )
        
        # Предсказание следующего токена
        logits = model.output_layer(decoder_output[-1])
        next_token_id = torch.argmax(logits, dim=-1).item()
        
        # Добавление токена к результату
        generated_ids.append(next_token_id)
        
        # Проверка на конец последовательности
        if next_token_id == model.eos_token_id:
            break
        
        # Обновление входа декодера только новым токеном
        decoder_input = torch.tensor([[next_token_id]], device=input_ids.device)
    
    return generated_ids
```

### Реализация KV-кэширования в Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Загрузка модели и токенизатора
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
with torch.no_grad():
    mobile_model = optimize_for_mobile(model, input_ids.shape)

# Сохранение модели для мобильных устройств
mobile_model._save_for_lite_interpreter("distilgpt2_mobile.ptl")

# Измерение времени инференса с оптимизацией для мобильных устройств
start_time = time.time()
with torch.no_grad():
    for _ in range(10):
        outputs = mobile_model(input_ids)
mobile_time = (time.time() - start_time) / 10

# Измерение времени инференса без оптимизаций
start_time = time.time()
with torch.no_grad():
    for _ in range(10):
        outputs = model(input_ids)
standard_time = (time.time() - start_time) / 10

print(f"Время инференса с оптимизацией для мобильных устройств: {mobile_time * 1000:.4f} мс")
print(f"Время инференса без оптимизаций: {standard_time * 1000:.4f} мс")
print(f"Ускорение: {standard_time / mobile_time:.2f}x")

# Квантизация модели для мобильных устройств
def quantize_for_mobile(model, input_shape):
    # Создание примера входных данных
    example_input = torch.randint(0, 50257, input_shape)
    
    # Трассировка модели для получения TorchScript
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    
    # Квантизация модели
    quantized_model = torch.quantization.quantize_dynamic(
        traced_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    # Оптимизация для мобильных устройств
    optimized_model = torch._C._jit_pass_optimize_for_mobile(quantized_model)
    
    return optimized_model

# Оптимизация модели с квантизацией для мобильных устройств
with torch.no_grad():
    mobile_quantized_model = quantize_for_mobile(model, input_ids.shape)

# Сохранение квантизованной модели для мобильных устройств
mobile_quantized_model._save_for_lite_interpreter("distilgpt2_mobile_quantized.ptl")

# Измерение времени инференса с квантизацией для мобильных устройств
start_time = time.time()
with torch.no_grad():
    for _ in range(10):
        outputs = mobile_quantized_model(input_ids)
mobile_quantized_time = (time.time() - start_time) / 10

print(f"Время инференса с квантизацией для мобильных устройств: {mobile_quantized_time * 1000:.4f} мс")
print(f"Ускорение (квантизация vs стандартный): {standard_time / mobile_quantized_time:.2f}x")
```

## Сравнение методов оптимизации инференса

Ниже приведена сравнительная таблица различных методов оптимизации инференса для LLM и VLM моделей:

| Метод оптимизации | Ускорение (относительно базовой версии) | Сложность реализации | Поддержка в библиотеках | Применимость к задачам |
|------------------|----------------------------------------|----------------------|------------------------|------------------------|
| KV-кэширование | 2-5x | Средняя | PyTorch, Hugging Face, vLLM | Генерация текста |
| Beam Search | 1-1.5x (качество) | Низкая | Hugging Face, PyTorch | Генерация текста |
| Nucleus Sampling | 1-1.5x (качество) | Низкая | Hugging Face, PyTorch | Генерация текста |
| Speculative Decoding | 2-4x | Высокая | Экспериментальная | Генерация текста |
| vLLM | 2-10x | Низкая | vLLM | Генерация текста |
| TensorRT-LLM | 3-7x | Высокая | TensorRT-LLM | Генерация текста, классификация |
| CTranslate2 | 2-5x | Средняя | CTranslate2 | Генерация текста, классификация |
| CUDA графы | 1.5-3x | Средняя | PyTorch | Все задачи |
| JIT-компиляция | 1.2-2x | Низкая | PyTorch, TensorFlow | Все задачи |
| Квантизация INT8 | 2-4x | Средняя | PyTorch, TensorFlow, ONNX | Все задачи |
| Квантизация INT4 | 3-6x | Высокая | bitsandbytes, GPTQ | Генерация текста |
| Оптимизация для CPU | 1.5-3x | Средняя | IPEX, MKL | Все задачи |
| Оптимизация для мобильных устройств | 2-5x | Высокая | PyTorch Mobile, TFLite | Классификация, небольшие модели |

### Практические рекомендации по оптимизации инференса

1. **Выбор метода оптимизации в зависимости от задачи**:
   - Для генерации текста: KV-кэширование + vLLM или TensorRT-LLM
   - Для классификации: квантизация + JIT-компиляция
   - Для мобильных устройств: квантизация + оптимизация для мобильных устройств

2. **Комбинирование методов оптимизации**:
   - Квантизация + KV-кэширование
   - Speculative Decoding + vLLM
   - JIT-компиляция + оптимизация для CPU/GPU

3. **Оптимизация для конкретной аппаратной платформы**:
   - NVIDIA GPU: CUDA графы, TensorRT-LLM
   - Intel CPU: IPEX, MKL
   - Мобильные устройства: квантизация, оптимизация для мобильных устройств

4. **Мониторинг и профилирование**:
   - Регулярно измеряйте время инференса и потребление памяти
   - Используйте инструменты профилирования (PyTorch Profiler, NVIDIA Nsight)
   - Выявляйте узкие места и оптимизируйте их

5. **Балансирование между скоростью и качеством**:
   - Для некоторых задач можно пожертвовать небольшим снижением качества ради значительного ускорения
   - Проводите A/B тестирование для оценки влияния оптимизаций на качество результатов

6. **Оптимизация пакетной обработки**:
   - Группируйте запросы в пакеты для повышения пропускной способности
   - Используйте динамическое формирование пакетов (как в vLLM)
   - Оптимизируйте размер пакета в зависимости от доступной памяти и требуемой задержки
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Подготовка входных данных
input_text = "Искусственный интеллект может"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Генерация с использованием встроенного KV-кэширования
def generate_with_huggingface_kv_cache(model, input_ids, max_length=50):
    # Начальное состояние модели
    past_key_values = None
    output_ids = input_ids.clone()
    
    # Первый проход через модель для получения начального кэша
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits
    
    # Выбор следующего токена
    next_token_logits = logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    output_ids = torch.cat([output_ids, next_token_id], dim=-1)
    
    # Генерация остальных токенов с использованием KV-кэша
    for _ in range(max_length - 1):
        with torch.no_grad():
            # Используем только новый токен и кэш для предсказания следующего токена
            outputs = model(next_token_id, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits
        
        # Выбор следующего токена
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        output_ids = torch.cat([output_ids, next_token_id], dim=-1)
        
        # Проверка на конец последовательности
        if next_token_id[0, 0].item() == tokenizer.eos_token_id:
            break
    
    return output_ids

# Генерация текста с KV-кэшированием
output_ids = generate_with_huggingface_kv_cache(model, input_ids)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)

# Сравнение с обычной генерацией (без явного KV-кэширования)
output_ids_standard = model.generate(
    input_ids,
    max_length=50,
    num_beams=1,
    do_sample=False
)
output_text_standard = tokenizer.decode(output_ids_standard[0], skip_special_tokens=True)
print(output_text_standard)

# Измерение времени генерации с KV-кэшированием и без него
import time

# С KV-кэшированием
start_time = time.time()
for _ in range(10):
    _ = generate_with_huggingface_kv_cache(model, input_ids)
kv_cache_time = (time.time() - start_time) / 10

# Без KV-кэширования (отключаем кэширование)
start_time = time.time()
for _ in range(10):
    _ = model.generate(
        input_ids,
        max_length=50,
        num_beams=1,
        do_sample=False,
        use_cache=False
    )
no_cache_time = (time.time() - start_time) / 10

print(f"Время генерации с KV-кэшированием: {kv_cache_time:.4f} сек.")
print(f"Время генерации без KV-кэширования: {no_cache_time:.4f} сек.")
print(f"Ускорение: {no_cache_time / kv_cache_time:.2f}x")
```

## Оптимизация генерации текста

### Beam Search

Beam Search — это алгоритм генерации, который на каждом шаге сохраняет несколько наиболее вероятных последовательностей (лучей) и выбирает лучшую в конце.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Загрузка модели и токенизатора
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Подготовка входных данных
input_text = "Искусственный интеллект может"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Реализация Beam Search
def beam_search(model, input_ids, beam_width=5, max_length=50):
    # Начальное состояние
    batch_size = input_ids.shape[0]
    device = input_ids.device
    
    # Первый проход через модель
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
    
    # Получение top-k вероятностей и индексов
    log_probs, indices = torch.topk(torch.log_softmax(logits, dim=-1), beam_width, dim=-1)
    
    # Инициализация лучей
    beams = []
    for b in range(batch_size):
        for i in range(beam_width):
            beam = {
                'sequence': torch.cat([input_ids[b], indices[b, i].unsqueeze(0)], dim=0),
                'log_prob': log_probs[b, i].item(),
                'past_key_values': [(kv[0][:, :, :, :], kv[1][:, :, :, :]) for kv in past_key_values]
            }
            beams.append(beam)
    
    # Генерация остальных токенов
    for _ in range(max_length - 1):
        new_beams = []
        
        for beam in beams:
            # Проверка на конец последовательности
            if beam['sequence'][-1].item() == tokenizer.eos_token_id:
                new_beams.append(beam)
                continue
            
            # Получение последнего токена
            next_token_id = beam['sequence'][-1].unsqueeze(0).unsqueeze(0)
            
            # Предсказание с использованием кэша
            with torch.no_grad():
                outputs = model(
                    next_token_id,
                    past_key_values=beam['past_key_values'],
                    use_cache=True
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            
            # Получение top-k вероятностей и индексов
            log_probs, indices = torch.topk(torch.log_softmax(logits, dim=-1), beam_width, dim=-1)
            
            # Создание новых лучей
            for i in range(beam_width):
                new_beam = {
                    'sequence': torch.cat([beam['sequence'], indices[0, i].unsqueeze(0)], dim=0),
                    'log_prob': beam['log_prob'] + log_probs[0, i].item(),
                    'past_key_values': [(kv[0].clone(), kv[1].clone()) for kv in past_key_values]
                }
                new_beams.append(new_beam)
        
        # Сортировка и выбор лучших лучей
        beams = sorted(new_beams, key=lambda x: x['log_prob'], reverse=True)[:beam_width * batch_size]
        
        # Проверка, все ли лучи закончились
        if all(beam['sequence'][-1].item() == tokenizer.eos_token_id for beam in beams):
            break
    
    # Выбор лучшего луча для каждого примера в батче
    result = []
    for b in range(batch_size):
        best_beam = max(beams[b * beam_width:(b + 1) * beam_width], key=lambda x: x['log_prob'])
        result.append(best_beam['sequence'])
    
    return torch.stack(result)

# Генерация с использованием Beam Search
start_time = time.time()
output_ids_beam = beam_search(model, input_ids, beam_width=5)
beam_search_time = time.time() - start_time
output_text_beam = tokenizer.decode(output_ids_beam[0], skip_special_tokens=True)
print(f"Beam Search (width=5): {output_text_beam}")

# Сравнение с встроенной реализацией Beam Search
start_time = time.time()
output_ids_hf_beam = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    early_stopping=True
)
hf_beam_search_time = time.time() - start_time
output_text_hf_beam = tokenizer.decode(output_ids_hf_beam[0], skip_special_tokens=True)
print(f"Hugging Face Beam Search (width=5): {output_text_hf_beam}")

print(f"Время генерации с нашей реализацией Beam Search: {beam_search_time:.4f} сек.")
print(f"Время генерации с Hugging Face Beam Search: {hf_beam_search_time:.4f} сек.")
```

### Nucleus Sampling

Nucleus Sampling (или Top-p sampling) — это метод генерации, который выбирает следующий токен из распределения, ограниченного наиболее вероятными токенами, сумма вероятностей которых превышает порог p.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import time

# Загрузка модели и токенизатора
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Подготовка входных данных
input_text = "Искусственный интеллект может"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Реализация Nucleus Sampling (Top-p)
def nucleus_sampling(model, input_ids, top_p=0.9, temperature=1.0, max_length=50):
    # Начальное состояние
    output_ids = input_ids.clone()
    past_key_values = None
    
    for _ in range(max_length):
        with torch.no_grad():
            # Предсказание с использованием кэша
            if past_key_values is None:
                outputs = model(output_ids, use_cache=True)
            else:
                outputs = model(output_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            logits = outputs.logits[:, -1, :] / temperature
            past_key_values = outputs.past_key_values
        
        # Применение softmax для получения вероятностей
        probs = torch.softmax(logits, dim=-1)
        
        # Сортировка вероятностей по убыванию
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Вычисление кумулятивной суммы вероятностей
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Определение порога для top-p
        nucleus = cumulative_probs < top_p
        
        # Добавление токена с наибольшей вероятностью, чтобы избежать пустого ядра
        nucleus[..., 0] = True
        
        # Определение последнего индекса, который входит в ядро
        nucleus_indices = torch.where(nucleus)[1]
        
        # Выбор токенов из ядра
        nucleus_probs = sorted_probs[0, nucleus_indices]
        nucleus_indices = sorted_indices[0, nucleus_indices]
        
        # Нормализация вероятностей
        nucleus_probs = nucleus_probs / nucleus_probs.sum()
        
        # Выбор следующего токена
        next_token_id = np.random.choice(nucleus_indices.cpu().numpy(), p=nucleus_probs.cpu().numpy())
        next_token_id = torch.tensor([[next_token_id]], device=input_ids.device)
        
        # Добавление токена к результату
        output_ids = torch.cat([output_ids, next_token_id], dim=1)
        
        # Проверка на конец последовательности
        if next_token_id[0, 0].item() == tokenizer.eos_token_id:
            break
    
    return output_ids

# Генерация с использованием Nucleus Sampling
start_time = time.time()
output_ids_nucleus = nucleus_sampling(model, input_ids, top_p=0.9, temperature=1.0)
nucleus_sampling_time = time.time() - start_time
output_text_nucleus = tokenizer.decode(output_ids_nucleus[0], skip_special_tokens=True)
print(f"Nucleus Sampling (p=0.9): {output_text_nucleus}")

# Сравнение с встроенной реализацией Nucleus Sampling
start_time = time.time()
output_ids_hf_nucleus = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_p=0.9,
    temperature=1.0
)
hf_nucleus_sampling_time = time.time() - start_time
output_text_hf_nucleus = tokenizer.decode(output_ids_hf_nucleus[0], skip_special_tokens=True)
print(f"Hugging Face Nucleus Sampling (p=0.9): {output_text_hf_nucleus}")

print(f"Время генерации с нашей реализацией Nucleus Sampling: {nucleus_sampling_time:.4f} сек.")
print(f"Время генерации с Hugging Face Nucleus Sampling: {hf_nucleus_sampling_time:.4f} сек.")
```

### Speculative Decoding

Speculative Decoding — это метод ускорения генерации, который использует меньшую модель для предсказания нескольких токенов, а затем проверяет их с помощью основной модели.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Загрузка основной и вспомогательной моделей
main_model_name = "gpt2-large"  # Большая модель
draft_model_name = "gpt2"       # Маленькая модель

main_model = AutoModelForCausalLM.from_pretrained(main_model_name)
main_model.eval()

draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)
draft_model.eval()

tokenizer = AutoTokenizer.from_pretrained(main_model_name)

# Подготовка входных данных
input_text = "Искусственный интеллект может"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Реализация Speculative Decoding
def speculative_decoding(main_model, draft_model, input_ids, n_draft=5, max_length=50):
    # Начальное состояние
    output_ids = input_ids.clone()
    main_past_key_values = None
    draft_past_key_values = None
    
    while output_ids.shape[1] < max_length:
        # Шаг 1: Генерация n_draft токенов с помощью вспомогательной модели
        draft_input_ids = output_ids
        draft_tokens = []
        
        for _ in range(n_draft):
            with torch.no_grad():
                # Предсказание с использованием кэша
                if draft_past_key_values is None:
                    draft_outputs = draft_model(draft_input_ids, use_cache=True)
                else:
                    draft_outputs = draft_model(
                        draft_input_ids[:, -1:],
                        past_key_values=draft_past_key_values,
                        use_cache=True
                    )
                
                draft_logits = draft_outputs.logits[:, -1, :]
                draft_past_key_values = draft_outputs.past_key_values
            
            # Выбор следующего токена
            next_token_id = torch.argmax(draft_logits, dim=-1).unsqueeze(-1)
            draft_tokens.append(next_token_id.item())
            
            # Обновление входных данных для следующего шага
            draft_input_ids = torch.cat([draft_input_ids, next_token_id], dim=1)
            
            # Проверка на конец последовательности
            if next_token_id[0, 0].item() == tokenizer.eos_token_id:
                break
        
        # Шаг 2: Проверка предсказанных токенов с помощью основной модели
        with torch.no_grad():
            # Получение вероятностей от основной модели для всех токенов
            if main_past_key_values is None:
                main_outputs = main_model(output_ids, use_cache=True)
                main_past_key_values = main_outputs.past_key_values
            
            # Проверка каждого предсказанного токена
            accepted_tokens = []
            for i, token_id in enumerate(draft_tokens):
                # Получение вероятностей для текущего токена
                main_outputs = main_model(
                    torch.tensor([[token_id]], device=input_ids.device),
                    past_key_values=main_past_key_values,
                    use_cache=True
                )
                main_logits = main_outputs.logits[:, -1, :]
                main_past_key_values = main_outputs.past_key_values
                
                # Вычисление вероятностей
                main_probs = torch.softmax(main_logits, dim=-1)
                token_prob = main_probs[0, token_id].item()
                
                # Принятие или отклонение токена
                if torch.rand(1).item() < token_prob:
                    accepted_tokens.append(token_id)
                else:
                    # Если токен отклонен, генерируем новый токен с основной моделью
                    next_token_id = torch.multinomial(main_probs, 1)
                    accepted_tokens.append(next_token_id.item())
                    break
            
            # Добавление принятых токенов к результату
            for token_id in accepted_tokens:
                output_ids = torch.cat([output_ids, torch.tensor([[token_id]], device=input_ids.device)], dim=1)
                
                # Проверка на конец последовательности или достижение максимальной длины
                if token_id == tokenizer.eos_token_id or output_ids.shape[1] >= max_length:
                    break
            
            # Если не было принято ни одного токена, генерируем один токен с основной моделью
            if not accepted_tokens:
                main_outputs = main_model(
                    output_ids[:, -1:],
                    past_key_values=main_past_key_values,
                    use_cache=True
                )
                main_logits = main_outputs.logits[:, -1, :]
                main_past_key_values = main_outputs.past_key_values
                
                next_token_id = torch.argmax(main_logits, dim=-1).unsqueeze(-1)
                output_ids = torch.cat([output_ids, next_token_id], dim=1)
                
                if next_token_id[0, 0].item() == tokenizer.eos_token_id:
                    break
        
        # Проверка на конец последовательности
        if output_ids[0, -1].item() == tokenizer.eos_token_id:
            break
    
    return output_ids

# Генерация с использованием Speculative Decoding
start_time = time.time()
output_ids_spec = speculative_decoding(main_model, draft_model, input_ids, n_draft=5)
spec_decoding_time = time.time() - start_time
output_text_spec = tokenizer.decode(output_ids_spec[0], skip_special_tokens=True)
print(f"Speculative Decoding: {output_text_spec}")

# Сравнение с обычной генерацией
start_time = time.time()
output_ids_standard = main_model.generate(
    input_ids,
    max_length=50,
    num_beams=1,
    do_sample=False
)
standard_time = time.time() - start_time
output_text_standard = tokenizer.decode(output_ids_standard[0], skip_special_tokens=True)
print(f"Стандартная генерация: {output_text_standard}")

print(f"Время генерации с Speculative Decoding: {spec_decoding_time:.4f} сек.")
print(f"Время генерации со стандартной генерацией: {standard_time:.4f} сек.")
print(f"Ускорение: {standard_time / spec_decoding_time:.2f}x")
```

## Оптимизация с помощью специализированных библиотек

### vLLM

vLLM — это библиотека для эффективного инференса LLM, которая использует оптимизированное KV-кэширование и пакетную обработку запросов.

```python
# Установка vLLM
# pip install vllm

from vllm import LLM, SamplingParams
import time

# Инициализация модели с vLLM
model_name = "gpt2"
vllm_model = LLM(model=model_name)

# Параметры генерации
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    max_tokens=50
)

# Подготовка входных данных
input_text = "Искусственный интеллект может"

# Генерация с использованием vLLM
start_time = time.time()
outputs = vllm_model.generate([input_text], sampling_params)
vllm_time = time.time() - start_time

for output in outputs:
    print(f"vLLM: {output.outputs[0].text}")

# Сравнение с Hugging Face Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Загрузка модели и токенизатора
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Подготовка входных данных
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Генерация с использованием Hugging Face
start_time = time.time()
output_ids = model.generate(
    input_ids,
    max_length=input_ids.shape[1] + 50,
    do_sample=True,
    temperature=1.0,
    top_p=0.95
)
hf_time = time.time() - start_time
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Hugging Face: {output_text}")

print(f"Время генерации с vLLM: {vllm_time:.4f} сек.")
print(f"Время генерации с Hugging Face: {hf_time:.4f} сек.")
print(f"Ускорение: {hf_time / vllm_time:.2f}x")

# Пакетная обработка запросов с vLLM
input_texts = [
    "Искусственный интеллект может",
    "Нейронные сети используются для",
    "Глубокое обучение позволяет",
    "Трансформеры — это архитектура, которая"
]

# Генерация с использованием vLLM (пакетная обработка)
start_time = time.time()
batch_outputs = vllm_model.generate(input_texts, sampling_params)
vllm_batch_time = time.time() - start_time

for output in batch_outputs:
    print(f"vLLM (batch): {output.outputs[0].text}")

# Генерация с использованием Hugging Face (последовательная обработка)
start_time = time.time()
hf_batch_outputs = []
for text in input_texts:
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 50,
        do_sample=True,
        temperature=1.0,
        top_p=0.95
    )
    hf_batch_outputs.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
hf_batch_time = time.time() - start_time

for output in hf_batch_outputs:
    print(f"Hugging Face (sequential): {output}")

print(f"Время пакетной генерации с vLLM: {vllm_batch_time:.4f} сек.")
print(f"Время последовательной генерации с Hugging Face: {hf_batch_time:.4f} сек.")
print(f"Ускорение: {hf_batch_time / vllm_batch_time:.2f}x")
```

### TensorRT-LLM

TensorRT-LLM — это библиотека от NVIDIA для оптимизации инференса LLM с использованием TensorRT.

```python
# Установка TensorRT-LLM
# pip install tensorrt-llm

import tensorrt_llm
import torch
import time
from pathlib import Path

# Путь для сохранения оптимизированной модели
engine_path = Path("./trt_engines/gpt2")
engine_path.mkdir(parents=True, exist_ok=True)

# Конвертация модели Hugging Face в TensorRT-LLM
def build_tensorrt_llm_engine(model_name, engine_path):
    from tensorrt_llm.builder import Builder
    from tensorrt_llm.models import PretrainedModel
    from tensorrt_llm.quantization import QuantMode
    
    # Загрузка модели из Hugging Face
    pretrained_model = PretrainedModel(model_name)
    
    # Настройка параметров сборки
    builder = Builder()
    builder.set_max_batch_size(8)
    builder.set_max_input_len(1024)
    builder.set_max_output_len(1024)
    
    # Сборка движка TensorRT
    engine = builder.build(pretrained_model, engine_path)
    
    return engine

# Инференс с использованием TensorRT-LLM
def generate_with_tensorrt_llm(engine_path, input_text, tokenizer, max_output_len=50):
    from tensorrt_llm.runtime import ModelRunner
    
    # Инициализация движка
    runner = ModelRunner.from_dir(engine_path)
    
    # Токенизация входного текста
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Генерация
    output_ids = runner.generate(
        input_ids=input_ids.numpy(),
        max_output_len=max_output_len,
        temperature=1.0,
        top_p=0.95
    )
    
    # Декодирование результата
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return output_text

# Загрузка токенизатора
from transformers import AutoTokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Сборка движка TensorRT-LLM (выполняется один раз)
# build_tensorrt_llm_engine(model_name, engine_path)

# Подготовка входных данных
input_text = "Искусственный интеллект может"

# Генерация с использованием TensorRT-LLM
start_time = time.time()
output_text_trt = generate_with_tensorrt_llm(engine_path, input_text, tokenizer)
trt_time = time.time() - start_time
print(f"TensorRT-LLM: {output_text_trt}")

# Сравнение с Hugging Face Transformers
from transformers import AutoModelForCausalLM

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Подготовка входных данных
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Генерация с использованием Hugging Face
start_time = time.time()
output_ids = model.generate(
    input_ids,
    max_length=input_ids.shape[1] + 50,
    do_sample=True,
    temperature=1.0,
    top_p=0.95
)
hf_time = time.time() - start_time
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Hugging Face: {output_text}")

print(f"Время генерации с TensorRT-LLM: {trt_time:.4f} сек.")
print(f"Время генерации с Hugging Face: {hf_time:.4f} сек.")
print(f"Ускорение: {hf_time / trt_time:.2f}x")
```

### CTranslate2

CTranslate2 — это библиотека для эффективного инференса моделей на основе трансформеров, оптимизированная для CPU и GPU.

```python
# Установка CTranslate2
# pip install ctranslate2

import ctranslate2
import time
from transformers import AutoTokenizer

# Конвертация модели Hugging Face в формат CTranslate2
def convert_to_ctranslate2(model_name, output_dir):
    import os
    from ctranslate2.converters import TransformersConverter
    
    # Создание директории для сохранения модели
    os.makedirs(output_dir, exist_ok=True)
    
    # Конвертация модели
    converter = TransformersConverter(model_name)
    converter.convert(output_dir, quantization="int8")

# Инференс с использованием CTranslate2
def generate_with_ctranslate2(model_path, input_text, tokenizer, max_length=50):
    # Загрузка модели
    generator = ctranslate2.Generator(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Токенизация входного текста
    input_tokens = tokenizer.encode(input_text)
    
    # Генерация
    results = generator.generate_batch(
        [input_tokens],
        max_length=max_length,
        sampling_topk=1,  # greedy decoding
        include_prompt_in_result=True
    )
    
    # Декодирование результата
    output_text = tokenizer.decode(results[0].sequences[0], skip_special_tokens=True)
    
    return output_text

# Загрузка токенизатора
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Путь для сохранения конвертированной модели
ctranslate2_model_path = "./ctranslate2_models/gpt2"

# Конвертация модели (выполняется один раз)
# convert_to_ctranslate2(model_name, ctranslate2_model_path)

# Подготовка входных данных
input_text = "Искусственный интеллект может"

# Генерация с использованием CTranslate2
start_time = time.time()
output_text_ct2 = generate_with_ctranslate2(ctranslate2_model_path, input_text, tokenizer)
ct2_time = time.time() - start_time
print(f"CTranslate2: {output_text_ct2}")

# Сравнение с Hugging Face Transformers
import torch
from transformers import AutoModelForCausalLM

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Подготовка входных данных
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Генерация с использованием Hugging Face
start_time = time.time()
output_ids = model.generate(
    input_ids,
    max_length=input_ids.shape[1] + 50,
    num_beams=1,
    do_sample=False
)
hf_time = time.time() - start_time
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Hugging Face: {output_text}")

print(f"Время генерации с CTranslate2: {ct2_time:.4f} сек.")
print(f"Время генерации с Hugging Face: {hf_time:.4f} сек.")
print(f"Ускорение: {hf_time / ct2_time:.2f}x")
```

## Оптимизация для различных аппаратных платформ

### CUDA оптимизации

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузка модели и токенизатора
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Перемещение модели на GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Оптимизация модели с помощью CUDA графов
def optimize_with_cuda_graphs(model, input_ids):
    # Создание статического входа для CUDA графа
    static_input = input_ids.clone().to(device)
    
    # Разогрев модели
    for _ in range(3):
        _ = model(static_input)
    
    # Создание CUDA графа
    g = torch.cuda.CUDAGraph()
    
    # Захват операций в граф
    with torch.cuda.graph(g):
        static_output = model(static_input)
    
    def inference_with_graph(new_input):
        # Копирование новых входных данных в статический вход
        static_input.copy_(new_input)
        # Выполнение графа
        g.replay()
        # Возврат результата
        return static_output
    
    return inference_with_graph

# Подготовка входных данных
input_text = "Искусственный интеллект может"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Оптимизация модели с CUDA графами
model.eval()
with torch.no_grad():
    inference_fn = optimize_with_cuda_graphs(model, input_ids)

# Измерение времени инференса с CUDA графами
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        outputs = inference_fn(input_ids)
cuda_graph_time = (time.time() - start_time) / 100

# Измерение времени инференса без CUDA графов
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        outputs = model(input_ids)
standard_time = (time.time() - start_time) / 100

print(f"Время инференса с CUDA графами: {cuda_graph_time * 1000:.4f} мс")
print(f"Время инференса без оптимизаций: {standard_time * 1000:.4f} мс")
print(f"Ускорение: {standard_time / cuda_graph_time:.2f}x")

# Оптимизация с использованием TensorRT
import torch_tensorrt

# Конвертация модели в TensorRT
def optimize_with_tensorrt(model, input_shape):
    # Создание примера входных данных
    example_input = torch.randint(0, 50257, input_shape).to(device)
    
    # Трассировка модели для получения TorchScript
    traced_model = torch.jit.trace(model, example_input)
    
    # Компиляция модели с TensorRT
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=[torch_tensorrt.Input(input_shape)],
        enabled_precisions={torch.float16}  # Использование FP16 для ускорения
    )
    
    return trt_model

# Оптимизация модели с TensorRT
# trt_model = optimize_with_tensorrt(model, input_ids.shape)

# Измерение времени инференса с TensorRT
# start_time = time.time()
# with torch.no_grad():
#     for _ in range(100):
#         outputs = trt_model(input_ids)
# tensorrt_time = (time.time() - start_time) / 100

# print(f"Время инференса с TensorRT: {tensorrt_time * 1000:.4f} мс")
# print(f"Ускорение (TensorRT vs стандартный): {standard_time / tensorrt_time:.2f}x")
```

### CPU оптимизации

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузка модели и токенизатора
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Оптимизация модели для CPU с использованием JIT
def optimize_with_jit(model, input_ids):
    # Трассировка модели для получения TorchScript
    with torch.no_grad():
        traced_model = torch.jit.trace(model, input_ids)
    
    # Оптимизация модели
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    return optimized_model

# Подготовка входных данных
input_text = "Искусственный интеллект может"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Оптимизация модели с JIT
model.eval()
with torch.no_grad():
    jit_model = optimize_with_jit(model, input_ids)

# Измерение времени инференса с JIT
start_time = time.time()
with torch.no_grad():
    for _ in range(10):
        outputs = jit_model(input_ids)
jit_time = (time.time() - start_time) / 10

# Измерение времени инференса без оптимизаций
start_time = time.time()
with torch.no_grad():
    for _ in range(10):
        outputs = model(input_ids)
standard_time = (time.time() - start_time) / 10

print(f"Время инференса с JIT: {jit_time * 1000:.4f} мс")
print(f"Время инференса без оптимизаций: {standard_time * 1000:.4f} мс")
print(f"Ускорение: {standard_time / jit_time:.2f}x")

# Оптимизация с использованием Intel MKL
import os

# Настройка параметров Intel MKL
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

# Измерение времени инференса с MKL
start_time = time.time()
with torch.no_grad():
    for _ in range(10):
        outputs = model(input_ids)
mkl_time = (time.time() - start_time) / 10

print(f"Время инференса с MKL: {mkl_time * 1000:.4f} мс")
print(f"Ускорение (MKL vs стандартный): {standard_time / mkl_time:.2f}x")

# Оптимизация с использованием Intel IPEX
import intel_extension_for_pytorch as ipex

# Оптимизация модели с IPEX
model_ipex = ipex.optimize(model)

# Измерение времени инференса с IPEX
start_time = time.time()
with torch.no_grad():
    for _ in range(10):
        outputs = model_ipex(input_ids)
ipex_time = (time.time() - start_time) / 10

print(f"Время инференса с IPEX: {ipex_time * 1000:.4f} мс")
print(f"Ускорение (IPEX vs стандартный): {standard_time / ipex_time:.2f}x")
```

### Оптимизации для мобильных устройств

```python
import torch
import time

# Оптимизация модели для мобильных устройств с использованием PyTorch Mobile
def optimize_for_mobile(model, input_shape):
    # Создание примера входных данных
    example_input = torch.randint(0, 50257, input_shape)
    
    # Трассировка модели для получения TorchScript
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    
    # Оптимизация модели для мобильных устройств
    optimized_model = torch._C._jit_pass_optimize_for_mobile(traced_model)
    
    return optimized_model

# Загрузка небольшой модели для мобильных устройств
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "distilgpt2"  # Меньшая версия GPT-2
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Подготовка входных данных
input_text = "Искусственный интеллект может"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Оптимизация модели для мобильных устройств
model.eval()