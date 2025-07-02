# Практические примеры квантизации моделей

В данном документе представлены практические примеры квантизации моделей с использованием различных библиотек и фреймворков.

## Содержание

- [Квантизация с использованием PyTorch](#квантизация-с-использованием-pytorch)
  - [FP16/BF16 квантизация](#fp16bf16-квантизация)
  - [INT8 квантизация](#int8-квантизация)
  - [INT4 квантизация](#int4-квантизация)
- [Квантизация с использованием Hugging Face](#квантизация-с-использованием-hugging-face)
  - [Квантизация с использованием bitsandbytes](#квантизация-с-использованием-bitsandbytes)
  - [Квантизация с использованием Optimum](#квантизация-с-использованием-optimum)
- [Квантизация с использованием специализированных библиотек](#квантизация-с-использованием-специализированных-библиотек)
  - [vLLM](#vllm)
  - [GPTQ](#gptq)
  - [AWQ](#awq)

## Квантизация с использованием PyTorch

### FP16/BF16 квантизация

FP16 и BF16 — форматы с плавающей точкой половинной точности, которые позволяют уменьшить размер модели в 2 раза при минимальной потере точности.

#### Смешанная точность с FP16

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Создание модели
model = Transformer(...).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Создание скейлера для предотвращения underflow
scaler = GradScaler()

# Обучение с использованием смешанной точности
for epoch in range(num_epochs):
    for batch in train_dataloader:
        inputs, targets = batch
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # Автоматическое приведение к FP16 внутри блока
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Скейлинг градиентов для предотвращения underflow
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

#### Инференс с BF16

```python
import torch

# Загрузка модели
model = Transformer(...).cuda()

# Приведение модели к BF16
model = model.to(torch.bfloat16)

# Инференс с использованием BF16
with torch.no_grad():
    inputs = inputs.cuda().to(torch.bfloat16)
    outputs = model(inputs)
    outputs = outputs.to(torch.float32)  # Приведение обратно к FP32 для дальнейшей обработки
```

### INT8 квантизация

INT8 квантизация представляет веса и активации модели в 8-битном целочисленном формате, что позволяет уменьшить размер модели в 4 раза.

#### Статическая квантизация (PTQ)

```python
import torch
import torch.quantization as quantization

# Определение модели с поддержкой квантизации
class QuantizableTransformer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(QuantizableTransformer, self).__init__(*args, **kwargs)
        # Замена обычных модулей на квантизируемые
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        # Обычный прямой проход
        x = super().forward(x)
        x = self.dequant(x)
        return x

# Создание модели
model = QuantizableTransformer(...)

# Настройка квантизации
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Подготовка модели к квантизации
model_prepared = torch.quantization.prepare(model)

# Калибровка модели на данных
with torch.no_grad():
    for batch in calibration_dataloader:
        inputs = batch[0]
        model_prepared(inputs)

# Квантизация модели
model_quantized = torch.quantization.convert(model_prepared)

# Сохранение квантизованной модели
torch.save(model_quantized.state_dict(), "transformer_int8.pt")
```

#### Динамическая квантизация

```python
import torch

# Загрузка предобученной модели
model = Transformer(...)
model.eval()  # Переключение в режим оценки

# Применение динамической квантизации
quantized_model = torch.quantization.quantize_dynamic(
    model,  # модель для квантизации
    {torch.nn.Linear},  # типы слоёв для квантизации
    dtype=torch.qint8  # тип данных для квантизации
)

# Инференс с квантизованной моделью
with torch.no_grad():
    for batch in test_dataloader:
        inputs = batch[0]
        outputs = quantized_model(inputs)
        # Дальнейшая обработка
```

#### Квантизация с учётом обучения (QAT)

```python
import torch
import torch.quantization as quantization

# Создание модели с поддержкой квантизации
model = QuantizableTransformer(...)

# Настройка квантизации
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# Подготовка модели к QAT
model_prepared = torch.quantization.prepare_qat(model)

# Обучение с учётом квантизации
for epoch in range(num_epochs):
    model_prepared.train()
    for batch in train_dataloader:
        inputs, targets = batch
        outputs = model_prepared(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Конвертация модели после обучения
model_prepared.eval()
model_quantized = torch.quantization.convert(model_prepared)

# Сохранение квантизованной модели
torch.save(model_quantized.state_dict(), "transformer_qat_int8.pt")
```

### INT4 квантизация

INT4 квантизация представляет веса и активации модели в 4-битном целочисленном формате, что позволяет уменьшить размер модели в 8 раз.

#### Квантизация с использованием bitsandbytes

```python
import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Настройка 4-битной квантизации
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",  # Normalized float 4
    bnb_4bit_use_double_quant=True
)

# Загрузка модели с 4-битной квантизацией
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# Инференс с квантизованной моделью
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Квантизация с использованием Hugging Face

### Квантизация с использованием bitsandbytes

#### 8-битная квантизация

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Настройка 8-битной квантизации
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# Загрузка модели с 8-битной квантизацией
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# Инференс с квантизованной моделью
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Квантизация с использованием Optimum

```python
from optimum.intel import INCQuantizer
from transformers import AutoModelForSequenceClassification

# Загрузка модели
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Создание квантизатора
quantizer = INCQuantizer.from_pretrained(model)

# Настройка квантизации
quantization_config = {
    "approach": "static",
    "quant_format": "QDQ",
    "op_type_dict": {
        "MatMul": {"weight": {"dtype": ["int8"], "scheme": ["sym"], "granularity": ["per_tensor"]}},
        "Gather": {"weight": {"dtype": ["int8"], "scheme": ["sym"], "granularity": ["per_tensor"]}},
    },
}

# Квантизация модели
quantized_model = quantizer.quantize(
    quantization_config=quantization_config,
    calibration_dataset="glue",
    calibration_dataset_config_name="sst2",
    calibration_split="train",
    batch_size=8,
    num_samples=100,
)

# Сохранение квантизованной модели
quantized_model.save_pretrained("bert-base-uncased-quantized")
```

## Квантизация с использованием специализированных библиотек

### vLLM

vLLM — библиотека для эффективного инференса LLM с поддержкой квантизованных моделей и оптимизированным PagedAttention.

```python
from vllm import LLM, SamplingParams

# Загрузка модели с квантизацией
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="awq",  # Поддерживаются "awq", "gptq", "squeezellm"
    dtype="half"  # "half" для FP16, "bfloat16" для BF16
)

# Настройка параметров генерации
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

# Генерация текста
prompts = ["Explain quantum computing in simple terms"]
outputs = llm.generate(prompts, sampling_params)

# Вывод результатов
for output in outputs:
    print(output.outputs[0].text)
```

### GPTQ

GPTQ — алгоритм квантизации, оптимизированный для трансформеров, который позволяет квантизовать модели до 3-4 бит с минимальной потерей точности.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузка модели с GPTQ квантизацией
model_name = "TheBloke/Llama-2-7B-GPTQ"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Генерация текста
prompt = "Explain quantum computing in simple terms"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
```

### AWQ

AWQ (Activation-aware Weight Quantization) — метод квантизации, который учитывает активации при квантизации весов, что позволяет достичь лучшей точности при низкой битности.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузка модели с AWQ квантизацией
model_name = "TheBloke/Llama-2-7B-AWQ"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Генерация текста
prompt = "Explain quantum computing in simple terms"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
```

## Сравнение методов квантизации

| Метод | Формат | Сжатие (относительно FP32) | Потеря точности | Скорость инференса | Сложность реализации | Поддержка в PyTorch | Поддержка в Hugging Face |
|-------|--------|----------------------------|-----------------|-------------------|----------------------|---------------------|---------------------------|
| FP16 | 16-бит с плавающей точкой | 2x | Минимальная | 1.5-2x быстрее | Низкая | Встроенная | Встроенная |
| BF16 | 16-бит с плавающей точкой (альтернативный формат) | 2x | Низкая | 1.5-2x быстрее | Низкая | Встроенная | Встроенная |
| INT8 (PTQ) | 8-бит целочисленный | 4x | Средняя | 2-3x быстрее | Средняя | Встроенная | Через Optimum, bitsandbytes |
| INT8 (QAT) | 8-бит целочисленный | 4x | Низкая-средняя | 2-3x быстрее | Высокая | Встроенная | Через Optimum |
| INT4 (PTQ) | 4-бит целочисленный | 8x | Высокая | 3-4x быстрее | Высокая | Через bitsandbytes | Через bitsandbytes |
| INT4 (QAT) | 4-бит целочисленный | 8x | Средняя-высокая | 3-4x быстрее | Очень высокая | Через bitsandbytes | Через bitsandbytes |
| GPTQ | 3-4-бит целочисленный | 8-10x | Средняя | 3-4x быстрее | Средняя | Через специализированные библиотеки | Через TheBloke модели |
| AWQ | 4-бит целочисленный | 8x | Низкая-средняя | 3-4x быстрее | Средняя | Через специализированные библиотеки | Через TheBloke модели |

## Практические рекомендации

1. **Выбор метода квантизации**:
   - Для задач, требующих высокой точности, начните с FP16/BF16
   - Для общих задач генерации текста, INT8 PTQ обычно обеспечивает хороший баланс
   - Для мобильных устройств и edge-вычислений, рассмотрите INT4 QAT или GPTQ/AWQ

2. **Оптимизация процесса квантизации**:
   - Используйте репрезентативный набор данных для калибровки
   - Сохраняйте первый и последний слои в более высокой точности (FP16/BF16)
   - Эмбеддинги часто можно квантизовать агрессивнее без значительной потери точности
   - Слои внимания обычно более чувствительны к квантизации, чем полносвязные слои

3. **Оценка производительности**:
   - Всегда проводите A/B тестирование между оригинальной и квантизованной моделями
   - Используйте метрики, специфичные для вашей задачи (BLEU, ROUGE, точность классификации и т.д.)
   - Измеряйте не только точность, но и скорость инференса и использование памяти