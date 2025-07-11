# Документация модуля evaluate_metrics.py

## Введение

Модуль `evaluate_metrics.py` предоставляет универсальный фреймворк для оценки производительности моделей машинного обучения на различных задачах, таких как классификация, перевод, генерация текста и компьютерное зрение. Он поддерживает модели Hugging Face, включая `GPT-2`, `LLaMA-3-8B`, `Mistral-7B` и `OwlV2`, и позволяет вычислять широкий спектр метрик, от точности (`accuracy`) до энергоэффективности (`energy`). Модуль разработан с акцентом на гибкость, читаемость и оптимальность, минимизируя дублирование кода и обеспечивая интуитивное использование.

## Архитектурные и функциональные решения

### 1. Модульная структура
Модуль разделён на три ключевые части:
- **Загрузка моделей** (`load_model`): Загружает модель и токенизатор/процессор для указанного имени модели.
- **Инференс** (`train_func`): Выполняет инференс модели и собирает данные для метрик.
- **Оценка метрик** (`metrics_evaluate`): Вычисляет метрики, специфичные для задачи, и агрегирует результаты.

**Преимущества**:
- **Разделение ответственности**: Каждая функция выполняет одну задачу, что упрощает отладку и поддержку.
- **Повторное использование**: Функции можно использовать независимо, например, для кастомных метрик или частичного инференса.

### 2. Поддержка различных типов задач
Используется перечисление `TaskType` (`CLASSIFICATION`, `TRANSLATION`, `GENERATION`, `VISION`), которое определяет:
- Какие метрики применяются к задаче (через словарь `TASK_METRICS`).
- Как обрабатывать входные данные и предсказания.

**Оптимальность**:
- Централизованное управление задачами через `TaskType` позволяет легко добавлять новые типы задач.
- `TASK_METRICS` сопоставляет задачи с метриками, минимизируя ручную настройку.

### 3. Универсальная обработка данных
- **Field Mapping**: Параметр `field_mapping` позволяет задавать пользовательские имена полей в датасете (например, `{"text": "input"}`), обеспечивая совместимость с различными форматами данных.
- **Raw Text vs. Text**: Для поддержки метрик, требующих строк (`List[str]`), таких как `perplexity`, и токенизированных данных (`Dict[str, torch.Tensor]`), таких как входы модели, используются два поля:
  - `raw_text`: Сохраняет исходные строки для метрик, требующих текст в формате `List[str]`.
  - `text`: Содержит токенизированные данные для инференса или метрик, таких как `flops` для задач компьютерного зрения.
- **Валидация данных**: Проверки входных данных (например, наличие `text` и `labels` для классификации) предотвращают ошибки на этапе инференса.

**Красота и оптимальность**:
- Разделение `raw_text` и `text` устраняет проблему перезаписи данных, обеспечивая совместимость с разными метриками.
- `field_mapping` делает модуль независимым от структуры датасета, повышая универсальность.

### 4. Метрики и их обработка
- **Базовые метрики**: Словарь `base_metrics` сопоставляет названия метрик с их функциями, обеспечивая единый интерфейс.
- **Обработка исключений**: Метрики вычисляются в блоке `try-except`, чтобы ошибки в одной метрике не прерывали выполнение остальных.
- **Агрегация результатов**: Скалярные метрики усредняются через `np.nanmean`, а словарные метрики (например, `rouge`) возвращаются без изменений.

**Элегантность**:
- Единый интерфейс для всех метрик упрощает добавление новых.
- Логирование (`logging.info`) и обработка ошибок (`logging.warning`) улучшают отладку.

### 5. Производительность
- Используется `torch.no_grad()` для оптимизации памяти при инференсе.
- Замеры производительности (`latency`, `memory`, `flops`, `throughput`, `energy`) интегрированы с помощью библиотек `psutil` и `codecarbon`.
- Поддержка CUDA с синхронизацией через `torch.cuda.synchronize()` обеспечивает точные замеры.

**Оптимальность**:
- Минимизация вычислительных затрат за счёт отключения градиентов.
- Гибкость в выборе устройства (`cuda` или `cpu`).

## Предобработка данных

### Извлечение данных
- Данные извлекаются из датасета Hugging Face с использованием `field_mapping` для гибкого доступа к полям (`text`, `label`, `references`, `image`).
- Для задач `TRANSLATION` и `GENERATION` поле `references` преобразуется в `List[List[str]]`, чтобы поддерживать метрики `cider` и `spice`, которые требуют нескольких референсов.
- Для задач `VISION` проверяется, что `references` имеет формат `List[Tuple[torch.Tensor, torch.Tensor]]` (bounding boxes и метки).

### Токенизация
- Для текстовых задач (`CLASSIFICATION`, `TRANSLATION`, `GENERATION`):
  - Текст (`List[str]`) токенизируется с помощью `processor` (Hugging Face tokenizer) в `Dict[str, torch.Tensor]` с параметрами `padding=True`, `truncation=True`.
  - Токенизированный текст сохраняется в `metrics_data["text"]`, а исходный — в `metrics_data["raw_text"]`.
- Для задач `VISION`:
  - `OwlViTProcessor` обрабатывает текст и изображения, возвращая `input_ids` и `pixel_values`, которые сохраняются в `metrics_data["text"]` и `metrics_data["images"]`.

### Валидация
- Проверяется наличие обязательных полей (например, `text` и `labels` для `CLASSIFICATION`).
- Для `VISION` проверяется формат `references` для соответствия требованиям метрик (`iou`, `map`, и т.д.).

**Почему это важно**:
- Сохранение `raw_text` решает проблему, когда токенизация делала текст непригодным для метрик, требующих строк (`perplexity`, `clip_score_vision`).
- Валидация предотвращает ошибки на этапе инференса, улучшая надёжность.

## Raw Text vs. Text

### Проблема
Ряд метрик, таких как `perplexity`, `clip_score_vision` и `flops` (для текстовых задач), требуют текст в формате `List[str]`. Однако `train_func` токенизирует текст для инференса модели, преобразуя его в `Dict[str, torch.Tensor]` или `torch.Tensor`. Ранее это приводило к ошибкам, так как метрики получали неподходящий формат.

### Решение
- **Введение `raw_text`**: Исходный текст (`List[str]`) сохраняется в `metrics_data["raw_text"]` до токенизации.
- **Сохранение `text`**: Токенизированный текст сохраняется в `metrics_data["text"]` для инференса и метрик, таких как `flops` для задач `VISION`.
- **Выбор в `metrics_evaluate`**: Для каждой метрики выбирается подходящий вход:
  - `raw_text` для `perplexity`, `clip_score_vision`, `flops` (текстовые задачи).
  - `text` для `flops` (задачи `VISION`), где требуется токенизированный текст.

### Преимущества
- **Универсальность**: Поддержка как строковых, так и токенизированных входов.
- **Читаемость**: Явное разделение `raw_text` и `text` в `metrics_data` делает код понятным.
- **Оптимальность**: Минимальные изменения (добавление одного поля) решают проблему без рефакторинга.

## Task Type and Task Name

### Task Type
`TaskType` — это перечисление (`Enum`), определяющее категорию задачи:
- `CLASSIFICATION`: Для задач классификации текста (например, анализ тональности).
- `TRANSLATION`: Для задач перевода текста.
- `GENERATION`: Для генерации текста (например, автодополнение).
- `VISION`: Для задач компьютерного зрения (например, детекция объектов).

**Роль**:
- Определяет, какие метрики вычисляются (через `TASK_METRICS`).
- Управляет логикой инференса и предобработки в `train_func`.

### Task Name
`task_name` — это строка, которая уточняет задачу внутри категории `TaskType`. По умолчанию равна `task_type.value` (например, `"classification"`), но может быть переопределена через `field_mapping["task_name"]`.

**Использование**:
- Метрики `glue`, `helm`, `mmlu` требуют специфичных значений `task_name`:
  - `glue`: Например, `"glue_sst2"`, `"glue_mnli"`.
  - `helm`: `"helm"`.
  - `mmlu`: `"mmlu"` или подзадачи, например, `"mmlu_math"`.
- Пользователь может задать `field_mapping={"task_name": "glue_sst2"}`, чтобы метрика `glue` корректно обработала задачу.

**Преимущества**:
- **Гибкость**: Позволяет поддерживать подзадачи без изменения `TaskType`.
- **Интуитивность**: Использует существующий параметр `field_mapping`, не требуя новых аргументов.

## Руководство по использованию

### 1. Загрузка моделей
Функция `load_model` загружает модель и токенизатор/процессор.

**Пример**:
```python
from evaluate_metrics import load_model, TaskType

# Загрузка модели GPT-2 для задачи генерации
model, tokenizer = load_model("gpt2", device="cuda")
```

**Поддерживаемые модели**:
- `"gpt2"`: Для генерации и классификации (если настроена как `BertForSequenceClassification`).
- `"llama3_8b"`, `"mistral_7b"`: Для генерации и перевода.
- `"owlv2"`: Для задач компьютерного зрения.

**Рекомендации**:
- Убедитесь, что модель соответствует задаче (например, `owlv2` только для `VISION`).
- Используйте `device="cpu"` для тестирования, если CUDA недоступен.

### 2. Подготовка датасета
Датасет должен быть в формате `datasets.Dataset` с полями, соответствующими задаче:
- `CLASSIFICATION`: `text` (список строк), `label` (список меток, например, `List[str]` или `List[int]`).
- `TRANSLATION`, `GENERATION`: `text` (список строк), `references` (список строк или список списков строк).
- `VISION`: `text` (список строк), `image` (список изображений `PIL.Image`), `references` (список кортежей `(boxes, labels)` с `torch.Tensor`).

**Пример**:
```python
from datasets import Dataset
import torch

# Датасет для классификации
data = {
    "text": ["This is positive", "This is negative"],
    "label": ["1", "0"]
}
dataset = Dataset.from_dict(data)

# Датасет для vision
vision_data = {
    "text": ["cat", "dog"],
    "image": [img1, img2],  # PIL.Image
    "references": [(torch.tensor([[0, 0, 10, 10]]), torch.tensor([1])), (torch.tensor([[5, 5, 15, 15]]), torch.tensor([2]))]
}
vision_dataset = Dataset.from_dict(vision_data)
```

**Field Mapping**:
- Если поля в датасете имеют другие имена, используйте `field_mapping`:
  ```python
  field_mapping = {"text": "input_text", "label": "target"}
  ```

### 3. Выполнение инференса и оценка метрик
Используйте `metrics_evaluate` для полной оценки.

**Пример для классификации**:
```python
from evaluate_metrics import metrics_evaluate, TaskType

results = metrics_evaluate(
    model_name="gpt2",
    dataset=dataset,
    f_type="classification",
    device="cuda",
    field_mapping={"task_name": "glue_sst2"},  # Для метрики glue
    log=True
)
print(results)  # {'accuracy': 0.85, 'ece': 0.1, 'glue': {'sst2': 0.85}, ...}
```

**Пример для генерации**:
```python
data = {
    "text": ["The capital of France is"],
    "references": [["Paris"]]
}
dataset = Dataset.from_dict(data)
results = metrics_evaluate(
    model_name="gpt2",
    dataset=dataset,
    f_type="generation",
    device="cuda"
)
print(results)  # {'bleu': 0.9, 'rouge': {'rouge-1': 0.95, ...}, 'perplexity': 10.5, ...}
```

**Пример для vision**:
```python
results = metrics_evaluate(
    model_name="owlv2",
    dataset=vision_dataset,
    f_type="vision",
    device="cuda"
)
print(results)  # {'iou': 0.75, 'map': 0.8, 'clip_score_vision': 0.9, ...}
```

### 4. Настройка Task Type и Task Name
- **Task Type**: Выберите из `TaskType`:
  - `TaskType.CLASSIFICATION`: Для задач классификации (например, анализ тональности).
  - `TaskType.TRANSLATION`: Для перевода текста.
  - `TaskType.GENERATION`: Для генерации текста.
  - `TaskType.VISION`: Для детекции объектов.
- **Task Name**: Для метрик `glue`, `helm`, `mmlu` задайте `task_name` через `field_mapping`:
  - `glue`: Например, `"glue_sst2"`, `"glue_mnli"`.
  - `helm`: `"helm"`.
  - `mmlu`: `"mmlu"` или, например, `"mmlu_math"`.

**Пример**:
```python
field_mapping = {"task_name": "glue_sst2"}
results = metrics_evaluate(
    model_name="gpt2",
    dataset=dataset,
    f_type="classification",
    field_mapping=field_mapping
)
```

### 5. Метрики и их особенности
Каждая метрика реализована в отдельном файле (например, `MMLU.py`, `SPICE.py`) и имеет единый интерфейс, принимая параметры из `metrics_data`.

#### Ключевые метрики
- **Текстовые**:
  - `accuracy`: Сравнивает `predictions: List[str]` и `labels: List[str]`.
  - `bleu`, `rouge`, `meteor`, `bert_score`: Сравнивают `predictions: List[str]` и `references: List[List[str]]`.
  - `perplexity`: Использует `model`, `raw_text: List[str]`, `processor`.
  - `cider`, `spice`: Требуют `references: List[List[str]]`, подходят для задач генерации подписей.
- **Классификация**:
  - `ece`, `mce`: Используют `confidences: List[float]` и `labels: List[str]`.
  - `mmlu`, `glue`, `helm`: Требуют специфичный `task_name`.
- **Зрение**:
  - `iou`, `map`, `precision`, `recall`, `f1`: Используют `predictions: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]` и `references: List[Tuple[torch.Tensor, torch.Tensor]]`.
  - `clip_score_vision`: Использует `raw_text: List[str]` и `image_features`.
- **Производительность**:
  - `latency`, `throughput`: Используют `timestamps` и `batch_size`.
  - `memory`: Использует `gpu_memory` и `cpu_memory`.
  - `energy`: Использует `emissions` от `codecarbon`.
  - `flops`: Использует `raw_text` (текстовые задачи) или `text` (зрение).

**Особенности**:
- Метрики, возвращающие словари (`rouge`, `bert_score`, `glue`), сохраняют структуру.
- Скалярные метрики усредняются через `np.nanmean`, игнорируя `None`.

**Пример кастомной метрики**:
Если вы хотите добавить новую метрику, создайте функцию в `metric_functions.py`:
```python
def compute_custom_metric(*, predictions: List[str], references: List[str], **kwargs) -> float:
    return sum(p == r for p, r in zip(predictions, references)) / len(predictions)
```
Добавьте её в `base_metrics` и `TASK_METRICS`.

## Оптимизация и красота кода

- **Читаемость**: Использование понятных имён (`raw_text`, `metrics_data`), подробных докстрингов и комментариев.
- **Гибкость**: Поддержка кастомных полей через `field_mapping` и различных моделей через `MODEL_CLASSES`.
- **Эффективность**: Минимизация памяти с помощью `torch.no_grad()` и точные замеры производительности.
- **Обработка ошибок**: Логирование и исключения обеспечивают устойчивость к сбоям.

## Заключение
Модуль `evaluate_metrics.py` — это мощный инструмент для оценки моделей, сочетающий гибкость, читаемость и производительность. Разделение `raw_text` и `text`, а также кастомизация `task_name` через `field_mapping`, решают ключевые проблемы совместимости и делают фреймворк интуитивным. Пользователи могут легко загружать модели, подготавливать датасеты и вычислять метрики, адаптируя параметры под свои задачи.