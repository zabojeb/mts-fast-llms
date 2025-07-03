# Практические примеры дистилляции знаний

В данном документе представлены практические примеры дистилляции знаний с использованием различных библиотек и фреймворков.

## Содержание

- [Основы дистилляции знаний](#основы-дистилляции-знаний)
- [Дистилляция знаний в PyTorch](#дистилляция-знаний-в-pytorch)
  - [Базовая дистилляция](#базовая-дистилляция)
  - [Дистилляция с использованием промежуточных представлений](#дистилляция-с-использованием-промежуточных-представлений)
- [Дистилляция знаний в Hugging Face](#дистилляция-знаний-в-hugging-face)
  - [Дистилляция BERT](#дистилляция-bert)
  - [Дистилляция GPT](#дистилляция-gpt)
- [Специализированные методы дистилляции](#специализированные-методы-дистилляции)
  - [TinyBERT](#tinybert)
  - [MobileBERT](#mobilebert)
  - [DistilBERT](#distilbert)
- [Оценка эффективности дистилляции](#оценка-эффективности-дистилляции)

## Основы дистилляции знаний

Дистилляция знаний — это процесс передачи знаний от большой модели (учителя) к меньшей модели (ученику). Основная идея заключается в том, что ученик обучается не только на жестких метках (ground truth), но и на мягких выходах учителя, которые содержат более богатую информацию о распределении вероятностей.

### Основные компоненты дистилляции знаний:

1. **Модель-учитель**: предварительно обученная большая модель с высокой точностью
2. **Модель-ученик**: меньшая модель, которую мы хотим обучить
3. **Функция потерь дистилляции**: комбинация потерь на жестких метках и мягких выходах учителя
4. **Температура**: параметр, контролирующий "мягкость" выходов учителя

## Дистилляция знаний в PyTorch

### Базовая дистилляция

Рассмотрим базовую реализацию дистилляции знаний в PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha  # Вес для потери дистилляции
        self.temperature = temperature  # Температура для смягчения выходов
        self.criterion_ce = nn.CrossEntropyLoss()  # Потеря для жестких меток
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')  # Потеря для мягких выходов
    
    def forward(self, outputs_student, outputs_teacher, targets):
        # Потеря на жестких метках
        hard_loss = self.criterion_ce(outputs_student, targets)
        
        # Потеря на мягких выходах (дистилляция)
        soft_student = F.log_softmax(outputs_student / self.temperature, dim=1)
        soft_teacher = F.softmax(outputs_teacher / self.temperature, dim=1)
        soft_loss = self.criterion_kl(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Общая потеря
        loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return loss

# Пример использования
def train_with_distillation(teacher_model, student_model, train_loader, optimizer, epochs=10):
    # Переключение учителя в режим оценки
    teacher_model.eval()
    # Переключение ученика в режим обучения
    student_model.train()
    
    # Создание функции потерь дистилляции
    distillation_loss = DistillationLoss(alpha=0.5, temperature=2.0)
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            # Прямой проход через учителя (без вычисления градиентов)
            with torch.no_grad():
                teacher_outputs = teacher_model(data)
            
            # Прямой проход через ученика
            student_outputs = student_model(data)
            
            # Вычисление потери дистилляции
            loss = distillation_loss(student_outputs, teacher_outputs, target)
            
            # Обратное распространение и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
```

### Дистилляция с использованием промежуточных представлений

Для более эффективной дистилляции можно использовать не только выходы моделей, но и их промежуточные представления:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IntermediateDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, temperature=2.0):
        super(IntermediateDistillationLoss, self).__init__()
        self.alpha = alpha  # Вес для потери дистилляции на выходах
        self.beta = beta  # Вес для потери дистилляции на промежуточных представлениях
        self.temperature = temperature  # Температура для смягчения выходов
        self.criterion_ce = nn.CrossEntropyLoss()  # Потеря для жестких меток
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')  # Потеря для мягких выходов
        self.criterion_mse = nn.MSELoss()  # Потеря для промежуточных представлений
    
    def forward(self, outputs_student, outputs_teacher, intermediates_student, intermediates_teacher, targets):
        # Потеря на жестких метках
        hard_loss = self.criterion_ce(outputs_student, targets)
        
        # Потеря на мягких выходах (дистилляция)
        soft_student = F.log_softmax(outputs_student / self.temperature, dim=1)
        soft_teacher = F.softmax(outputs_teacher / self.temperature, dim=1)
        soft_loss = self.criterion_kl(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Потеря на промежуточных представлениях
        intermediate_loss = 0
        for student_feat, teacher_feat in zip(intermediates_student, intermediates_teacher):
            # Если размеры не совпадают, можно использовать проекцию или адаптивный пулинг
            if student_feat.shape != teacher_feat.shape:
                # Пример адаптивного пулинга для приведения размеров
                adaptive_pool = nn.AdaptiveAvgPool2d((student_feat.size(2), student_feat.size(3)))
                teacher_feat = adaptive_pool(teacher_feat)
            intermediate_loss += self.criterion_mse(student_feat, teacher_feat)
        
        # Общая потеря
        loss = (1 - self.alpha - self.beta) * hard_loss + self.alpha * soft_loss + self.beta * intermediate_loss
        
        return loss

# Пример использования с промежуточными представлениями
def train_with_intermediate_distillation(teacher_model, student_model, train_loader, optimizer, epochs=10):
    teacher_model.eval()
    student_model.train()
    
    distillation_loss = IntermediateDistillationLoss(alpha=0.3, beta=0.3, temperature=2.0)
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            # Прямой проход через учителя (без вычисления градиентов)
            with torch.no_grad():
                teacher_outputs, teacher_intermediates = teacher_model(data, return_intermediates=True)
            
            # Прямой проход через ученика
            student_outputs, student_intermediates = student_model(data, return_intermediates=True)
            
            # Вычисление потери дистилляции с промежуточными представлениями
            loss = distillation_loss(
                student_outputs, teacher_outputs,
                student_intermediates, teacher_intermediates,
                target
            )
            
            # Обратное распространение и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
```

## Дистилляция знаний в Hugging Face

### Дистилляция BERT

Пример дистилляции модели BERT с использованием библиотеки Hugging Face Transformers:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Загрузка предобученной модели-учителя
teacher_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Создание конфигурации для модели-ученика (меньшая модель)
student_config = BertConfig.from_pretrained(
    'bert-base-uncased',
    num_hidden_layers=4,  # Уменьшение количества слоев
    hidden_size=512,      # Уменьшение размера скрытого состояния
    intermediate_size=2048,  # Уменьшение размера промежуточного слоя
    num_attention_heads=8,   # Уменьшение количества голов внимания
    num_labels=2
)

# Создание модели-ученика
student_model = BertForSequenceClassification(student_config)

# Загрузка и подготовка данных
dataset = load_dataset('glue', 'sst2')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Создание класса для дистилляции
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = 2.0
        self.alpha = 0.5
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Получение выходов ученика
        outputs_student = model(**inputs)
        student_logits = outputs_student.logits
        
        # Получение выходов учителя
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            teacher_logits = outputs_teacher.logits
        
        # Потеря на жестких метках
        loss_fct = nn.CrossEntropyLoss()
        hard_loss = loss_fct(student_logits.view(-1, self.model.config.num_labels), inputs['labels'].view(-1))
        
        # Потеря на мягких выходах (дистилляция)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Общая потеря
        loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return (loss, outputs_student) if return_outputs else loss

# Настройка обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# Создание тренера с дистилляцией
trainer = DistillationTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# Обучение модели-ученика
trainer.train()

# Сохранение дистиллированной модели
student_model.save_pretrained('./distilled-bert')
tokenizer.save_pretrained('./distilled-bert')
```

### Дистилляция GPT

Пример дистилляции модели GPT с использованием библиотеки Hugging Face Transformers:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Загрузка предобученной модели-учителя
teacher_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Создание конфигурации для модели-ученика (меньшая модель)
student_config = GPT2Config.from_pretrained(
    'gpt2',
    n_layer=6,           # Уменьшение количества слоев
    n_head=8,            # Уменьшение количества голов внимания
    n_embd=512,          # Уменьшение размера эмбеддингов
    n_inner=2048         # Уменьшение размера промежуточного слоя
)

# Создание модели-ученика
student_model = GPT2LMHeadModel(student_config)

# Загрузка и подготовка данных
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Создание класса для дистилляции
class GPTDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = 2.0
        self.alpha = 0.5
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Получение выходов ученика
        outputs_student = model(**inputs)
        student_logits = outputs_student.logits
        
        # Получение выходов учителя
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            teacher_logits = outputs_teacher.logits
        
        # Потеря на жестких метках (стандартная потеря языкового моделирования)
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        hard_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Потеря на мягких выходах (дистилляция)
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        soft_student = F.log_softmax(shift_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(shift_teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student.view(-1, soft_student.size(-1)), 
                            soft_teacher.view(-1, soft_teacher.size(-1)), 
                            reduction='batchmean') * (self.temperature ** 2)
        
        # Общая потеря
        loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return (loss, outputs_student) if return_outputs else loss

# Настройка обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# Создание тренера с дистилляцией
trainer = GPTDistillationTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# Обучение модели-ученика
trainer.train()

# Сохранение дистиллированной модели
student_model.save_pretrained('./distilled-gpt2')
tokenizer.save_pretrained('./distilled-gpt2')
```

## Специализированные методы дистилляции

### TinyBERT

TinyBERT — это метод дистилляции, который использует двухэтапный процесс обучения и дистилляцию на уровне внимания и скрытых состояний:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, AutoTokenizer

# Загрузка предобученной модели-учителя
teacher_model = BertModel.from_pretrained('bert-base-uncased')

# Создание конфигурации для TinyBERT
tiny_config = BertConfig.from_pretrained(
    'bert-base-uncased',
    num_hidden_layers=4,  # Уменьшение количества слоев
    hidden_size=312,      # Уменьшение размера скрытого состояния
    intermediate_size=1200,  # Уменьшение размера промежуточного слоя
    num_attention_heads=12   # Сохранение количества голов внимания
)

# Создание модели TinyBERT
tiny_bert = BertModel(tiny_config)

# Функция потерь для дистилляции внимания
def attention_distillation_loss(student_attentions, teacher_attentions):
    loss = 0
    for student_att, teacher_att in zip(student_attentions, teacher_attentions):
        # Приведение размеров, если необходимо
        if student_att.shape != teacher_att.shape:
            # Пример: изменение размера с помощью линейного слоя
            projection = nn.Linear(student_att.size(-1), teacher_att.size(-1)).to(student_att.device)
            student_att = projection(student_att)
        
        # MSE потеря для матриц внимания
        loss += F.mse_loss(student_att, teacher_att)
    
    return loss / len(student_attentions)

# Функция потерь для дистилляции скрытых состояний
def hidden_state_distillation_loss(student_hidden_states, teacher_hidden_states):
    loss = 0
    for student_hs, teacher_hs in zip(student_hidden_states, teacher_hidden_states):
        # Приведение размеров, если необходимо
        if student_hs.shape != teacher_hs.shape:
            # Пример: изменение размера с помощью линейного слоя
            projection = nn.Linear(student_hs.size(-1), teacher_hs.size(-1)).to(student_hs.device)
            student_hs = projection(student_hs)
        
        # MSE потеря для скрытых состояний
        loss += F.mse_loss(student_hs, teacher_hs)
    
    return loss / len(student_hidden_states)

# Пример обучения TinyBERT
def train_tinybert(teacher_model, student_model, train_loader, optimizer, epochs=10):
    teacher_model.eval()
    student_model.train()
    
    for epoch in range(epochs):
        for batch_idx, (input_ids, attention_mask, token_type_ids) in enumerate(train_loader):
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
            
            # Прямой проход через учителя (без вычисления градиентов)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True,
                    output_hidden_states=True
                )
                teacher_attentions = teacher_outputs.attentions
                teacher_hidden_states = teacher_outputs.hidden_states
            
            # Прямой проход через ученика
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=True,
                output_hidden_states=True
            )
            student_attentions = student_outputs.attentions
            student_hidden_states = student_outputs.hidden_states
            
            # Вычисление потерь дистилляции
            att_loss = attention_distillation_loss(student_attentions, teacher_attentions)
            hid_loss = hidden_state_distillation_loss(student_hidden_states, teacher_hidden_states)
            
            # Общая потеря
            loss = att_loss + hid_loss
            
            # Обратное распространение и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}, '
                      f'Att Loss: {att_loss.item()}, Hid Loss: {hid_loss.item()}')
```

### MobileBERT

MobileBERT — это метод дистилляции, который использует архитектуру с узкими слоями и глубокими блоками:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, AutoTokenizer

# Создание конфигурации для MobileBERT
mobile_config = BertConfig.from_pretrained(
    'bert-base-uncased',
    num_hidden_layers=24,  # Увеличение количества слоев
    hidden_size=128,       # Уменьшение размера скрытого состояния
    intermediate_size=512, # Уменьшение размера промежуточного слоя
    num_attention_heads=4, # Уменьшение количества голов внимания
    use_bottleneck=True,   # Использование бутылочного горлышка
    intra_bottleneck_size=512  # Размер внутреннего бутылочного горлышка
)

# Создание модели MobileBERT
# Примечание: стандартная реализация BertModel не поддерживает все параметры MobileBERT,
# поэтому это упрощенный пример. В реальности потребуется кастомная реализация.
mobile_bert = BertModel(mobile_config)

# Функция потерь для дистилляции с использованием бутылочного горлышка
def bottleneck_distillation_loss(student_features, teacher_features, bottleneck_transforms):
    loss = 0
    for student_feat, teacher_feat, transform in zip(student_features, teacher_features, bottleneck_transforms):
        # Преобразование признаков ученика через бутылочное горлышко
        transformed_student = transform(student_feat)
        
        # MSE потеря между преобразованными признаками ученика и признаками учителя
        loss += F.mse_loss(transformed_student, teacher_feat)
    
    return loss / len(student_features)

# Пример обучения MobileBERT
def train_mobilebert(teacher_model, student_model, bottleneck_transforms, train_loader, optimizer, epochs=10):
    teacher_model.eval()
    student_model.train()
    
    for epoch in range(epochs):
        for batch_idx, (input_ids, attention_mask, token_type_ids) in enumerate(train_loader):
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
            
            # Прямой проход через учителя (без вычисления градиентов)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=True
                )
                teacher_hidden_states = teacher_outputs.hidden_states
            
            # Прямой проход через ученика
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True
            )
            student_hidden_states = student_outputs.hidden_states
            
            # Вычисление потери дистилляции с использованием бутылочного горлышка
            bottleneck_loss = bottleneck_distillation_loss(
                student_hidden_states, 
                teacher_hidden_states, 
                bottleneck_transforms
            )
            
            # Обратное распространение и оптимизация
            optimizer.zero_grad()
            bottleneck_loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {bottleneck_loss.item()}')
```

### DistilBERT

DistilBERT — это метод дистилляции, который использует тройную потерю: дистилляцию, маскированное языковое моделирование и косинусное сходство скрытых состояний:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer

# Загрузка предобученной модели-учителя
teacher_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Создание конфигурации для DistilBERT
distil_config = BertConfig.from_pretrained(
    'bert-base-uncased',
    num_hidden_layers=6,  # Уменьшение количества слоев в 2 раза
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

# Создание модели DistilBERT
distil_bert = BertForMaskedLM(distil_config)

# Класс для тройной потери DistilBERT
class DistilBERTLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1, temperature=2.0):
        super(DistilBERTLoss, self).__init__()
        self.alpha = alpha  # Вес для потери дистилляции
        self.beta = beta    # Вес для потери косинусного сходства
        self.temperature = temperature  # Температура для смягчения выходов
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)  # Потеря для MLM
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')  # Потеря для дистилляции
        self.cosine_loss = nn.CosineEmbeddingLoss()  # Потеря для косинусного сходства
    
    def forward(self, student_logits, teacher_logits, student_hidden, teacher_hidden, masked_lm_labels):
        # Потеря маскированного языкового моделирования
        mlm_loss = self.criterion_ce(student_logits.view(-1, student_logits.size(-1)), masked_lm_labels.view(-1))
        
        # Потеря дистилляции
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        distil_loss = self.criterion_kl(soft_student.view(-1, soft_student.size(-1)), 
                                      soft_teacher.view(-1, soft_teacher.size(-1))) * (self.temperature ** 2)
        
        # Потеря косинусного сходства для скрытых состояний
        # Создание целевого тензора для косинусного сходства (все 1, т.е. максимальное сходство)
        target = torch.ones(student_hidden.size(0)).to(student_hidden.device)
        cos_loss = self.cosine_loss(
            student_hidden.view(student_hidden.size(0), -1),
            teacher_hidden.view(teacher_hidden.size(0), -1),
            target
        )
        
        # Общая потеря
        loss = mlm_loss + self.alpha * distil_loss + self.beta * cos_loss
        
        return loss, mlm_loss, distil_loss, cos_loss

# Пример обучения DistilBERT
def train_distilbert(teacher_model, student_model, train_loader, optimizer, epochs=10):
    teacher_model.eval()
    student_model.train()
    
    distil_loss_fn = DistilBERTLoss(alpha=0.5, beta=0.1, temperature=2.0)
    
    for epoch in range(epochs):
        for batch_idx, (input_ids, attention_mask, masked_lm_labels) in enumerate(train_loader):
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            masked_lm_labels = masked_lm_labels.cuda()
            
            # Прямой проход через учителя (без вычисления градиентов)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=masked_lm_labels,
                    output_hidden_states=True
                )
                teacher_logits = teacher_outputs.logits
                teacher_hidden = teacher_outputs.hidden_states[-1]  # Последнее скрытое состояние
            
            # Прямой проход через ученика
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=masked_lm_labels,
                output_hidden_states=True
            )
            student_logits = student_outputs.logits
            student_hidden = student_outputs.hidden_states[-1]  # Последнее скрытое состояние
            
            # Вычисление тройной потери
            loss, mlm_loss, distil_loss, cos_loss = distil_loss_fn(
                student_logits, teacher_logits,
                student_hidden, teacher_hidden,
                masked_lm_labels
            )
            
            # Обратное распространение и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}, '
                      f'MLM Loss: {mlm_loss.item()}, Distil Loss: {distil_loss.item()}, '
                      f'Cos Loss: {cos_loss.item()}')
```

## Оценка эффективности дистилляции

После дистилляции важно оценить эффективность полученной модели по нескольким параметрам:

```python
import torch
import time
import numpy as np
from transformers import BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# Загрузка моделей для сравнения
teacher_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
student_model = BertForSequenceClassification.from_pretrained('./distilled-bert')

# Загрузка тестового набора данных
dataset = load_dataset('glue', 'sst2')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
test_dataset = tokenized_datasets['validation']

# Функция для оценки модели
def evaluate_model(model, test_dataloader):
    model.eval()
    device = next(model.parameters()).device
    
    all_preds = []
    all_labels = []
    total_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Измерение времени инференса
            start_time = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_samples += input_ids.size(0)
            
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Вычисление метрик
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_inference_time = total_time / total_samples
    
    # Вычисление размера модели
    model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # в МБ (для FP32)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'avg_inference_time': avg_inference_time,
        'model_size': model_size
    }

# Создание загрузчика данных
from torch.utils.data import DataLoader
from transformers import default_data_collator

test_dataloader = DataLoader(
    test_dataset,
    batch_size=32,
    collate_fn=default_data_collator
)

# Оценка моделей
teacher_metrics = evaluate_model(teacher_model, test_dataloader)
student_metrics = evaluate_model(student_model, test_dataloader)

# Вывод результатов
print("Teacher Model Metrics:")
print(f"Accuracy: {teacher_metrics['accuracy']:.4f}")
print(f"F1 Score: {teacher_metrics['f1_score']:.4f}")
print(f"Avg Inference Time: {teacher_metrics['avg_inference_time'] * 1000:.2f} ms")
print(f"Model Size: {teacher_metrics['model_size']:.2f} MB")

print("\nStudent Model Metrics:")
print(f"Accuracy: {student_metrics['accuracy']:.4f}")
print(f"F1 Score: {student_metrics['f1_score']:.4f}")
print(f"Avg Inference Time: {student_metrics['avg_inference_time'] * 1000:.2f} ms")
print(f"Model Size: {student_metrics['model_size']:.2f} MB")

print("\nComparison:")
print(f"Accuracy Retention: {student_metrics['accuracy'] / teacher_metrics['accuracy'] * 100:.2f}%")
print(f"Speed Improvement: {teacher_metrics['avg_inference_time'] / student_metrics['avg_inference_time']:.2f}x")
print(f"Size Reduction: {teacher_metrics['model_size'] / student_metrics['model_size']:.2f}x")
```

## Сравнение методов дистилляции

| Метод | Размер модели | Количество параметров | Скорость инференса | Сохранение точности | Особенности |
|-------|--------------|----------------------|-------------------|---------------------|-------------|
| BERT-base | 418 МБ | 110M | 1x | 100% | Базовая модель |
| DistilBERT | 257 МБ | 66M | 1.6x | 97% | Тройная потеря (дистилляция, MLM, косинусное сходство) |
| TinyBERT | 55 МБ | 14.5M | 9.4x | 96% | Дистилляция на уровне внимания и скрытых состояний |
| MobileBERT | 95 МБ | 25.3M | 4x | 99.5% | Архитектура с узкими слоями и глубокими блоками |
| BERT-large | 1.34 ГБ | 340M | 0.5x | 104% | Большая модель для сравнения |

## Практические рекомендации

1. **Выбор архитектуры ученика**:
   - Начните с уменьшения количества слоев (обычно в 2-3 раза)
   - Уменьшите размер скрытого состояния (обычно в 1.5-2 раза)
   - Уменьшите количество голов внимания пропорционально размеру скрытого состояния

2. **Настройка процесса дистилляции**:
   - Используйте более высокую скорость обучения для ученика, чем для учителя
   - Экспериментируйте с температурой (обычно в диапазоне 2-5)
   - Балансируйте веса различных компонентов потери

3. **Данные для дистилляции**:
   - Используйте большой набор данных без меток (для дистилляции мягких выходов)
   - Дополните данными с метками для задачи (для потери на жестких метках)
   - Рассмотрите возможность использования синтетических данных, сгенерированных учителем

4. **Многоэтапная дистилляция**:
   - Начните с общей дистилляции (на уровне выходов)
   - Продолжите с дистилляцией промежуточных представлений
   - Завершите тонкой настройкой на конкретной задаче

5. **Оценка и выбор модели**:
   - Оценивайте не только точность, но и скорость инференса и размер модели
   - Используйте метрику, учитывающую компромисс между точностью и эффективностью
   - Тестируйте на различных устройствах, если модель предназначена для мобильных или edge-устройств