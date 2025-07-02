# Основы программирования на PyTorch

## Оглавление

1. [Введение в PyTorch](#введение-в-pytorch)
2. [Установка и настройка](#установка-и-настройка)
3. [Тензоры: основные операции](#тензоры-основные-операции)
4. [Автоматическое дифференцирование](#автоматическое-дифференцирование)
5. [Создание нейронных сетей](#создание-нейронных-сетей)
6. [Загрузка и обработка данных](#загрузка-и-обработка-данных)
7. [Обучение моделей](#обучение-моделей)
8. [Сохранение и загрузка моделей](#сохранение-и-загрузка-моделей)
9. [Перенос на GPU](#перенос-на-gpu)
10. [Оптимизация производительности](#оптимизация-производительности)
11. [Квантование моделей](#квантование-моделей)
12. [Экспорт моделей](#экспорт-моделей)
13. [Архитектура трансформера](#архитектура-трансформера)
14. [Работа с Hugging Face](#работа-с-hugging-face)
15. [Практические задачи](#практические-задачи)

---

## Введение в PyTorch

PyTorch — это библиотека машинного обучения с открытым исходным кодом, разработанная Facebook (Meta). Она предоставляет гибкий и интуитивно понятный интерфейс для создания и обучения нейронных сетей. PyTorch отличается от других фреймворков своим динамическим вычислительным графом, что делает его особенно удобным для исследований и разработки.

Основные преимущества PyTorch:
- **Динамический вычислительный граф**: позволяет изменять структуру сети на лету
- **Императивный стиль программирования**: код выполняется последовательно, что упрощает отладку
- **Интеграция с Python**: естественное взаимодействие с экосистемой Python (NumPy, SciPy и др.)
- **Обширная экосистема**: множество готовых моделей, инструментов и библиотек

## Установка и настройка

### Установка PyTorch

```bash
# Установка с поддержкой CUDA (для GPU NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Установка только для CPU
pip install torch torchvision torchaudio
```

### Проверка установки

```python
import torch

# Проверка версии
print(f"PyTorch version: {torch.__version__}")

# Проверка доступности CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Тензоры: основные операции

Тензоры — это многомерные массивы, являющиеся основным типом данных в PyTorch. Они похожи на массивы NumPy, но могут работать на GPU и автоматически отслеживать градиенты.

### Создание тензоров

```python
import torch
import numpy as np

# Создание тензора из списка
x = torch.tensor([1, 2, 3, 4])
print(x)

# Создание тензора из NumPy массива
np_array = np.array([1, 2, 3, 4])
x = torch.from_numpy(np_array)
print(x)

# Создание тензора заданной формы
x = torch.zeros(3, 4)  # тензор 3x4, заполненный нулями
print(x)

x = torch.ones(2, 3, 4)  # тензор 2x3x4, заполненный единицами
print(x)

x = torch.rand(3, 4)  # тензор 3x4 со случайными значениями от 0 до 1
print(x)

x = torch.randn(3, 4)  # тензор 3x4 со случайными значениями из нормального распределения
print(x)

# Создание тензора с заданным типом данных
x = torch.zeros(3, 4, dtype=torch.float16)  # половинная точность (FP16)
print(x)

x = torch.zeros(3, 4, dtype=torch.int8)  # 8-битные целые числа
print(x)
```

### Операции с тензорами

```python
# Базовые арифметические операции
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)  # сложение
print(a - b)  # вычитание
print(a * b)  # поэлементное умножение
print(a / b)  # деление

# Альтернативный синтаксис
print(torch.add(a, b))
print(torch.sub(a, b))
print(torch.mul(a, b))
print(torch.div(a, b))

# Операции с изменением исходного тензора
a.add_(b)  # a = a + b (in-place операция)
print(a)

# Матричное умножение
a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = torch.matmul(a, b)  # или a @ b
print(c.shape)  # [2, 4]

# Изменение формы тензора
a = torch.randn(12)
b = a.view(3, 4)  # изменение формы на 3x4
c = a.reshape(2, 6)  # альтернативный способ изменения формы

# Транспонирование
a = torch.randn(2, 3)
b = a.t()  # транспонирование 2D тензора
c = a.transpose(0, 1)  # эквивалентно a.t() для 2D

# Конкатенация тензоров
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.cat([a, b], dim=0)  # конкатенация по первой размерности (результат: [4, 3])
d = torch.cat([a, b], dim=1)  # конкатенация по второй размерности (результат: [2, 6])

# Индексация и срезы (как в NumPy)
a = torch.randn(4, 3)
print(a[0])  # первая строка
print(a[:, 1])  # второй столбец
print(a[1:3, :])  # строки с индексами 1 и 2
```

## Автоматическое дифференцирование

Одна из ключевых особенностей PyTorch — автоматическое дифференцирование, которое позволяет автоматически вычислять градиенты функций. Это основа для обучения нейронных сетей.

```python
# Создание тензора с отслеживанием градиента
x = torch.tensor([2.0], requires_grad=True)

# Определение функции y = x^2
y = x ** 2

# Вычисление градиента dy/dx
y.backward()

# Вывод градиента (должен быть 2*x = 4)
print(x.grad)

# Более сложный пример
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = 3 * x ** 2 + 2 * x + 1
y.sum().backward()  # вычисление градиента по сумме выходов
print(x.grad)  # [14., 20.]

# Отключение отслеживания градиентов
with torch.no_grad():
    z = x * 2  # градиент не будет отслеживаться

# Альтернативный способ отключения градиентов
x_detached = x.detach()  # создает новый тензор без истории вычислений
```

## Создание нейронных сетей

PyTorch предоставляет модуль `nn` для создания нейронных сетей. Он содержит готовые слои, функции активации, функции потерь и другие компоненты.

### Определение простой нейронной сети

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Определение класса нейронной сети
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        # Определение слоев
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Определение прямого прохода
        x = self.fc1(x)
        x = F.relu(x)  # применение функции активации ReLU
        x = self.fc2(x)
        return x

# Создание экземпляра модели
model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
print(model)

# Применение модели к входным данным
x = torch.randn(32, 784)  # батч из 32 примеров, каждый размером 784
output = model(x)
print(output.shape)  # [32, 10]
```

### Использование готовых слоев и блоков

```python
# Создание последовательной модели
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Создание сверточной нейронной сети (CNN)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # Сверточные слои
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Слой пулинга
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # x имеет форму [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))  # [batch_size, 32, 28, 28]
        x = self.pool(x)  # [batch_size, 32, 14, 14]
        
        x = F.relu(self.conv2(x))  # [batch_size, 64, 14, 14]
        x = self.pool(x)  # [batch_size, 64, 7, 7]
        
        x = x.view(-1, 64 * 7 * 7)  # [batch_size, 64 * 7 * 7]
        
        x = F.relu(self.fc1(x))  # [batch_size, 128]
        x = self.fc2(x)  # [batch_size, 10]
        
        return x

# Создание рекуррентной нейронной сети (RNN)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        
        # RNN слой
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # Выходной слой
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x имеет форму [batch_size, seq_length, input_size]
        
        # Инициализация скрытого состояния
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Прямой проход через RNN
        out, _ = self.rnn(x, h0)  # out: [batch_size, seq_length, hidden_size]
        
        # Берем выход последнего временного шага
        out = self.fc(out[:, -1, :])  # [batch_size, output_size]
        
        return out
```

## Загрузка и обработка данных

PyTorch предоставляет удобные инструменты для загрузки и предобработки данных через модуль `torch.utils.data`.

### Создание датасетов и загрузчиков данных

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Загрузка встроенного датасета MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Создание загрузчиков данных
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Пример итерации по загрузчику данных
for batch_idx, (data, target) in enumerate(train_loader):
    print(f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
    if batch_idx == 2:
        break

# Создание собственного датасета
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

# Пример использования собственного датасета
import numpy as np

# Создание синтетических данных
data = np.random.randn(100, 10)
targets = np.random.randint(0, 2, size=100)

# Создание датасета и загрузчика
dataset = CustomDataset(data, targets)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

## Обучение моделей

Обучение моделей в PyTorch обычно включает определение функции потерь, оптимизатора и цикла обучения.

### Основной цикл обучения

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Определение модели
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)  # преобразование входа в плоский вектор
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Загрузка данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Создание модели, функции потерь и оптимизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Функция для обучения модели
def train(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()  # переключение модели в режим обучения
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Перенос данных на устройство (CPU/GPU)
            data, target = data.to(device), target.to(device)
            
            # Обнуление градиентов
            optimizer.zero_grad()
            
            # Прямой проход
            output = model(data)
            
            # Вычисление функции потерь
            loss = criterion(output, target)
            
            # Обратное распространение ошибки
            loss.backward()
            
            # Обновление весов
            optimizer.step()
            
            # Статистика
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 99:  # вывод каждые 100 батчей
                print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {running_loss/100:.3f}, Accuracy: {100.*correct/total:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0

# Функция для оценки модели
def evaluate(model, test_loader, criterion, device):
    model.eval()  # переключение модели в режим оценки
    
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # отключение вычисления градиентов
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    return accuracy

# Запуск обучения и оценки
train(model, train_loader, criterion, optimizer, device, epochs=5)
accuracy = evaluate(model, test_loader, criterion, device)
```

### Использование различных оптимизаторов

```python
# SGD (стохастический градиентный спуск)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)

# AdamW (Adam с исправленной регуляризацией весов)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
```

### Планировщики скорости обучения

```python
# Планировщик с пошаговым уменьшением
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Планировщик с уменьшением при достижении плато
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Косинусный планировщик
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Использование планировщика в цикле обучения
for epoch in range(epochs):
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    # Обновление скорости обучения
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()
```

## Сохранение и загрузка моделей

PyTorch предоставляет несколько способов сохранения и загрузки моделей.

```python
# Сохранение и загрузка всей модели
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# Сохранение и загрузка только параметров модели (рекомендуемый способ)
torch.save(model.state_dict(), 'model_weights.pth')
model.load_state_dict(torch.load('model_weights.pth'))

# Сохранение контрольной точки (checkpoint) для возобновления обучения
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    # другая информация, которую нужно сохранить
}
torch.save(checkpoint, 'checkpoint.pth')

# Загрузка контрольной точки
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## Перенос на GPU

PyTorch позволяет легко переносить вычисления на GPU для ускорения.

```python
# Проверка доступности CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Перенос модели на GPU
model = SimpleNN().to(device)

# Перенос тензоров на GPU
x = torch.randn(32, 784).to(device)
y = model(x)  # вычисления будут выполнены на GPU

# Перенос данных на GPU в цикле обучения
for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    # ...

# Перенос модели обратно на CPU (если нужно)
model = model.to('cpu')
```

## Оптимизация производительности

PyTorch предоставляет различные инструменты для оптимизации производительности.

### Использование смешанной точности

```python
from torch.cuda.amp import autocast, GradScaler

# Создание скейлера градиентов
scaler = GradScaler()

# Цикл обучения с использованием смешанной точности
for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    
    # Автоматическое приведение операций к FP16 где это возможно
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Масштабирование градиентов для предотвращения underflow
    scaler.scale(loss).backward()
    
    # Разминирование градиентов и обновление весов
    scaler.step(optimizer)
    
    # Обновление масштаба для следующей итерации
    scaler.update()
```

### Оптимизация загрузки данных

```python
# Использование нескольких рабочих процессов для загрузки данных
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,  # количество подпроцессов для загрузки данных
    pin_memory=True  # ускорение передачи данных на GPU
)
```

### Профилирование кода

```python
from torch.profiler import profile, record_function, ProfilerActivity

# Профилирование модели
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True,
             profile_memory=True) as prof:
    with record_function("model_inference"):
        output = model(input)

# Вывод результатов профилирования
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Квантование моделей

PyTorch предоставляет инструменты для квантования моделей, что позволяет уменьшить их размер и ускорить инференс.

### Динамическое квантование

```python
import torch.quantization

# Определение модели
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Создание и обучение модели
model = SimpleNN()
# ... обучение модели ...

# Динамическое квантование
quantized_model = torch.quantization.quantize_dynamic(
    model,  # модель для квантования
    {nn.Linear},  # типы слоев для квантования
    dtype=torch.qint8  # тип данных для квантования
)

# Сохранение квантованной модели
torch.save(quantized_model.state_dict(), 'quantized_model.pth')
```

### Статическое квантование

```python
import torch.quantization

# Определение модели с поддержкой квантования
class QuantizableNN(nn.Module):
    def __init__(self):
        super(QuantizableNN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.quant(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

# Создание модели
model = QuantizableNN()

# Настройка квантования
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Подготовка модели к квантованию
torch.quantization.prepare(model, inplace=True)

# Калибровка модели на данных
def calibrate(model, data_loader, num_batches=100):
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            model(data)

calibrate(model, train_loader)

# Конвертация модели в квантованную версию
torch.quantization.convert(model, inplace=True)

# Оценка квантованной модели
accuracy = evaluate(model, test_loader, criterion, device='cpu')
```

## Экспорт моделей

PyTorch позволяет экспортировать модели в различные форматы для развертывания.

### Экспорт в TorchScript

```python
# Трассировка модели
example_input = torch.randn(1, 784)
traced_model = torch.jit.trace(model, example_input)

# Сохранение трассированной модели
traced_model.save('traced_model.pt')

# Загрузка трассированной модели
loaded_model = torch.jit.load('traced_model.pt')

# Скриптование модели (альтернативный способ)
scripted_model = torch.jit.script(model)
scripted_model.save('scripted_model.pt')
```

### Экспорт в ONNX

```python
import torch.onnx

# Экспорт модели в формат ONNX
dummy_input = torch.randn(1, 1, 28, 28)  # пример входных данных
torch.onnx.export(
    model,  # модель для экспорта
    dummy_input,  # пример входных данных
    "model.onnx",  # имя выходного файла
    export_params=True,  # сохранить параметры модели в файл
    opset_version=11,  # версия ONNX
    do_constant_folding=True,  # оптимизация констант
    input_names=['input'],  # имена входов
    output_names=['output'],  # имена выходов
    dynamic_axes={'input': {0: 'batch_size'},  # динамические размерности
                  'output': {0: 'batch_size'}}
)
```

---

## Архитектура трансформера

Трансформер — это архитектура нейронной сети, представленная в статье "Attention Is All You Need" (2017), которая произвела революцию в обработке естественного языка и других последовательных данных. В отличие от рекуррентных нейронных сетей, трансформеры используют механизм внимания (attention) для обработки всей последовательности параллельно.

### Основные компоненты трансформера

#### 1. Механизм внимания (Multi-Head Attention)

Механизм внимания позволяет модели фокусироваться на различных частях входной последовательности при генерации каждого элемента выходной последовательности.

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Линейные преобразования
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Вычисление внимания
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        
        # Конкатенация и финальное преобразование
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.wo(output)
        
        return output, attention
```

#### 2. Позиционное кодирование (Positional Encoding)

Поскольку трансформер обрабатывает последовательность параллельно, ему необходима информация о позиции каждого элемента. Позиционное кодирование добавляет эту информацию к входным эмбеддингам.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

#### 3. Нормализация слоя (Layer Normalization)

Нормализация слоя помогает стабилизировать обучение глубоких нейронных сетей.

```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

#### 4. Полносвязная нейронная сеть (Feed-Forward Network)

Каждый блок трансформера содержит полносвязную нейронную сеть, которая применяется к каждой позиции отдельно.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
```

#### 5. Блок энкодера (Encoder Block)

Блок энкодера состоит из механизма внимания и полносвязной нейронной сети с добавлением остаточных соединений и нормализации слоя.

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x
```

#### 6. Блок декодера (Decoder Block)

Блок декодера содержит два механизма внимания: маскированное внимание для обработки выходной последовательности и внимание для связи с выходом энкодера.

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        enc_attn_output, _ = self.encoder_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(enc_attn_output))
        
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x
```

#### 7. Полная архитектура трансформера

Полная архитектура трансформера объединяет все вышеперечисленные компоненты.

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.d_model = d_model
        
    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        
        subsequent_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device), diagonal=1).bool()
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(0)
        
        tgt_mask = tgt_mask & ~subsequent_mask
        return tgt_mask
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src) * math.sqrt(self.d_model)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        output = self.fc(dec_output)
        return output
```

### Оптимизация трансформеров в PyTorch

Для оптимизации трансформеров в PyTorch можно использовать следующие техники:

1. **JIT-компиляция**: Ускорение выполнения модели с помощью компиляции в оптимизированный код.

```python
import torch.jit

# Трассировка модели
model = Transformer(...)
example_src = torch.randint(0, 1000, (16, 32))  # [batch_size, seq_len]
example_tgt = torch.randint(0, 1000, (16, 32))  # [batch_size, seq_len]
traced_model = torch.jit.trace(model, (example_src, example_tgt))
```

2. **Оптимизация графа вычислений**: Использование `torch.fx` для анализа и оптимизации графа вычислений.

```python
import torch.fx

# Символическая трассировка модели
symbolic_traced = torch.fx.symbolic_trace(model)

# Оптимизация графа
optimized_graph = symbolic_traced.graph
# ... оптимизации графа ...

# Создание оптимизированной модели
optimized_model = torch.fx.GraphModule(symbolic_traced, optimized_graph)
```

3. **Квантизация**: Уменьшение точности вычислений для ускорения и уменьшения размера модели.

```python
import torch.quantization

# Динамическая квантизация
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

4. **Распределенное обучение**: Обучение на нескольких GPU с помощью `torch.distributed`.

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(rank, world_size):
    setup(rank, world_size)
    model = Transformer(...).to(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    # ... обучение ...

world_size = torch.cuda.device_count()
mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

## Работа с Hugging Face

Hugging Face — это экосистема библиотек и инструментов для работы с моделями машинного обучения, особенно в области обработки естественного языка. Основная библиотека, `transformers`, предоставляет доступ к предобученным моделям и инструментам для их использования и дообучения.

### Установка Hugging Face Transformers

```bash
pip install transformers
```

Для работы с PyTorch также рекомендуется установить дополнительные зависимости:

```bash
pip install transformers[torch]
```

### Основные компоненты Hugging Face

1. **Модели**: Предобученные модели для различных задач.
2. **Токенизаторы**: Инструменты для преобразования текста в токены.
3. **Конфигурации**: Настройки моделей.
4. **Пайплайны**: Готовые конвейеры для типовых задач.

### Загрузка предобученных моделей

```python
from transformers import AutoModel, AutoTokenizer

# Загрузка модели и токенизатора
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### Токенизация текста

```python
# Токенизация одного предложения
text = "Hello, how are you?"
tokens = tokenizer(text, return_tensors="pt")
print(tokens)

# Токенизация нескольких предложений с паддингом
texts = ["Hello, how are you?", "I'm fine, thank you!"]
tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print(tokens)
```

### Инференс с предобученными моделями

```python
# Получение эмбеддингов
outputs = model(**tokens)
last_hidden_states = outputs.last_hidden_state
pooled_output = outputs.pooler_output

# Для моделей с конкретной задачей (например, классификация)
from transformers import AutoModelForSequenceClassification

classifier = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
outputs = classifier(**tokens)
logits = outputs.logits
predictions = torch.argmax(logits, dim=1)
print(predictions)
```

### Дообучение моделей на собственных данных

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Загрузка данных
dataset = load_dataset("glue", "sst2")

# Функция для токенизации данных
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Настройка обучения
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Создание тренера
trainer = Trainer(
    model=classifier,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Обучение модели
trainer.train()

# Сохранение модели
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
```

### Использование пайплайнов для типовых задач

```python
from transformers import pipeline

# Классификация текста
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Генерация текста
generator = pipeline("text-generation")
result = generator("Once upon a time", max_length=50, num_return_sequences=2)
print(result)

# Заполнение маскированного текста
unmasker = pipeline("fill-mask")
result = unmasker("The man worked as a [MASK].")
print(result)

# Ответы на вопросы
qa = pipeline("question-answering")
context = "Hugging Face is a company based in New York and Paris."
result = qa(question="Where is Hugging Face based?", context=context)
print(result)  # {'answer': 'New York and Paris', 'start': 31, 'end': 49, 'score': 0.9975}
```

### Оптимизация моделей Hugging Face

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Загрузка модели
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Квантизация модели
model.eval()
quantized_model = torch.quantization.quantize_dynamic(
    model,  # модель для квантования
    {torch.nn.Linear},  # типы слоев для квантования
    dtype=torch.qint8  # тип данных для квантования
)

# Экспорт модели в ONNX
input_names = ["input_ids", "attention_mask", "token_type_ids"]
output_names = ["logits"]

dummy_input = tokenizer("This is a test", return_tensors="pt")

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"], dummy_input["token_type_ids"]),
    "model.onnx",
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    },
    opset_version=12,
)
```

### Использование Hugging Face Accelerate для распределённого обучения

```python
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from torch.utils.data import DataLoader

# Инициализация акселератора
accelerator = Accelerator()

# Загрузка модели и данных
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Подготовка данных
train_dataloader = DataLoader(...)
eval_dataloader = DataLoader(...)

# Оптимизатор
optimizer = AdamW(model.parameters(), lr=5e-5)

# Подготовка к распределённому обучению
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Цикл обучения
for epoch in range(3):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

## Практические задачи

### Задача 1: Классификация изображений с использованием предобученной модели

В этой задаче мы будем использовать предобученную модель ResNet для классификации изображений.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Загрузка предобученной модели
model = models.resnet18(pretrained=True)
model.eval()

# Предобработка изображения
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка и предобработка изображения
image_path = "path/to/your/image.jpg"
image = Image.open(image_path)
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # создание батча

# Если доступен GPU, перенос данных на него
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model = model.to('cuda')

# Инференс
with torch.no_grad():
    output = model(input_batch)

# Загрузка меток классов ImageNet
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# Получение топ-5 предсказаний
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

for i in range(5):
    print(f"{classes[top5_catid[i]]}: {top5_prob[i].item()*100:.2f}%")
```

### Задача 2: Генерация текста с использованием GPT-2

В этой задаче мы будем использовать модель GPT-2 для генерации текста.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка модели и токенизатора
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Перенос модели на GPU, если доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Функция для генерации текста
def generate_text(prompt, max_length=100):
    # Токенизация входного текста
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Генерация текста
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True
    )
    
    # Декодирование и вывод результата
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Генерация текста с заданным началом
prompt = "Once upon a time in a land far, far away"
generated_text = generate_text(prompt)
print(generated_text)
```

### Задача 3: Перенос стиля изображения

В этой задаче мы будем использовать нейронные сети для переноса стиля с одного изображения на другое.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

# Загрузка и предобработка изображений
def load_image(image_path, size=None):
    image = Image.open(image_path)
    if size is not None:
        image = image.resize((size, size))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Загрузка изображений
content_img = load_image("path/to/content.jpg", size=512)
style_img = load_image("path/to/style.jpg", size=512)

# Если доступен GPU, перенос данных на него
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_img = content_img.to(device)
style_img = style_img.to(device)

# Загрузка предобученной модели VGG19
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Функция для извлечения признаков из определенных слоев
class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layers = layers
        self._features = {}
        
    def hook(self, layer_name):
        def hook_function(module, input, output):
            self._features[layer_name] = output
        return hook_function
    
    def forward(self, x):
        self._features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                self._features[name] = x
        return self._features

# Определение слоев для извлечения признаков
content_layers = ['19']
style_layers = ['0', '5', '10', '19', '28']

# Создание экстрактора признаков
feature_extractor = FeatureExtractor(vgg, content_layers + style_layers)

# Функция для вычисления функции потерь содержания
def content_loss(target_features, content_features):
    loss = 0
    for layer in content_layers:
        loss += torch.mean((target_features[layer] - content_features[layer]) ** 2)
    return loss

# Функция для вычисления функции потерь стиля
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (c * h * w)

def style_loss(target_features, style_features):
    loss = 0
    for layer in style_layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram = gram_matrix(style_features[layer])
        loss += torch.mean((target_gram - style_gram) ** 2)
    return loss

# Создание изображения для оптимизации
target_img = content_img.clone().requires_grad_(True)

# Оптимизатор
optimizer = optim.LBFGS([target_img])

# Извлечение признаков из изображений содержания и стиля
content_features = feature_extractor(content_img)
style_features = feature_extractor(style_img)

# Веса для функций потерь
content_weight = 1
style_weight = 1000

# Цикл оптимизации
num_steps = 300
for step in range(num_steps):
    def closure():
        optimizer.zero_grad()
        target_features = feature_extractor(target_img)
        
        c_loss = content_loss(target_features, content_features)
        s_loss = style_loss(target_features, style_features)
        
        total_loss = content_weight * c_loss + style_weight * s_loss
        total_loss.backward()
        
        if step % 50 == 0:
            print(f"Step {step}: Content Loss: {c_loss.item()}, Style Loss: {s_loss.item()}")
        
        return total_loss
    
    optimizer.step(closure)

# Преобразование результата обратно в изображение
def tensor_to_image(tensor):
    tensor = tensor.cpu().clone().detach()
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor.squeeze(0))
    return image

# Сохранение результата
result_img = tensor_to_image(target_img)
result_img.save("path/to/result.jpg")
```

### Задача 4: Квантизация модели для ускорения инференса

В этой задаче мы будем квантовать модель для ускорения инференса и уменьшения размера модели.

```python
import torch
import torch.nn as nn
import torch.quantization
import time

# Определение модели с поддержкой квантизации
class QuantizableModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QuantizableModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x

# Создание и обучение модели
input_size = 784
hidden_size = 128
output_size = 10

model = QuantizableModel(input_size, hidden_size, output_size)

# Предположим, что модель уже обучена
# model.train(...)

# Подготовка модели к квантизации
model.eval()

# Создание калибровочного набора данных
calibration_data = torch.randn(100, input_size)

# Измерение времени инференса до квантизации
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        output = model(torch.randn(1, input_size))
fp32_inference_time = time.time() - start_time
print(f"FP32 Inference Time: {fp32_inference_time:.4f} seconds")

# Настройка квантизации
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Калибровка модели
with torch.no_grad():
    for data in calibration_data:
        model(data.unsqueeze(0))

# Квантизация модели
torch.quantization.convert(model, inplace=True)

# Измерение времени инференса после квантизации
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        output = model(torch.randn(1, input_size))
int8_inference_time = time.time() - start_time
print(f"INT8 Inference Time: {int8_inference_time:.4f} seconds")
print(f"Speedup: {fp32_inference_time / int8_inference_time:.2f}x")

# Сохранение квантованной модели
torch.save(model.state_dict(), "quantized_model.pth")

# Сравнение размеров моделей
import os

# Сохранение FP32 модели
torch.save(model.state_dict(), "fp32_model.pth")

fp32_size = os.path.getsize("fp32_model.pth") / (1024 * 1024)  # в МБ
int8_size = os.path.getsize("quantized_model.pth") / (1024 * 1024)  # в МБ

print(f"FP32 Model Size: {fp32_size:.2f} MB")
print(f"INT8 Model Size: {int8_size:.2f} MB")
print(f"Size Reduction: {(1 - int8_size / fp32_size) * 100:.2f}%")
```

Это базовое руководство по программированию на PyTorch, охватывающее основные концепции и операции, а также работу с трансформерами и Hugging Face. Для более глубокого изучения рекомендую обратиться к официальной документации PyTorch и учебным материалам.