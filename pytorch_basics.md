# Основы программирования на PyTorch

## Оглавление


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

Это базовое руководство по программированию на PyTorch, охватывающее основные концепции и операции. Для более глубокого изучения рекомендую обратиться к официальной документации PyTorch и учебным материалам.