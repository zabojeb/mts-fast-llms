# Практические примеры бинаризации нейронных сетей

В данном документе представлены практические примеры бинаризации нейронных сетей с использованием различных библиотек и фреймворков.

## Содержание

- [Основы бинаризации нейронных сетей](#основы-бинаризации-нейронных-сетей)
- [Бинаризация в PyTorch](#бинаризация-в-pytorch)
  - [Бинарные нейронные сети (BNN)](#бинарные-нейронные-сети-bnn)
  - [XNOR-сети](#xnor-сети)
  - [Бинаризация с сохранением градиентов (STE)](#бинаризация-с-сохранением-градиентов-ste)
- [Бинаризация в TensorFlow/Keras](#бинаризация-в-tensorflowkeras)
  - [Бинарные слои в Keras](#бинарные-слои-в-keras)
  - [Квантизация до 1 бита](#квантизация-до-1-бита)
- [Специализированные библиотеки для бинаризации](#специализированные-библиотеки-для-бинаризации)
  - [Larq](#larq)
  - [Brevitas](#brevitas)
- [Оценка эффективности бинаризации](#оценка-эффективности-бинаризации)

## Основы бинаризации нейронных сетей

Бинаризация нейронных сетей — это процесс преобразования весов и/или активаций нейронной сети в бинарные значения (обычно -1 и +1 или 0 и 1). Это позволяет значительно уменьшить размер модели и ускорить вычисления, заменяя операции с плавающей точкой на битовые операции.

### Основные подходы к бинаризации:

1. **Бинаризация весов**: только веса сети преобразуются в бинарные значения, активации остаются в формате с плавающей точкой
2. **Бинаризация активаций**: только активации преобразуются в бинарные значения, веса остаются в формате с плавающей точкой
3. **Полная бинаризация**: и веса, и активации преобразуются в бинарные значения

### Преимущества бинаризации:

- Уменьшение размера модели (до 32 раз по сравнению с FP32)
- Ускорение вычислений (замена операций с плавающей точкой на битовые операции)
- Снижение энергопотребления (особенно важно для мобильных и edge-устройств)

### Недостатки бинаризации:

- Снижение точности модели
- Сложности при обучении (проблемы с распространением градиентов)
- Не все архитектуры хорошо поддаются бинаризации

## Бинаризация в PyTorch

### Бинарные нейронные сети (BNN)

Реализация бинарного линейного слоя в PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Параметры с плавающей точкой (для обучения)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Инициализация параметров
        nn.init.uniform_(self.weight, -1, 1)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, input):
        # Бинаризация весов во время прямого прохода
        binary_weight = torch.sign(self.weight)
        # Замена нулей на единицы (если есть)
        binary_weight = torch.where(binary_weight == 0, torch.ones_like(binary_weight), binary_weight)
        
        # Линейное преобразование с бинарными весами
        output = F.linear(input, binary_weight, self.bias)
        return output
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
```

Реализация бинарного сверточного слоя:

```python
class BinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BinaryConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Параметры с плавающей точкой (для обучения)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Инициализация параметров
        nn.init.uniform_(self.weight, -1, 1)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, input):
        # Бинаризация весов во время прямого прохода
        binary_weight = torch.sign(self.weight)
        # Замена нулей на единицы (если есть)
        binary_weight = torch.where(binary_weight == 0, torch.ones_like(binary_weight), binary_weight)
        
        # Свертка с бинарными весами
        output = F.conv2d(input, binary_weight, self.bias, self.stride, self.padding)
        return output
    
    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None}'
```

Реализация бинарной функции активации:

```python
class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()
    
    def forward(self, input):
        # Бинаризация активаций во время прямого прохода
        output = torch.sign(input)
        # Замена нулей на единицы (если есть)
        output = torch.where(output == 0, torch.ones_like(output), output)
        return output
```

Пример создания полностью бинарной нейронной сети:

```python
class BinaryNeuralNetwork(nn.Module):
    def __init__(self):
        super(BinaryNeuralNetwork, self).__init__()
        
        # Первый слой (обычно не бинаризуется для сохранения точности)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Бинарные слои
        self.bin_act = BinaryActivation()
        self.bin_conv2 = BinaryConv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.bin_conv3 = BinaryConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Полносвязные слои
        self.flatten = nn.Flatten()
        self.bin_fc1 = BinaryLinear(64 * 8 * 8, 512)  # Предполагается, что размер входа 8x8 после сверток и пулинга
        self.bn4 = nn.BatchNorm1d(512)
        self.bin_fc2 = BinaryLinear(512, 10)  # 10 классов
    
    def forward(self, x):
        # Первый слой (не бинарный)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        
        # Бинарные слои
        x = self.bin_act(self.bn2(self.bin_conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.bin_act(self.bn3(self.bin_conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Полносвязные слои
        x = self.flatten(x)
        x = self.bin_act(self.bn4(self.bin_fc1(x)))
        x = self.bin_fc2(x)
        
        return x
```

### XNOR-сети

XNOR-сети — это разновидность бинарных нейронных сетей, которые используют операцию XNOR для замены операций умножения, что позволяет еще больше ускорить вычисления.

```python
class XNORConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(XNORConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Параметры с плавающей точкой (для обучения)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Инициализация параметров
        nn.init.uniform_(self.weight, -1, 1)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, input):
        # Бинаризация входа
        input_norm = torch.mean(torch.abs(input), dim=(1, 2, 3), keepdim=True)
        input_binary = torch.sign(input) * input_norm
        
        # Бинаризация весов
        weight_norm = torch.mean(torch.abs(self.weight), dim=(1, 2, 3), keepdim=True)
        weight_binary = torch.sign(self.weight) * weight_norm
        
        # Свертка с бинарными весами и входами
        output = F.conv2d(input_binary, weight_binary, self.bias, self.stride, self.padding)
        
        return output
    
    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None}'
```

### Бинаризация с сохранением градиентов (STE)

Проблема с бинаризацией заключается в том, что функция sign не имеет полезного градиента для обратного распространения. Метод Straight-Through Estimator (STE) позволяет обойти эту проблему, используя суррогатный градиент.

```python
class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.sign(input)
        # Замена нулей на единицы (если есть)
        output = torch.where(output == 0, torch.ones_like(output), output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Суррогатный градиент: пропускаем градиент только для входов в диапазоне [-1, 1]
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

# Использование функции бинаризации с STE
binarize = BinarizeFunction.apply

class BinaryLinearSTE(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinearSTE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Параметры с плавающей точкой (для обучения)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Инициализация параметров
        nn.init.uniform_(self.weight, -1, 1)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, input):
        # Бинаризация весов с использованием STE
        binary_weight = binarize(self.weight)
        
        # Линейное преобразование с бинарными весами
        output = F.linear(input, binary_weight, self.bias)
        return output
```

## Бинаризация в TensorFlow/Keras

### Бинарные слои в Keras

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

# Функция бинаризации с STE
def binarize(x):
    # Прямой проход: бинаризация
    bin_x = tf.sign(x)
    # Замена нулей на единицы (если есть)
    bin_x = tf.where(tf.equal(bin_x, 0), tf.ones_like(bin_x), bin_x)
    
    # Обратный проход: суррогатный градиент
    return bin_x + tf.stop_gradient(bin_x - x)

# Бинарный сверточный слой
class BinaryConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', use_bias=True, **kwargs):
        super(BinaryConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.upper()
        self.use_bias = use_bias
    
    def build(self, input_shape):
        input_channels = input_shape[-1]
        kernel_shape = self.kernel_size + (input_channels, self.filters)
        
        # Создание весов
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer='zeros',
                trainable=True
            )
        
        super(BinaryConv2D, self).build(input_shape)
    
    def call(self, inputs):
        # Бинаризация весов
        binary_kernel = binarize(self.kernel)
        
        # Свертка с бинарными весами
        outputs = tf.nn.conv2d(
            inputs,
            binary_kernel,
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding
        )
        
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        
        return outputs
    
    def get_config(self):
        config = super(BinaryConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding.lower(),
            'use_bias': self.use_bias
        })
        return config

# Бинарная функция активации
class BinaryActivation(layers.Layer):
    def __init__(self, **kwargs):
        super(BinaryActivation, self).__init__(**kwargs)
    
    def call(self, inputs):
        return binarize(inputs)
    
    def get_config(self):
        return super(BinaryActivation, self).get_config()

# Пример создания бинарной сети в Keras
def create_binary_model():
    model = models.Sequential([
        # Первый слой (не бинарный)
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Бинарные слои
        BinaryActivation(),
        BinaryConv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        BinaryActivation(),
        BinaryConv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Полносвязные слои
        layers.Flatten(),
        BinaryActivation(),
        layers.Dense(512, use_bias=False),
        layers.BatchNormalization(),
        BinaryActivation(),
        layers.Dense(10)  # Выходной слой (не бинарный)
    ])
    
    return model
```

### Квантизация до 1 бита

TensorFlow также предоставляет инструменты для квантизации моделей до 1 бита, что эквивалентно бинаризации:

```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Создание обычной модели
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10)
    ])
    
    return model

# Квантизация модели до 1 бита
def quantize_model(model):
    # Определение квантизационной схемы
    quantization_scheme = tfmot.quantization.keras.quantize_annotate_layer
    
    # Аннотирование слоев для квантизации
    annotated_model = tf.keras.Sequential([
        quantization_scheme(layer) if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)) else layer
        for layer in model.layers
    ])
    
    # Настройка квантизации до 1 бита
    quantization_config = tfmot.quantization.keras.QuantizeConfig(
        weight_quantizer=tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=1, symmetric=True, narrow_range=False
        ),
        activation_quantizer=tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=1, symmetric=True, narrow_range=False
        ),
        quantize_output=True
    )
    
    # Применение квантизации
    quantized_model = tfmot.quantization.keras.quantize_apply(annotated_model, quantization_config)
    
    return quantized_model

# Создание и квантизация модели
model = create_model()
quantized_model = quantize_model(model)

# Компиляция и обучение квантизованной модели
quantized_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Обучение модели
# quantized_model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
```

## Специализированные библиотеки для бинаризации

### Larq

Larq — это библиотека для TensorFlow/Keras, специально разработанная для бинарных нейронных сетей:

```python
import tensorflow as tf
import larq as lq

# Создание бинарной модели с использованием Larq
def create_binary_model_larq():
    model = tf.keras.Sequential([
        # Первый слой (не бинарный)
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Бинарные слои
        lq.layers.QuantConv2D(
            64, (3, 3), padding='same',
            input_quantizer=lq.quantizers.SteSign(clip_value=1.0),
            kernel_quantizer=lq.quantizers.SteSign(clip_value=1.0),
            kernel_constraint=lq.constraints.WeightClip(clip_value=1.0),
            use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        lq.layers.QuantConv2D(
            128, (3, 3), padding='same',
            input_quantizer=lq.quantizers.SteSign(clip_value=1.0),
            kernel_quantizer=lq.quantizers.SteSign(clip_value=1.0),
            kernel_constraint=lq.constraints.WeightClip(clip_value=1.0),
            use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Полносвязные слои
        tf.keras.layers.Flatten(),
        lq.layers.QuantDense(
            512,
            input_quantizer=lq.quantizers.SteSign(clip_value=1.0),
            kernel_quantizer=lq.quantizers.SteSign(clip_value=1.0),
            kernel_constraint=lq.constraints.WeightClip(clip_value=1.0),
            use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10)  # Выходной слой (не бинарный)
    ])
    
    return model

# Создание модели
model = create_binary_model_larq()

# Компиляция модели
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Вывод сводки модели с информацией о бинаризации
lq.models.summary(model)

# Обучение модели
# model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Конвертация модели для эффективного инференса
model_converter = lq.models.converter.ModelConverter(model)
binary_model = model_converter.convert()

# Сохранение бинарной модели
binary_model.save('binary_model.h5')
```

### Brevitas

Brevitas — это библиотека для PyTorch, которая предоставляет инструменты для квантизации и бинаризации нейронных сетей:

```python
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int1Bias as Bin
from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.restrict_val import RestrictValueType

# Создание бинарной модели с использованием Brevitas
class BinaryNetBrevitas(nn.Module):
    def __init__(self, num_classes=10):
        super(BinaryNetBrevitas, self).__init__()
        
        # Первый слой (не бинарный)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Бинарные слои
        self.bin_conv2 = qnn.QuantConv2d(
            32, 64, kernel_size=3, stride=1, padding=1,
            weight_bit_width=1,
            weight_quant_type=QuantType.BINARY,
            weight_scaling_impl_type=ScalingImplType.STATS,
            weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            weight_scaling_stats_op='max',
            return_quant_tensor=True
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.bin_act2 = qnn.QuantReLU(
            bit_width=1,
            quant_type=QuantType.BINARY,
            scaling_impl_type=ScalingImplType.STATS,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_stats_op='max',
            return_quant_tensor=True
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.bin_conv3 = qnn.QuantConv2d(
            64, 128, kernel_size=3, stride=1, padding=1,
            weight_bit_width=1,
            weight_quant_type=QuantType.BINARY,
            weight_scaling_impl_type=ScalingImplType.STATS,
            weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            weight_scaling_stats_op='max',
            return_quant_tensor=True
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.bin_act3 = qnn.QuantReLU(
            bit_width=1,
            quant_type=QuantType.BINARY,
            scaling_impl_type=ScalingImplType.STATS,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_stats_op='max',
            return_quant_tensor=True
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Полносвязные слои
        self.flatten = nn.Flatten()
        self.bin_fc1 = qnn.QuantLinear(
            128 * 4 * 4, 512,  # Предполагается, что размер входа 4x4 после сверток и пулинга
            bias=False,
            weight_bit_width=1,
            weight_quant_type=QuantType.BINARY,
            weight_scaling_impl_type=ScalingImplType.STATS,
            weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            weight_scaling_stats_op='max',
            return_quant_tensor=True
        )
        self.bn4 = nn.BatchNorm1d(512)
        self.bin_act4 = qnn.QuantReLU(
            bit_width=1,
            quant_type=QuantType.BINARY,
            scaling_impl_type=ScalingImplType.STATS,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            scaling_stats_op='max',
            return_quant_tensor=True
        )
        self.fc2 = nn.Linear(512, num_classes)  # Выходной слой (не бинарный)
    
    def forward(self, x):
        # Первый слой (не бинарный)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        
        # Бинарные слои
        x = self.bin_act2(self.bn2(self.bin_conv2(x)))
        x = self.pool2(x)
        x = self.bin_act3(self.bn3(self.bin_conv3(x)))
        x = self.pool3(x)
        
        # Полносвязные слои
        x = self.flatten(x)
        x = self.bin_act4(self.bn4(self.bin_fc1(x)))
        x = self.fc2(x)
        
        return x

# Создание модели
model = BinaryNetBrevitas(num_classes=10)

# Перемещение модели на GPU, если доступно
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.3f}, Accuracy: {100 * correct / total:.2f}%')
                running_loss = 0.0

# Экспорт модели в ONNX
def export_onnx(model, dummy_input, file_path):
    torch.onnx.export(
        model,
        dummy_input,
        file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f'Model exported to {file_path}')

# Экспорт модели
# dummy_input = torch.randn(1, 3, 32, 32).to(device)
# export_onnx(model, dummy_input, 'binary_model.onnx')
```

## Оценка эффективности бинаризации

После бинаризации важно оценить эффективность полученной модели по нескольким параметрам:

```python
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

# Функция для оценки модели
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    inference_times = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Измерение времени инференса
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Вычисление метрик
    accuracy = 100 * correct / total
    avg_inference_time = np.mean(inference_times) * 1000  # в миллисекундах
    
    # Вычисление размера модели
    model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # в МБ (для FP32)
    binary_size = sum(p.numel() for p in model.parameters()) / 8 / (1024 * 1024)  # в МБ (для бинарной модели)
    
    return {
        'accuracy': accuracy,
        'avg_inference_time': avg_inference_time,
        'model_size_fp32': model_size,
        'model_size_binary': binary_size
    }

# Сравнение обычной и бинарной моделей
def compare_models(fp32_model, binary_model, test_loader, device):
    fp32_metrics = evaluate_model(fp32_model, test_loader, device)
    binary_metrics = evaluate_model(binary_model, test_loader, device)
    
    # Вывод результатов
    print("FP32 Model Metrics:")
    print(f"Accuracy: {fp32_metrics['accuracy']:.2f}%")
    print(f"Avg Inference Time: {fp32_metrics['avg_inference_time']:.2f} ms")
    print(f"Model Size: {fp32_metrics['model_size_fp32']:.2f} MB")
    
    print("\nBinary Model Metrics:")
    print(f"Accuracy: {binary_metrics['accuracy']:.2f}%")
    print(f"Avg Inference Time: {binary_metrics['avg_inference_time']:.2f} ms")
    print(f"Model Size: {binary_metrics['model_size_binary']:.2f} MB")
    
    print("\nComparison:")
    print(f"Accuracy Retention: {binary_metrics['accuracy'] / fp32_metrics['accuracy'] * 100:.2f}%")
    print(f"Speed Improvement: {fp32_metrics['avg_inference_time'] / binary_metrics['avg_inference_time']:.2f}x")
    print(f"Size Reduction: {fp32_metrics['model_size_fp32'] / binary_metrics['model_size_binary']:.2f}x")
    
    # Визуализация результатов
    labels = ['Accuracy (%)', 'Inference Time (ms)', 'Model Size (MB)']
    fp32_values = [fp32_metrics['accuracy'], fp32_metrics['avg_inference_time'], fp32_metrics['model_size_fp32']]
    binary_values = [binary_metrics['accuracy'], binary_metrics['avg_inference_time'], binary_metrics['model_size_binary']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, fp32_values, width, label='FP32 Model')
    rects2 = ax.bar(x + width/2, binary_values, width, label='Binary Model')
    
    ax.set_ylabel('Values')
    ax.set_title('Comparison of FP32 and Binary Models')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
```

## Сравнение методов бинаризации

| Метод | Тип бинаризации | Потеря точности | Ускорение вычислений | Уменьшение размера | Сложность реализации | Поддержка в PyTorch | Поддержка в TensorFlow |
|-------|----------------|-----------------|----------------------|-------------------|----------------------|---------------------|-------------------------|
| BNN | Веса и активации | Высокая (10-20%) | 3-7x | 32x | Средняя | Ручная реализация | Ручная реализация |
| XNOR-Net | Веса и активации | Средняя (5-15%) | 4-10x | 32x | Высокая | Ручная реализация | Ручная реализация |
| DoReFa-Net | Веса, активации и градиенты | Низкая-средняя (3-10%) | 2-5x | 32x | Высокая | Через Brevitas | Через Larq |
| ABC-Net | Аппроксимация весов и активаций | Низкая (2-5%) | 2-4x | 8-16x | Очень высокая | Через Brevitas | Нет прямой поддержки |
| Bi-Real Net | Улучшенные бинарные активации | Низкая (2-5%) | 3-6x | 32x | Высокая | Через Brevitas | Через Larq |

## Практические рекомендации

1. **Выбор архитектуры для бинаризации**:
   - Не все архитектуры одинаково хорошо поддаются бинаризации
   - Лучше всего работают сети с большим количеством параметров и избыточностью
   - Избегайте бинаризации первого и последнего слоев для сохранения точности

2. **Обучение бинарных сетей**:
   - Используйте более высокую скорость обучения, чем для обычных сетей
   - Применяйте STE (Straight-Through Estimator) для обратного распространения
   - Используйте BatchNormalization после каждого бинарного слоя
   - Рассмотрите возможность предварительного обучения с плавающей точкой

3. **Гибридные подходы**:
   - Комбинируйте бинаризацию с другими методами оптимизации (квантизация, дистилляция)
   - Используйте разную битность для разных слоев (например, бинаризация только для сверточных слоев)
   - Рассмотрите возможность частичной бинаризации (только веса или только активации)

4. **Оценка и выбор модели**:
   - Оценивайте не только точность, но и скорость инференса и размер модели
   - Тестируйте на целевом оборудовании (особенно важно для мобильных и edge-устройств)
   - Учитывайте требования конкретной задачи (точность vs. эффективность)

5. **Инструменты и библиотеки**:
   - Для PyTorch: Brevitas, FBGEMM, PyTorch Quantization
   - Для TensorFlow: Larq, TensorFlow Lite, TensorFlow Model Optimization
   - Для специализированного оборудования: рассмотрите специфические инструменты (например, TVM, TensorRT)