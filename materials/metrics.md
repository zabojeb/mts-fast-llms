Метрики качества модели
---

### **1. Метрики для сравнения "до/после" оптимизации**
#### **Базовые метрики (всегда включаем в отчет):**
| Категория       | Метрика                          | Инструменты измерения                          |
|-----------------|----------------------------------|-----------------------------------------------|
| **Производительность** | - Время инференса (ms/token) <br> - Пропускная способность (tokens/sec) | `torch.cuda.Event`, `time.time()` |
| **Ресурсы**     | - Память GPU (GB) <br> - Загрузка GPU (%) <br> - CPU RAM (GB) | `torch.cuda.memory_allocated()`, `nvidia-smi`, `psutil` |
| **Точность**    | - Perplexity (PPL) <br> - Accuracy на датасете (если есть) | `evaluate` (Hugging Face) |

#### **Экономические метрики:**
```python
# Пример расчета стоимости инференса (для облачных GPU)
cost_per_hour = 3.0  # $/час для A100
inference_time_hours = total_time / 3600
cost = inference_time_hours * cost_per_hour
```

---

### **2. Инструменты для углеродного следа**
#### **Лучшие варианты:**
1. **CodeCarbon**  
   - Учет энергии CPU/GPU, перевод в CO₂-эквивалент  
   - Интеграция с Python-скриптами:  
   ```python
   from codecarbon import track_emissions

   @track_emissions()
   def run_inference(model, inputs):
       return model(inputs)
   ```

2. **ML CO2 Impact** (Hugging Face)  
   - Автоматический расчет для моделей HF:  
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b", tool="CO2")
   ```

3. **Carbontracker**  
   - Специализирован для ML-моделей:  
   ```python
   from carbontracker.tracker import CarbonTracker
   tracker = CarbonTracker(epochs=1, devices_by_pid=True)
   tracker.start()
   # Ваш инференс
   tracker.stop()
   ```

#### **Что фиксировать:**
- **Энергопотребление (кВт·ч)**  
- **CO₂-эквивалент (г)**  
- **Эквиваленты** (например, "как проехать 5 км на авто")  

---

### **3. Практическое применение**
#### **Шаг 1: Настройка измерений**
```python
import torch
from codecarbon import EmissionsTracker

def benchmark(model, input_ids, repetitions=100):
    tracker = EmissionsTracker(log_level="error") 
    metrics = {"time": [], "co2": []}
    
    with tracker:
        for _ in range(repetitions):
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_ids)
            torch.cuda.synchronize()
            metrics["time"].append(time.time() - start_time)
        
    metrics["co2"] = tracker.final_emissions
    return metrics
```

#### **Шаг 2: Сравнение методов**
| Метод         | Время (ms/token) | Память (GB) | CO₂ (г) | Стоимость ($/1M токенов) |
|---------------|------------------|-------------|---------|---------------------------|
| Baseline      | 50               | 14.2        | 12.3    | 0.45                      |
| 4-bit Quant   | 28               | 5.1         | 6.8     | 0.25                      |
| LoRA          | 35               | 9.3         | 8.2     | 0.30                      |

---

### **4. Визуализация результатов**
#### **Графики для отчета:**
1. **Radar Chart** для сравнения методов:  
   ```python
   import plotly.express as px
   fig = px.line_polar(results_df, r="value", theta="metric", line_close=True)
   fig.show()
   ```
   ![radar-chart](https://i.imgur.com/XYZ1234.png)

2. **CO₂ vs Cost Scatter Plot**:  
   ```python
   px.scatter(results_df, x="cost", y="co2", color="method", size="speed")
   ```

---

### **5. Полезные ссылки**
- [CodeCarbon Docs](https://codecarbon.io/)  
- [Hugging Face CO2 Guide](https://huggingface.co/docs/transformers/en/model_sharing#environmental-impact)  
- [Пример отчета по выбросам](https://github.com/mlco2/codecarbon/blob/master/examples/emissions.csv)  

---

### **Ключевые выводы для вашего проекта:**
1. **Интегрируйте CodeCarbon** с первого дня для автоматического сбора данных.  
2. **Сравнивайте не только скорость**, но и CO₂/стоимость — это сильный аргумент для презентации.  
3. **Используйте эквиваленты** (например, "Оптимизация сэкономила столько CO₂, сколько поглощает 10 деревьев за год").  

Если нужно помочь с настройкой конкретного инструмента или интерпретацией данных — готов углубиться в детали! 🌱