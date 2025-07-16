from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_model_byherself import metrics_evaluate

# Загрузка оптимизированной модели (например, квантизированной Llama)
model = AutoModelForCausalLM.from_pretrained("path/to/optimized/llama", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("path/to/optimized/llama")

# Пример датасета
from datasets import Dataset
dataset = Dataset.from_dict({"text": ["Привет, мир!"], "references": ["Hello, world!"]})

# Оценка метрик
results = metrics_evaluate(
    model=model,
    processor=tokenizer,
    dataset=dataset,
    f_type="translation",
    device="cuda",
    batch_size=1,
    field_mapping={"text": "text", "reference": "references"}
)
print(results)