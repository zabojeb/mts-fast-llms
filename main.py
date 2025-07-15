import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import nltk
from .quantization.quantize import quantize
from .quantization.config import QuantizationConfig
from .metrics.evaluate_metrics import metrics_evaluate

# Настройка логирования для диагностики
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка NLTK ресурсов
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Загрузка модели и токенизатора
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    device_map="cuda",
    torch_dtype=torch.bfloat16
)

# Квантизация
config = QuantizationConfig(compute_dtype=torch.bfloat16, num_bits=8)
quantize(model, config)

# Проверка поддержки perplexity
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    dummy_input = tokenizer(["test"], return_tensors="pt").to("cuda")
    outputs = model(**dummy_input, labels=dummy_input["input_ids"])
    if not hasattr(outputs, "loss") or outputs.loss is None:
        logger.warning("Квантованная модель Qwen3-4B может не поддерживать вычисление perplexity")
except Exception as e:
    logger.warning(f"Ошибка проверки perplexity: {str(e)}")

# Загрузка датасета WMT14
dataset = load_dataset("wmt14", "ru-en", split="test[:100]")
dataset = dataset.map(
    lambda x: {
        "text": f"Переведи на английский: {x['translation']['ru']}",
        "references": [[x['translation']['en']]]  # Оборачиваем строку в список
    },
    remove_columns=["translation"]
)

# Оценка метрик
results = metrics_evaluate(
    model=model,
    processor=tokenizer,
    dataset=dataset,
    f_type="translation",
    device="cuda",
    batch_size=8,
    field_mapping={
        "text": "text",
        "reference": "references",
        "generate_kwargs": {
            "max_length": 300,  # Увеличен для длинных переводов
            "num_beams": 5,
            "do_sample": False,
            "early_stopping": True
        }
    },
    log=True
)

print("Результаты метрик:")
for metric, value in results.items():
    print(f"{metric}: {value}")