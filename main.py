import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import nltk
import logging
from typing import Dict

from .quantization.config import QuantizationConfig, QuantizationConfigLayer, QINT8
from .quantization.quantize import quantize
from .metrics.evaluate_metrics import metrics_evaluate
from .pool_2_pool import pool_2_pool

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка NLTK ресурсов
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

qconfig = QuantizationConfig(
    layers={
        nn.Linear: QuantizationConfigLayer(
            qtype=QINT8,
            fraction=1,
        )
    },
)

def main():
    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = torch.load("chmo.pth", weights_only=False)

    # Проверка поддержки perplexity
    try:
        dummy_input = tokenizer(["test"], return_tensors="pt").to("cuda")
        with torch.amp.autocast('cuda'):
            outputs = model(**dummy_input, labels=dummy_input["input_ids"])
        if not hasattr(outputs, "loss") or outputs.loss is None:
            logger.warning("Модель Qwen3-4B может не поддерживать вычисление perplexity")
    except Exception as e:
        logger.warning(f"Ошибка проверки perplexity: {str(e)}")

    # Очистка памяти
    torch.cuda.empty_cache()

    # Загрузка датасета WMT14
    dataset = load_dataset("wmt14", "ru-en", split="test[:10]")
    dataset = dataset.map(
        lambda x: {
            "text": f"Переведи на английский: {x['translation']['ru']}",
            "references": [x['translation']['en']]  # Оборачиваем в список для единообразия
        },
        remove_columns=["translation"]
    )

    # Логирование примеров датасета
    logger.info(f"Примеры датасета: {dataset[:5]}")

    # Оценка метрик
    results = metrics_evaluate(
        model=model,
        processor=tokenizer,
        dataset=dataset,
        f_type="generation",
        device="cuda",
        batch_size=4,
        field_mapping={
            "text": "text",
            "reference": "references",
            "generate_kwargs": {
                "max_new_tokens": 300,
                "num_beams": 1,
                "do_sample": False,
            }
        },
        log=True
    )

    # Формирование конфига квантизации
    configs = [
        {
            "method": {"__name__": "quantize"},
            "params": {
                "qtype": "QINT8",
                "fraction": 1
            }
        }
    ] if not model.training else []

    # Добавление метрик в pool
    pool_2_pool(
        metrics_results=results,
        configs=configs,
        model_name=model_name,
        task_type="generation",
        original=not model.training  # True для исходной модели, False для квантованной
    )

    # Вывод результатов
    print("Результаты метрик:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()