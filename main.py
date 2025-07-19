import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
from metrics import metrics_evaluate
from quantization.config import QuantizationConfig, QuantizationConfigLayer, QINT4
from quantization.quantize import quantize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Инициализация модели
    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    # Квантизация (исправленная версия)
    qconfig = QuantizationConfig(
        layers={
            nn.Linear: QuantizationConfigLayer(
                qtype=QINT4,
                fraction=1.0,  # Явно указываем float
            )
        }
    )
    quantize(model, qconfig)

    # Подготовка датасета
    dataset = load_dataset("wmt14", "ru-en", split="test[:100]").map(
        lambda x: {
            "text": f"Переведи на английский: {x['translation']['ru']}",
            "references": [x['translation']['en']]
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
        batch_size=4,
        field_mapping={
            "text": "text",
            "reference": "references",
            "generate_kwargs": {
                "max_new_tokens": 300,
                "num_beams": 1,
                "do_sample": False,
            }
        }
    )

    # Вывод результатов
    print("\nРезультаты оценки:")
    for metric, value in results.items():
        if isinstance(value, dict):
            print(f"{metric}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()