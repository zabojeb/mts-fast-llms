import io
import torch
import time
import wandb

# Это должно быть в самом начале каждого скрипта
wandb.init(
    entity="zabojeb-hse-university",
    project="MTS-fast-llms",
    name="quant8_distill_temp2",  # имя запуска, оно должно быть уникальным!
    config={
        "quantization_bits": 8,
        "distillation_temperature": 2.0,
        "batch_size": 16,
        "learning_rate": 5e-5,
        # любые другие гиперпараметры
    },
)

# Эту функцию нужно вызывать внутри функции для квантизации модели


def log_quantization(
    model: torch.nn.Module,
    stage: str,
    quant_bits: int,
    input_sample: torch.Tensor = None,
    step: int = None,
):
    """
    Функция для логирования метрик квантизации модели в WandB.

    Args:
      model:        квантизированная (или промежуточная) модель
      stage:        строка — название этапа ('pre', 'post', 'calibration', ...)
      quant_bits:   число бит для квантования
      input_sample: опциональный тензор, на котором можно измерить latency
      step:         опциональный номер шага/эпохи
    """
    metrics = {
        "quant/stage": stage,
        "quant/bits": quant_bits,
    }

    # 1) модельный вес (размер на диске в мегабайтах)
    try:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_mb = buffer.getbuffer().nbytes / (1024**2)
        metrics["model/size_mb"] = size_mb
    except Exception:
        pass

    # 2) если есть сэмпл, замерим латентность
    if input_sample is not None:
        model.eval()
        with torch.no_grad():
            start = time.time()
            _ = model(input_sample)
            metrics["inference/latency_s"] = time.time() - start

    # 3) необязательные метрики: min/max весов
    all_weights = torch.cat([p.data.view(-1).abs() for p in model.parameters()])
    metrics.update(
        {
            "weight/min": float(all_weights.min()),
            "weight/max": float(all_weights.max()),
        }
    )

    # 4) логируем вместе с шагом (если он есть)
    log_kwargs = {"step": step} if step is not None else {}
    wandb.log(metrics, **log_kwargs)


# Пример использования функции


def quantize_model(model, quantize_function):
    # до квантизации
    log_quantization(model, stage="pre", quant_bits=bits)

    # сам процесс квантизации
    q_model = quantize_function(model)

    # после квантизации
    log_quantization(q_model, stage="post", quant_bits=bits)

    return q_model
