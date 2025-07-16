import torch
import torch.nn.utils.prune as prune
import os
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def _replace_module(model: torch.nn.Module, module_name: str, new_module: torch.nn.Module):
    parts = module_name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)

def structured_prune_llm(model: torch.nn.Module, prune_ratio: float) -> torch.nn.Module:
    """
    Структурный L1-прунинг только FFN-слоёв:
      – fc1: Linear(hidden_size → intermediate_size)  prune dim=0 (строки)
      – fc2: Linear(intermediate_size → hidden_size)  prune dim=1 (столбцы)
    """
    cfg = getattr(model, "config", None)
    if cfg is None or not hasattr(cfg, "hidden_size") or not hasattr(cfg, "intermediate_size"):
        raise ValueError("Model config не содержит hidden_size/intermediate_size")
    H, I = cfg.hidden_size, cfg.intermediate_size

    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue

        W = module.weight.data
        shape = W.shape
        # fc1-подслой
        if shape[0] == I and shape[1] == H:
            dim = 0
        # fc2-подслой
        elif shape[0] == H and shape[1] == I:
            dim = 1
        else:
            continue  # не FFN — пропускаем

        # 1) структурный L1-прунинг
        prune.ln_structured(module, name="weight", amount=prune_ratio, n=1, dim=dim)
        prune.remove(module, "weight")

        # 2) «вырезание» нулевых строк/столбцов
        W = module.weight.data
        B = module.bias.data if module.bias is not None else None

        if dim == 0:
            scores = W.abs().sum(dim=1)
            keep = torch.nonzero(scores, as_tuple=True)[0]
            new_out, new_in = keep.numel(), W.size(1)
            new_W = W[keep, :].clone()
            new_B = B[keep].clone() if B is not None else None
        else:
            scores = W.abs().sum(dim=0)
            keep = torch.nonzero(scores, as_tuple=True)[0]
            new_out, new_in = W.size(0), keep.numel()
            new_W = W[:, keep].clone()
            new_B = B.clone() if B is not None else None

        # 3) новая линейка и копирование
        new_lin = torch.nn.Linear(new_in, new_out, bias=(B is not None))
        new_lin.weight.data.copy_(new_W)
        if B is not None:
            new_lin.bias.data.copy_(new_B)

        # 4) встроить обратно
        _replace_module(model, name, new_lin)

    return model

def measure_size_on_disk(model: torch.nn.PreTrainedModel) -> int:
    with tempfile.TemporaryDirectory() as td:
        model.save_pretrained(td)
        total = 0
        for root, _, files in os.walk(td):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
    return total

def sanity_check(model: torch.nn.PreTrainedModel, tokenizer: AutoTokenizer, device: torch.device):
    model.eval()
    prompt = "Hello, world!"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    assert logits is not None, "Модель не вернула logits"
    print(f"✅ Sanity check passed: logits shape {tuple(logits.shape)}")

if __name__ == "__main__":
    model_name = "Qwen/Qwen-3-0.6B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model and tokenizer…")
    cfg = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(cfg).from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    size_before = measure_size_on_disk(model)
    print(f"Размер до прунинга: {size_before/1e9:.3f} GB")
    sanity_check(model, tokenizer, device)

    # Применяем 20% прунинга к FFN
    pruned = structured_prune_llm(model, prune_ratio=0.2).to(device)

    size_after = measure_size_on_disk(pruned)
    print(f"Размер после прунинга: {size_after/1e9:.3f} GB")
    print(f"Сэкономлено: {(size_before-size_after)/1e9:.3f} GB")
    sanity_check(pruned, tokenizer, device)