import torch
from optimizer import optimize


def quantize(model, bits: int = 8):
    return model


def prune(model, amount: float = 0.2):
    return model


class GPT2(torch.nn.Module):
    def __init__(self):
        super(GPT2, self).__init__()
        pass

    def forward(self, x):
        return x


model = GPT2()

configs = [
    {"method": quantize, "params": {"bits": 4}},
    {"method": prune,    "params": {"amount": 0.3}},
]

optimized_model = optimize(model, configs)
