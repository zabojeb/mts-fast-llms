import pytest
import torch
import psutil
import wandb

from optimizer import optimize


class DummyModel:
    def __init__(self):
        self.value = 0


def inc(model, inc: int = 1):
    model.value += inc
    return model


def set_to(model, x: int = 0):
    model.value = x
    return model


@pytest.fixture(autouse=True)
def mock_wandb(monkeypatch):
    calls = {"init": [], "log": []}
    monkeypatch.setattr(wandb, "init", lambda **
                        kwargs: calls["init"].append(kwargs) or None)
    monkeypatch.setattr(
        wandb, "log", lambda data: calls["log"].append(data) or None)
    monkeypatch.setattr(wandb, "define_metric", lambda name: None)
    return calls


@pytest.fixture(autouse=True)
def mock_system(monkeypatch):
    class VM:
        total = 1024
        used = 512
        available = 512
        percent = 50.0

    monkeypatch.setattr(psutil, "virtual_memory", lambda: VM())
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 12.5)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_optimize_applies_steps_and_returns_model(mock_wandb):
    model = DummyModel()
    configs = [
        {"method": inc,    "params": {"inc": 5}},
        {"method": set_to, "params": {"x": 42}},
    ]
    out = optimize(model, configs, project="prj", run_name="run", reinit=True)
    assert isinstance(out, DummyModel)

    assert out.value == 42  # 42 ЭЙ БРАТУХА 42 42 ЗНАЕТ ВСЯ МОЯ БРАТВА ЧТО ТАКОЕ 42


def test_wandb_init_and_log_called(mock_wandb):
    calls = mock_wandb
    model = DummyModel()
    configs = [{"method": inc, "params": {"inc": 3}}]

    optimize(model, configs)

    assert len(calls["init"]) == 1
    assert calls["init"][0] == {}

    assert len(calls["log"]) == 1
    log_data = calls["log"][0]

    assert log_data["opt_step"] == 1
    assert log_data["method"] == "inc"
    assert log_data["inc"] == 3

    for key in ["sys_ram_total", "sys_ram_used", "sys_ram_available", "sys_ram_percent", "cpu_percent"]:
        assert key in log_data
        assert isinstance(log_data[key], (int, float))


def test_non_callable_method_raises():
    model = DummyModel()
    bad_cfg = [{"method": "not_a_func", "params": {}}]
    with pytest.raises(ValueError):
        optimize(model, bad_cfg)
