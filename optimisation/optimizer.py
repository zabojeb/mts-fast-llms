import wandb
import psutil
import torch
from typing import Iterable, Dict, Any


def optimize(
    model: torch.nn.Module,
    configs: Iterable[Dict[str, Any]],
    project: str = None,
    run_name: str = None,
    reinit: bool = False
) -> torch.nn.Module:
    """
    Args:
        model: исходная модель
        configs: итерируемый набор конфигов, где каждый конфиг — это dict с ключами:
            - "method": callable, принимающий в качестве первого аргумента модель
            - "params": dict с именованными параметрами для метода
        project: название проекта W&B (если None инициализирован уже, будет использоваться существующий).
        run_name: имя запуска W&B.
        reinit: если True, принудительно создаст новый запуск, даже если уже есть активный.

    Returns:
        torch.nn.Module: модель после применения всех оптимизаций.
    """

    if reinit or wandb.run is None:
        init_args = {}
        if project is not None:
            init_args['project'] = project
        if run_name is not None:
            init_args['name'] = run_name
        wandb.init(**init_args)

    for step, cfg in enumerate(configs, start=1):
        method = cfg.get("method")
        params = cfg.get("params", {})

        if not callable(method):
            raise ValueError(
                f"Ожидался callable в поле 'method', получено {type(method)}")

        model = method(model, **params)

        log_data = {
            'opt_step': step,
            'method': method.__name__,
        }

        for key, value in params.items():
            try:
                wandb.define_metric(key)
                log_data[key] = value
            except Exception:
                continue

        vm = psutil.virtual_memory()
        log_data.update({
            'sys_ram_total': vm.total,
            'sys_ram_used': vm.used,
            'sys_ram_available': vm.available,
            'sys_ram_percent': vm.percent,
            'cpu_percent': psutil.cpu_percent(interval=None),
        })

        if torch.cuda.is_available():
            log_data['gpu_count'] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                total_mem = torch.cuda.get_device_properties(i).total_memory
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)

                log_data[f'gpu_{i}_mem_total'] = total_mem
                log_data[f'gpu_{i}_mem_allocated'] = allocated
                log_data[f'gpu_{i}_mem_reserved'] = reserved

        wandb.log(log_data)

    return model
