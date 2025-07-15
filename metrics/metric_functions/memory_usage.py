from typing import List, Dict, Optional


def compute_memory(
    *,
    gpu_memory: Optional[List[float]] = None,
    cpu_memory: Optional[float] = None,
    **kwargs
) -> Dict[str, Optional[float]]:
    """Возвращает использование памяти в MB."""
    return {
        "gpu_peak_mb": max(gpu_memory) if gpu_memory else None,
        "cpu_mb": cpu_memory
    }