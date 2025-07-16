from typing import Optional


def compute_energy(
    *,
    emissions: Optional[float] = None,
    **kwargs
) -> Optional[float]:
    """Возвращает потребление энергии в kWh."""
    return emissions * 1000 if emissions is not None else None  # Convert kgCO2 to kWh