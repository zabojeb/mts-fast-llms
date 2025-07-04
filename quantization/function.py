from typing import Protocol, Optional, NoReturn
from torch import Tensor

class QuantizationFunction(Protocol):
    """
    Quantization function kwargs:
        weight: tensor with model weight for quantization
        num_bits: quantity of bits for quantization
    """
    
    def __call__(
            self,
            weight: Tensor,
            num_bits: Optional[int],
            ) -> NoReturn: ...