from typing import Optional
from torch import dtype


class QuantizationConfig:
    def __init__(
        self, compute_dtype: Optional[dtype] = None, num_bits: Optional[int] = None
    ):
        """
        Kwargs:
            compute_dtype: dtype for weight
            num_bits: quantity of bits for quantization (less than 8)

        """

        self.compute_dtype = compute_dtype
        self.num_bits = num_bits


if __name__ == "__main__":
    qconfig = QuantizationConfig()
    print(vars(qconfig))
