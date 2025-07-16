from typing import Optional
from torch import dtype


class QuantizationConfig:
    def __init__(
        self,
        compute_dtype: Optional[dtype] = None,
        num_bits: Optional[int] = None,
    ):
        """
        Kwargs:
            num_bits: quanity of bits for quantization (from 1 to 8)
            compute_dtype: dtype for weight

        """

        self.compute_dtype = compute_dtype
        self.num_bits = num_bits


if __name__ == "__main__":
    qconfig = QuantizationConfig()
    print(vars(qconfig))
