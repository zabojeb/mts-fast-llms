# Module done on 16.07 in 11:57

from typing import Optional, List
from torch import dtype


class QuantizationConfig:
    def __init__(
        self,
        compute_dtype: Optional[dtype] = None,
        num_bits: Optional[int] = None,
        targets_names: Optional[List[str]] = None,
    ):
        """
        Kwargs:
            compute_dtype: dtype for weight
            num_bits: quanity of bits for quantization 4 or 8
            targets_names: names of layers for quantization
        """

        self.compute_dtype = compute_dtype
        self.num_bits = num_bits
        self.targets_names = targets_names


if __name__ == "__main__":
    qconfig = QuantizationConfig()
    print(vars(qconfig))
