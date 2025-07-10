from typing import Optional
from torch import dtype


class QuantizationConfig:
    def __init__(
        self,
        compute_dtype: Optional[dtype] = None,
        quantize_int8: Optional[bool] = None,
    ):
        """
        Kwargs:
            compute_dtype: dtype for weight
            quantize_int8: apply int8 quantization

        """

        self.compute_dtype = compute_dtype
        self.quantize_int8 = quantize_int8


if __name__ == "__main__":
    qconfig = QuantizationConfig()
    print(vars(qconfig))
