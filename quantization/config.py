from typing import Optional

class QuantizationConfig:
    def __init__(self, num_bits: Optional[int] = None):
        """
        Kwargs:
            num_bits: quantity of bits for quantization
        """

        self.num_bits = num_bits
