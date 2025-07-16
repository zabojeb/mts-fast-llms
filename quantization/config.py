<<<<<<< HEAD
# Module done on 16.07 in 11:57

from typing import Optional, List
=======
from typing import Optional
>>>>>>> a523bfc160e57732f7fa65d0fad6e9cf377873d2
from torch import dtype


class QuantizationConfig:
    def __init__(
        self,
        compute_dtype: Optional[dtype] = None,
        num_bits: Optional[int] = None,
<<<<<<< HEAD
        targets_names: Optional[List[str]] = None,
    ):
        """
        Kwargs:
            compute_dtype: dtype for weight
            num_bits: quanity of bits for quantization 4 or 8
            targets_names: names of layers for quantization
=======
    ):
        """
        Kwargs:
            num_bits: quanity of bits for quantization (from 1 to 8)
            compute_dtype: dtype for weight

>>>>>>> a523bfc160e57732f7fa65d0fad6e9cf377873d2
        """

        self.compute_dtype = compute_dtype
        self.num_bits = num_bits
<<<<<<< HEAD
        self.targets_names = targets_names
=======
>>>>>>> a523bfc160e57732f7fa65d0fad6e9cf377873d2


if __name__ == "__main__":
    qconfig = QuantizationConfig()
    print(vars(qconfig))
