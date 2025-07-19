# Module done on 19.07 in 9:03

from typing import Optional, Dict, List, Type
from torch import dtype
from torch import nn

QINT8 = 8
QINT4 = 4


class QuantizationConfigLayer:
    def __init__(
            self,
            qtype: int,
            fraction: Optional[float] = 1,
            names: Optional[List[str]] = None,
    ):
        """
        Configuration for quantizing a layer

        Args:
            qtype: Quantization type (QINT8 or QINT4)
            fraction: Fraction of parameters to quantize (0 to 1)
            names: List of parameter names to quantize (if specified, fraction is ignored)
        """
        if names is not None:
            fraction = None

        self.qtype = qtype
        self.fraction = fraction
        self.names = names

    def __repr__(self):
        return "QuantizationConfigLayer(qtype={}, fraction={}, names={})".format(
            self.qtype, self.fraction, self.names
        )


class QuantizationConfig:
    def __init__(
            self,
            compute_dtype: Optional[dtype] = None,
            layers: Optional[Dict[Type[nn.Module], QuantizationConfigLayer]] = None,
    ):
        """
        Quantization configuration for a model

        Args:
            compute_dtype: Data type for computations
            layers: Dictionary mapping layer types to their quantization configs
        """
        self.compute_dtype = compute_dtype
        self.layers = layers or {}

    def __repr__(self):
        return "QuantizationConfig(compute_dtype={}, layers={})".format(
            self.compute_dtype, self.layers
        )


if __name__ == "__main__":
    from torch import nn
    import torch

    qconfig = QuantizationConfig(
        compute_dtype=torch.bfloat16,
        layers={
            nn.Linear: QuantizationConfigLayer(
                qtype=QINT8,
                fraction=0.8,
            )
        },
    )
    print(qconfig)