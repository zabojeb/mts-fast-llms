# Module done on 17.07 in 14:33

from typing import Optional, Dict, List
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
        # TODO: description

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
        layers: Dict[nn.Module, QuantizationConfigLayer] = None,
    ):
        """
        Kwargs:
            compute_dtype: dtype for weight
        """

        # TODO: description

        self.compute_dtype = compute_dtype
        self.layers = layers

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
