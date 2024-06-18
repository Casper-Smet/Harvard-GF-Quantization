from torch import nn
from torchvision.models import efficientnet as efn

import torch
from functools import partial
from torchvision.models.quantization.mobilenetv3 import QuantizableSqueezeExcitation
from typing import Callable


def replace_activations(model):
    for name, module in model.named_children():
        if isinstance(module, nn.SiLU):
            setattr(model, name, nn.Hardswish(inplace=module.inplace))
        else:
            replace_activations(module)


class QuantizableMBConv(efn.MBConv):
    def __init__(
        self,
        cnf: efn.MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = QuantizableSqueezeExcitation,
    ):
        super().__init__(
            cnf=cnf,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            se_layer=se_layer,
        )
        # super().__init__()
        # self.skip_mul = nn.quantized.FloatFunctional()
        self.f_add = nn.quantized.FloatFunctional()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # return self.skip_mul.mul(self._scale(input), input)
        result = self.block(input)
        if self.use_res_connect:
            result = self.f_add.add(input, self.stochastic_depth(result))
            # result = self.additive.add
            # result += input
        return result


def quantifiable_efficientnet(weights=None, progress=True, **kwargs):
    block = partial(efn.MBConv, se_layer=QuantizableSqueezeExcitation)
    block = partial(QuantizableMBConv, se_layer=QuantizableSqueezeExcitation)
    bneck_conf = partial(
        efn.MBConvConfig,
        width_mult=kwargs.pop("width_mult"),
        depth_mult=kwargs.pop("depth_mult"),
        block=block,
    )
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    last_channel = None

    model = efn._efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.2),
        last_channel,
        weights,
        progress,
        **kwargs
    )
    model.features[0][0] = nn.Conv2d(
        1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    )
    model.classifier[1] = nn.Identity()
    replace_activations(model)
    return model
# qnet = quantifiable_efficientnet(width_mult=1.0, depth_mult=1.1).to('cuda:0')
