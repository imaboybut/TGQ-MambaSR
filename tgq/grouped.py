"""Swap the activation quantizers at the group-conditional sites for
``GroupedFakeQuantizerAct`` (one clipping-bound pair per token group)."""

import torch.nn as nn

from quant.quant_arch import (
    FakeQuantizerAct, GroupedFakeQuantizerAct, QuantConv2d, QuantLinear, TokenGroupContext,
)
from tgq.backbones import BackboneSpec


def swap_to_group_quantizers(model: nn.Module, ctx: TokenGroupContext, spec: BackboneSpec,
                             bit: int, search_steps: int = 100) -> int:
    """Returns the number of quantizers replaced. New quantizers start uncalibrated."""
    device = next(model.parameters()).device
    n = 0
    for name, m in model.named_modules():
        if spec.block_of(name) is None:
            continue
        if any(name.endswith(s) for s in spec.linear_sites) and isinstance(m, (QuantLinear, QuantConv2d)):
            m.act_quantizer = GroupedFakeQuantizerAct(
                bit, ctx.num_groups, ctx, token_dim=-2, search_steps=search_steps).to(device)
            n += 1
        for attr in spec.scan_sites:
            old = getattr(m, attr, None)
            if isinstance(old, FakeQuantizerAct) and not isinstance(old, GroupedFakeQuantizerAct):
                setattr(m, attr, GroupedFakeQuantizerAct(
                    bit, ctx.num_groups, ctx, token_dim=-1, search_steps=search_steps).to(device))
                n += 1
    return n
