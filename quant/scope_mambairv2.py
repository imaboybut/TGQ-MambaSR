"""Quantization scope for MambaIRv2 (Light / Base), injected in place.

Per ``AttentiveLayer`` block:
  Linear -> QuantLinear (``wqkv`` -> QuantLinearQKV when ``qkv_separation``):
    wqkv, win_mhsa.proj, convffn{1,2}.fc{1,2}, assm.out_proj, assm.route[0|2]
  Conv2d -> QuantConv2d (when ``quant_conv``):
    convffn{1,2}.{sg.conv | dwconv.depthwise_conv[0]}, assm.in_proj[0], assm.CPE[0]
  Selective-scan internals (nn.Identity placeholders -> fake quantizers):
    selectiveScan.dt_weight / x_weight (W),
    selectiveScan.mymodule_{xs,dts,delta,Bs,Cs,As,dt_projs_bias,hs} (A)
  Prompt library:
    assm.fq_embeddingB_w / fq_token_w (W), assm.fq_cls_policy_a / fq_full_embedding_a (A)
Per ``WindowAttention``: mymodule_{q,k,v,a} (A)

``build_lora_network`` wraps every QuantLinear with ``QuantLinearLoRA``
(weight-training protocol); call it after the bounds are initialised.
"""

import torch.nn as nn

from quant.quant_arch import (
    FakeQuantizerWeight, FakeQuantizerAct, QuantLinear, QuantLinearQKV,
    QuantConv2d, QuantLinearLoRA,
)


def _linear(linear: nn.Linear, config: dict, qkv: bool = False):
    q = QuantLinearQKV(config) if qkv else QuantLinear(config)
    q.set_param(linear)
    return q


def _conv(conv: nn.Conv2d, config: dict):
    q = QuantConv2d(config)
    q.set_param(conv)
    return q


def build_quantized_network(net: nn.Module, bit: int, search_steps: int = 100,
                            qkv_separation: bool = True, quant_conv: bool = True):
    from arch.mambairv2_arch import AttentiveLayer, WindowAttention

    config = {"bit": bit, "search_steps": search_steps}
    blocks = [m for m in net.modules() if isinstance(m, AttentiveLayer)]
    win_attns = [m for m in net.modules() if isinstance(m, WindowAttention)]

    for blk in blocks:
        sel = blk.assm.selectiveScan
        sel.dt_weight = FakeQuantizerWeight(bit, search_steps)
        sel.x_weight = FakeQuantizerWeight(bit, search_steps)
        for name in ("xs", "dts", "delta", "Bs", "Cs", "As", "dt_projs_bias", "hs"):
            setattr(sel, f"mymodule_{name}", FakeQuantizerAct(bit, search_steps))

        blk.wqkv = _linear(blk.wqkv, config, qkv=qkv_separation)
        blk.win_mhsa.proj = _linear(blk.win_mhsa.proj, config)
        blk.convffn1.fc1 = _linear(blk.convffn1.fc1, config)
        blk.convffn1.fc2 = _linear(blk.convffn1.fc2, config)
        blk.convffn2.fc1 = _linear(blk.convffn2.fc1, config)
        blk.convffn2.fc2 = _linear(blk.convffn2.fc2, config)
        blk.assm.out_proj = _linear(blk.assm.out_proj, config)
        blk.assm.route[0] = _linear(blk.assm.route[0], config)
        blk.assm.route[2] = _linear(blk.assm.route[2], config)

        if quant_conv:
            for ffn in (blk.convffn1, blk.convffn2):
                if hasattr(ffn, "sg"):          # GatedMLP (Light)
                    ffn.sg.conv = _conv(ffn.sg.conv, config)
                elif hasattr(ffn, "dwconv"):    # ConvFFN (Base)
                    ffn.dwconv.depthwise_conv[0] = _conv(ffn.dwconv.depthwise_conv[0], config)
            blk.assm.in_proj[0] = _conv(blk.assm.in_proj[0], config)
            blk.assm.CPE[0] = _conv(blk.assm.CPE[0], config)

        blk.assm.fq_embeddingB_w = FakeQuantizerWeight(bit, search_steps)
        blk.assm.fq_token_w = FakeQuantizerWeight(bit, search_steps)
        blk.assm.fq_cls_policy_a = FakeQuantizerAct(bit, search_steps)
        blk.assm.fq_full_embedding_a = FakeQuantizerAct(bit, search_steps)

    for attn in win_attns:
        for name in ("q", "k", "v", "a"):
            setattr(attn, f"mymodule_{name}", FakeQuantizerAct(bit, search_steps))


def build_lora_network(net: nn.Module, bit: int, search_steps: int = 100,
                       r: int = 8, num_lora: int = 4, qkv_separation: bool = True) -> int:
    from arch.mambairv2_arch import AttentiveLayer

    config = {"bit": bit, "search_steps": search_steps}
    n = 0

    def _wrap(q: QuantLinear) -> QuantLinearLoRA:
        nonlocal n
        n += 1
        return QuantLinearLoRA(q, config, r=r, num_lora=num_lora)

    for blk in (m for m in net.modules() if isinstance(m, AttentiveLayer)):
        if qkv_separation:
            blk.wqkv.q, blk.wqkv.k, blk.wqkv.v = _wrap(blk.wqkv.q), _wrap(blk.wqkv.k), _wrap(blk.wqkv.v)
        else:
            blk.wqkv = _wrap(blk.wqkv)
        blk.win_mhsa.proj = _wrap(blk.win_mhsa.proj)
        blk.convffn1.fc1 = _wrap(blk.convffn1.fc1)
        blk.convffn1.fc2 = _wrap(blk.convffn1.fc2)
        blk.convffn2.fc1 = _wrap(blk.convffn2.fc1)
        blk.convffn2.fc2 = _wrap(blk.convffn2.fc2)
        blk.assm.out_proj = _wrap(blk.assm.out_proj)
        blk.assm.route[0] = _wrap(blk.assm.route[0])
        blk.assm.route[2] = _wrap(blk.assm.route[2])
    return n
