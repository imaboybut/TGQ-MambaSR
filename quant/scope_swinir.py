"""Quantization scope for SwinIR-light (same scope as 2DQuant / SPR2Q), injected in place.

Per ``SwinTransformerBlock``:
  Linear -> QuantLinear (``attn.qkv`` -> QuantLinearQKV when ``qkv_separation``):
    attn.qkv, attn.proj, mlp.fc1, mlp.fc2
  WindowAttention operands (nn.Identity placeholders -> fake quantizers):
    attn.mymodule_{q,k,v,a}
Convolutions, LayerNorms, softmax, GELU and the upsampler stay in full precision.
"""

import torch.nn as nn

from quant.quant_arch import FakeQuantizerAct, QuantLinear, QuantLinearQKV, QuantLinearLoRA


def _linear(linear: nn.Linear, config: dict, qkv: bool = False):
    q = QuantLinearQKV(config) if qkv else QuantLinear(config)
    q.set_param(linear)
    return q


def build_quantized_network(net: nn.Module, bit: int, search_steps: int = 100,
                            qkv_separation: bool = True):
    from arch.swinir_arch import SwinTransformerBlock, WindowAttention

    config = {"bit": bit, "search_steps": search_steps}
    for blk in [m for m in net.modules() if isinstance(m, SwinTransformerBlock)]:
        blk.attn.qkv = _linear(blk.attn.qkv, config, qkv=qkv_separation)
        blk.attn.proj = _linear(blk.attn.proj, config)
        blk.mlp.fc1 = _linear(blk.mlp.fc1, config)
        blk.mlp.fc2 = _linear(blk.mlp.fc2, config)
    for attn in [m for m in net.modules() if isinstance(m, WindowAttention)]:
        for name in ("q", "k", "v", "a"):
            setattr(attn, f"mymodule_{name}", FakeQuantizerAct(bit, search_steps))


def build_lora_network(net: nn.Module, bit: int, search_steps: int = 100,
                       r: int = 8, num_lora: int = 4, qkv_separation: bool = True) -> int:
    from arch.swinir_arch import SwinTransformerBlock

    config = {"bit": bit, "search_steps": search_steps}
    n = 0

    def _wrap(q: QuantLinear) -> QuantLinearLoRA:
        nonlocal n
        n += 1
        return QuantLinearLoRA(q, config, r=r, num_lora=num_lora)

    for blk in (m for m in net.modules() if isinstance(m, SwinTransformerBlock)):
        if qkv_separation:
            blk.attn.qkv.q = _wrap(blk.attn.qkv.q)
            blk.attn.qkv.k = _wrap(blk.attn.qkv.k)
            blk.attn.qkv.v = _wrap(blk.attn.qkv.v)
        else:
            blk.attn.qkv = _wrap(blk.attn.qkv)
        blk.attn.proj = _wrap(blk.attn.proj)
        blk.mlp.fc1 = _wrap(blk.mlp.fc1)
        blk.mlp.fc2 = _wrap(blk.mlp.fc2)
    return n
