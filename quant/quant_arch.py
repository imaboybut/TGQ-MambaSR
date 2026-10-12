"""Fake-quantization primitives used by TGQ-MambaSR.

* ``RoundSTE`` / ``ClipSTE``: straight-through estimators for round and clip.
  ``ClipSTE`` also back-propagates to the (scalar or per-group) clipping bounds.
* ``mse_bound_search``: clipping-bound initialisation by empirical
  quantization-error minimisation (Eq. 10 in the paper).
* ``FakeQuantizerWeight`` / ``FakeQuantizerAct``: per-tensor uniform quantizers
  with learnable lower/upper bounds. The first forward pass initialises the
  bounds from the observed tensor and returns it unquantized.
* ``QuantLinear`` / ``QuantLinearQKV`` / ``QuantConv2d``: layers that quantize
  both their weight and their input activation.
* ``QuantLinearLoRA``: ``QuantLinear`` with a softmax-weighted group of LoRA
  rectifiers added to the weight before quantization (weight-training protocol).
* ``TokenGroupContext`` + ``GroupedFakeQuantizerAct``: group-conditional
  activation quantizer whose bounds are indexed by the predicted token group
  (Eq. 5 in the paper).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Straight-through estimators
# ---------------------------------------------------------------------------
class RoundSTE(Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output


class ClipSTE(Function):
    """Clamp ``x`` to ``[lb, ub]``; ``lb``/``ub`` may be scalars or tensors that
    broadcast with ``x``. Gradients reach ``lb`` where ``x <= lb`` and ``ub``
    where ``x >= ub``; broadcast dims are summed back to the bound shape."""

    @staticmethod
    def forward(ctx, x: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
        ctx.save_for_backward(x, lb, ub)
        return x.clamp(lb, ub)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, lb, ub = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x < lb) | (x > ub)] = 0
        grad_lb = grad_output.clone()
        grad_lb[x > lb] = 0
        grad_ub = grad_output.clone()
        grad_ub[x < ub] = 0

        def _reduce(t: Tensor, target_shape: torch.Size) -> Tensor:
            while t.dim() > len(target_shape):
                t = t.sum(dim=0)
            for dim, size in enumerate(target_shape):
                if size == 1 and t.size(dim) != 1:
                    t = t.sum(dim=dim, keepdim=True)
            return t

        return grad_input, _reduce(grad_lb, lb.shape), _reduce(grad_ub, ub.shape)


# ---------------------------------------------------------------------------
# Bound initialisation (Eq. 10): grid search over symmetric shrinkage of the
# [min, max] range, keeping the pair with the smallest quantization error.
# ---------------------------------------------------------------------------
@torch.no_grad()
def mse_bound_search(x: Tensor, bit: int, steps: int = 100) -> Tuple[float, float]:
    x = x.detach().float()
    lo, hi = x.min(), x.max()
    levels = 2 ** int(bit) - 1
    step = (hi - lo) / (2 * steps)
    best_lb, best_ub, best_err = lo, hi, float("inf")
    for i in range(steps):
        lb = lo + step * i
        ub = hi - step * i
        s = ((ub - lb) / levels).clamp_min(1e-12)
        q = torch.round((x.clamp(lb, ub) - lb) / s) * s + lb
        err = float(torch.norm(x - q))
        if err < best_err:
            best_lb, best_ub, best_err = lb, ub, err
    return float(best_lb), float(best_ub)


# ---------------------------------------------------------------------------
# Fake quantizers
# ---------------------------------------------------------------------------
class FakeQuantizerBase(nn.Module):
    def __init__(self, bit: int = 4, search_steps: int = 100):
        super().__init__()
        self.lower_bound = nn.Parameter(torch.zeros(()))
        self.upper_bound = nn.Parameter(torch.ones(()))
        self.register_buffer("n_bit", torch.tensor(float(bit)))
        self.search_steps = search_steps
        self.calibrated = False

    @property
    def bit(self) -> int:
        return int(self.n_bit.item())

    def set_bit(self, bit: int):
        with torch.no_grad():
            self.n_bit.fill_(float(bit))

    def set_bounds(self, lb, ub):
        with torch.no_grad():
            self.lower_bound.copy_(torch.as_tensor(lb, dtype=torch.float32,
                                                   device=self.lower_bound.device).reshape(self.lower_bound.shape))
            self.upper_bound.copy_(torch.as_tensor(ub, dtype=torch.float32,
                                                   device=self.upper_bound.device).reshape(self.upper_bound.shape))

    def _quantize(self, x: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
        s = (ub - lb) / (2 ** self.n_bit - 1)
        c = ClipSTE.apply(x, lb, ub)
        return RoundSTE.apply((c - lb) / s) * s + lb

    def _init_bounds(self, x: Tensor):
        self.set_bounds(*mse_bound_search(x, self.bit, self.search_steps))
        self.calibrated = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.calibrated:
            self._init_bounds(x)
            return x
        return self._quantize(x, self.lower_bound, self.upper_bound)


class FakeQuantizerWeight(FakeQuantizerBase):
    pass


class FakeQuantizerAct(FakeQuantizerBase):
    pass


# ---------------------------------------------------------------------------
# Quantized layers
# ---------------------------------------------------------------------------
class QuantBase(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.bit = config["bit"]
        self.search_steps = config.get("search_steps", 100)
        self.weight_quantizer = FakeQuantizerWeight(self.bit, self.search_steps)
        self.act_quantizer = FakeQuantizerAct(self.bit, self.search_steps)
        self.weight: nn.Parameter
        self.bias: Optional[nn.Parameter] = None


class QuantLinear(QuantBase):
    def set_param(self, linear: nn.Linear):
        self.out_feature = linear.weight.shape[0]
        self.in_feature = linear.weight.shape[1]
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

    def quantized_weight(self) -> Tensor:
        return self.weight_quantizer(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(self.act_quantizer(x), self.quantized_weight(), self.bias)


class QuantLinearLoRA(QuantLinear):
    """``QuantLinear`` plus a rectifier group of ``num_lora`` LoRA branches whose
    softmax-weighted sum is added to the frozen weight before quantization.
    The branches and their mixing logits are the only weight-side trainables
    (weight-training protocol of Table 3)."""

    def __init__(self, base: QuantLinear, config: dict, r: int = 8, num_lora: int = 4):
        super().__init__(config)
        self.in_feature, self.out_feature = base.in_feature, base.out_feature
        self.weight, self.bias = base.weight, base.bias
        self.weight_quantizer, self.act_quantizer = base.weight_quantizer, base.act_quantizer
        self.r, self.num_lora = r, num_lora
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.zeros(r, self.in_feature)) for _ in range(num_lora)])
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.out_feature, r)) for _ in range(num_lora)])
        for a in self.lora_A:
            nn.init.kaiming_uniform_(a, a=5 ** 0.5)
        self.lora_alpha_params = nn.Parameter(torch.full((num_lora,), 1.0 / num_lora))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def lora_parameters(self):
        return [*self.lora_A, *self.lora_B, self.lora_alpha_params]

    def quantized_weight(self) -> Tensor:
        w = F.softmax(self.lora_alpha_params, dim=0)
        delta = sum(w[i] * (self.lora_B[i] @ self.lora_A[i]) for i in range(self.num_lora))
        return self.weight_quantizer(self.weight + delta)


class QuantLinearQKV(nn.Module):
    """A fused QKV ``nn.Linear`` split into three ``QuantLinear`` with independent quantizers."""

    def __init__(self, config: dict):
        super().__init__()
        self.q = QuantLinear(config)
        self.k = QuantLinear(config)
        self.v = QuantLinear(config)

    def set_param(self, linear: nn.Linear):
        out3, in_f = linear.weight.shape
        assert out3 % 3 == 0, "fused qkv Linear out_features must be divisible by 3"
        out_f = out3 // 3
        has_bias = linear.bias is not None
        ws = linear.weight.detach().chunk(3, dim=0)
        bs = linear.bias.detach().chunk(3, dim=0) if has_bias else (None, None, None)
        for sub, w, b in zip((self.q, self.k, self.v), ws, bs):
            lin = nn.Linear(in_f, out_f, bias=has_bias)
            with torch.no_grad():
                lin.weight.copy_(w)
                if has_bias:
                    lin.bias.copy_(b)
            sub.set_param(lin)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1)


class QuantConv2d(QuantBase):
    def set_param(self, conv: nn.Conv2d):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.conv_kwargs = dict(stride=conv.stride, padding=conv.padding,
                                dilation=conv.dilation, groups=conv.groups)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(self.act_quantizer(x), self.weight_quantizer(self.weight),
                        self.bias, **self.conv_kwargs)


# ---------------------------------------------------------------------------
# Token-group (group-conditional) activation quantization
# ---------------------------------------------------------------------------
class TokenGroupContext:
    """Holds the per-token group ids ``[B, L]`` predicted for the block that is
    currently executing. The pre-scan predictor hook writes them; every
    ``GroupedFakeQuantizerAct`` in the block reads them."""

    def __init__(self, num_groups: int):
        self.num_groups = num_groups
        self.group_ids: Optional[Tensor] = None

    def set_group_ids(self, ids: Tensor):
        self.group_ids = ids.detach()

    def clear(self):
        self.group_ids = None


class GroupedFakeQuantizerAct(FakeQuantizerAct):
    """Activation quantizer with one (lb, ub) pair per token group.

    ``token_dim`` is the axis of the input that indexes tokens:
    ``-2`` for ``[B, L, C]`` linear inputs, ``-1`` for ``[B, K, C, L]`` scan operands.
    """

    def __init__(self, bit: int, num_groups: int, ctx: TokenGroupContext,
                 token_dim: int = -2, search_steps: int = 100):
        super().__init__(bit=bit, search_steps=search_steps)
        self.ctx = ctx
        self.num_groups = num_groups
        self.token_dim = token_dim
        self.lower_bound = nn.Parameter(torch.zeros(num_groups))
        self.upper_bound = nn.Parameter(torch.ones(num_groups))

    def _token_axis(self, x: Tensor) -> int:
        return self.token_dim if self.token_dim >= 0 else x.dim() + self.token_dim

    def _group_ids(self, x: Tensor) -> Tensor:
        """``[B, L]`` ids from the context; all-zero if no matching ids are set."""
        B, L = x.shape[0], x.shape[self._token_axis(x)]
        ids = self.ctx.group_ids
        if ids is None or ids.dim() != 2 or ids.shape[0] != B or ids.shape[1] != L:
            return torch.zeros(B, L, device=x.device, dtype=torch.long)
        return ids.to(x.device)

    def _expand(self, ids: Tensor, x: Tensor) -> Tensor:
        shape = [1] * x.dim()
        shape[0], shape[self._token_axis(x)] = ids.shape
        return ids.view(shape)

    def _init_bounds(self, x: Tensor, ids: Tensor):
        td = self._token_axis(x)
        x_t = x if td == 1 else x.transpose(1, td)
        lb = torch.zeros(self.num_groups, device=x.device)
        ub = torch.zeros(self.num_groups, device=x.device)
        present = torch.zeros(self.num_groups, dtype=torch.bool, device=x.device)
        for g in torch.unique(ids).tolist():
            vals = x_t[ids == g]
            if vals.numel() == 0:
                continue
            lb[g], ub[g] = mse_bound_search(vals, self.bit, self.search_steps)
            present[g] = True
        if not present.all():  # groups absent from the calibration set fall back to tensor-wise bounds
            lb_all, ub_all = mse_bound_search(x, self.bit, self.search_steps)
            lb = torch.where(present, lb, torch.tensor(lb_all, device=x.device))
            ub = torch.where(present, ub, torch.tensor(ub_all, device=x.device))
        self.set_bounds(lb, ub)
        self.calibrated = True

    def forward(self, x: Tensor) -> Tensor:
        ids = self._group_ids(x)
        if not self.calibrated:
            self._init_bounds(x, ids)
            return x
        ids = self._expand(ids, x)
        return self._quantize(x, self.lower_bound[ids], self.upper_bound[ids])


def count_fake_quantizers(net: nn.Module) -> dict:
    weight = act = calibrated = 0
    for m in net.modules():
        if isinstance(m, FakeQuantizerBase):
            if isinstance(m, FakeQuantizerWeight):
                weight += 1
            else:
                act += 1
            calibrated += int(m.calibrated)
    return {"weight": weight, "act": act, "total": weight + act, "calibrated": calibrated}
