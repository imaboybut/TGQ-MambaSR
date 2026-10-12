"""Where Token-Group Quantization attaches to each backbone.

A ``BackboneSpec`` names (i) the residual blocks used for feature distillation
and pseudo-label construction, (ii) the group-conditional quantization sites
inside each block, and (iii) the pre-scan feature the group predictor reads.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

import torch.nn as nn


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    block_pattern: str              # regex whose group(1) is the dotted prefix of one residual block
    linear_sites: Tuple[str, ...]   # suffixes of QuantLinear/QuantConv2d modules whose input is group-quantized
    scan_sites: Tuple[str, ...]     # attribute names of scan-operand quantizers (token axis last)
    predictor_module: str           # sub-module of a block whose input/output feeds the predictor
    predictor_hook: str             # "input" (forward_pre_hook) or "output" (forward_hook)

    def block_of(self, module_name: str):
        m = re.match(self.block_pattern + r"\.", module_name)
        return m.group(1) if m else None


# MambaIRv2: eight group-conditional sites per ASSM block (three downstream
# linears + five selective-scan operands); the predictor reads the ASSM input.
MAMBAIRV2 = BackboneSpec(
    name="mambairv2",
    block_pattern=r"^(layers\.\d+\.residual_group\.layers\.\d+)",
    linear_sites=("assm.out_proj", "convffn2.fc1", "convffn2.fc2"),
    scan_sites=("mymodule_xs", "mymodule_dts", "mymodule_delta", "mymodule_Bs", "mymodule_Cs"),
    predictor_module="assm",
    predictor_hook="input",
)

# SwinIR-light: no scan stage; the post-attention MLP linears are grouped and the
# predictor reads the pre-block (norm1) feature of the same block.
SWINIR = BackboneSpec(
    name="swinir",
    block_pattern=r"^(layers\.\d+\.residual_group\.blocks\.\d+)",
    linear_sites=("mlp.fc1", "mlp.fc2"),
    scan_sites=(),
    predictor_module="norm1",
    predictor_hook="output",
)

SPECS = {s.name: s for s in (MAMBAIRV2, SWINIR)}


def iter_blocks(model: nn.Module, spec: BackboneSpec) -> List[Tuple[str, nn.Module]]:
    """Residual blocks in forward order as ``(dotted_name, module)``."""
    pat = re.compile(spec.block_pattern + r"$")
    return [(n, m) for n, m in model.named_modules() if pat.match(n)]
