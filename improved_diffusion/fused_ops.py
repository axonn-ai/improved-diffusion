import torch
import torch.nn.functional as F


@torch.jit.script
def silu_dropout_fused(x, p: float, training: bool):
    out = x
    out = F.silu(out)
    out = F.dropout(out, p=p, training=training)
    return out


@torch.jit.script
def fused_residual_bias_add(x, bias, residual):
    return x + bias + residual


@torch.jit.script
def fused_residual_bias_add_prediv(x, bias, residual, div):
    return x + bias / div + residual / div
