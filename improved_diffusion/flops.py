import torch


def attention_flops(N, H, W, C, act_ckp=True):
    flops_attention_qkv = N * (2 * (H * W) ** 2 * C)
    if act_ckp:
        return flops_attention_qkv * 4
    else:
        return flops_attention_qkv * 3


def conv3x3_flops(N, Cin, Cout, H, W, act_ckp=True):
    ##calcuates flops for a 3x3 conv layer
    act_ckp=False
    flops_conv = ((3 * 3 * Cin * Cout * 2) * H * W) * N
    if act_ckp:
        return flops_conv * 4
    else:
        return flops_conv * 3
