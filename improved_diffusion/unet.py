from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .nn import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from .fused_ops import (
    silu_dropout_fused,
    fused_residual_bias_add,
    fused_residual_bias_add_prediv,
)


from .flops import conv3x3_flops

# from axonn_tensor_parallel.layers import Conv2d, Linear
# from axonn_tensor_parallel.communication_ops import all_reduce
# from axonn_tensor_parallel import isAsync, log_dist

# do we need this?
# from axonn_tensor_parallel.layers.global_vars import Method

# ok to replace with these imports instead?
from axonn.intra_layer.conv import Conv2d
from axonn.intra_layer.fully_connected import Linear
from axonn.intra_layer.communication import _all_reduce

import asyncio
def isAsync(someFunc):
    return asyncio.iscoroutinefunction(someFunc)

def log_dist(msg, ranks=[]):
    assert dist.is_initialized()
    if dist.get_rank() in ranks:
        print(f"Rank {dist.get_rank()} : {msg}")

try: # experimental right now
    from flash_attn.flash_attention import FlashAttention
except:
    pass

from torch.utils.checkpoint import checkpoint

# do we need these?
# from axonn_tensor_parallel.residual import residual_prologue
# from axonn_tensor_parallel import backward_scheduler

# from axonn import axonn as ax

def checkpoint_wrapper(f, *args, use_checkpoint=False):
    if use_checkpoint:
        return checkpoint(f, *args)
    else:
        return f(*args)

def divide(a, b):
    assert a%b == 0
    return a // b

class fast_silu_dropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        return silu_dropout_fused(x, self.dropout, self.training)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        #log_dist(f'upsample input - {x.reshape(-1)[:-5]}' ,[0])
        if self.use_conv:
            assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        #log_dist(f'upsample output - {x.reshape(-1)[:-5]}' ,[0])
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, handle=None):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(2, kernel_size=2, stride=2)

    def forward(self, x):
        if self.use_conv:
            assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        handle=None,
        async_comm=False,
        # method=Method.AGARWAL,
        mid_channels=32
    ):
        super().__init__()
        log_dist(f'Creating res block IC={channels} OC={out_channels}', [0])
        assert not use_scale_shift_norm
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.handle = handle

        # row group is outer in axonn?
        inner_parallel_group = handle.outer_intra_layer_parallel_group
        inner_parallel_size = dist.get_world_size(inner_parallel_group)
        inner_parallel_rank = dist.get_rank(inner_parallel_group)

        # column group is inner in axonn?
        outer_parallel_group = handle.inner_intra_layer_parallel_group
        outer_parallel_size = dist.get_world_size(outer_parallel_group)
        depth_parallel_group = handle.depth_intra_layer_parallel_group

        #self.inner_parallel_rank = inner_parallel_rank
        #self.inner_parallel_size = inner_parallel_size

        """
        if method in [Method.CAI_3D, Method.DRYDEN]:
            self.in_layers = nn.Sequential(
                #normalization(channels),
                normalization(divide(channels, inner_parallel_size * outer_parallel_size)),
                nn.SiLU(),
            )
        else:
        """
        print(channels, inner_parallel_size)
        self.in_layers = nn.Sequential(
            #normalization(channels),
            normalization(divide(channels, inner_parallel_size)),
            nn.SiLU(),
        )

        #conv_nd(dims, channels, self.out_channels, 3, padding=1, bias=False)
        self.in_conv = Conv2d(
                    # inner_process_group=inner_parallel_group,
                    # outer_process_group=outer_parallel_group,
                    # depth_process_group=depth_parallel_group,
                    in_channels=channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                    # transpose_groups=False,
                    # async_comm=async_comm,
                    # method=method,
                )
        """
        if method in [Method.CAI_3D, Method.DRYDEN]:
            self.in_conv_bias = nn.Parameter(
                torch.zeros(divide(self.out_channels, outer_parallel_size * inner_parallel_size), 1, 1), requires_grad=True
            )
        else:
        """
        self.in_conv_bias = nn.Parameter(
            torch.zeros(divide(self.out_channels, outer_parallel_size), 1, 1), requires_grad=True
        )

        self.emb_layers_silu = nn.SiLU()
        self.emb_layers_linear = Linear(
                # inner_process_group=inner_parallel_group,
                # outer_process_group=outer_parallel_group,
                # depth_process_group=depth_parallel_group,
                in_features=emb_channels,
                out_features=self.out_channels
                # transpose_groups=False,
                # async_comm=async_comm,
                # bias=False,
                # method=method
                )

        """
        if method in [Method.CAI_3D, Method.DRYDEN]:
            self.out_layers = nn.Sequential(
                #normalization(self.out_channels),
                #nn.BatchNorm2d(divide(self.out_channels, outer_parallel_size * inner_parallel_size)),
                normalization(divide(self.out_channels, outer_parallel_size * inner_parallel_size)),
                fast_silu_dropout(self.dropout),
            )
        else:
        """
        self.out_layers = nn.Sequential(
            #normalization(self.out_channels),
            normalization(divide(self.out_channels, outer_parallel_size)),
            fast_silu_dropout(self.dropout),
        )

            
        self.out_conv = zero_module( Conv2d(
                    # inner_process_group=inner_parallel_group,
                    # outer_process_group=outer_parallel_group,
                    # depth_process_group=depth_parallel_group,
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    transpose=True,
                    bias=False,
                    padding=1
                    # transpose_groups=transpose_condition(),
                    # async_comm=async_comm,
                    # method=method
                )
            )

        """
        if method in [Method.CAI_3D, Method.DRYDEN]:
            self.out_conv_bias = nn.Parameter(
                    torch.zeros(divide(self.out_channels, inner_parallel_size * outer_parallel_size), 1, 1), requires_grad=True
                )
        else:
        """
        self.out_conv_bias = nn.Parameter(
            torch.zeros(divide(self.out_channels, inner_parallel_size), 1, 1), requires_grad=True
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            raise NotImplementedError
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1, bias=False
            )
        else:
            #raise  NotImplementedError
            self.skip_connection = Conv2d(
                # inner_process_group=inner_parallel_group,
                # outer_process_group=outer_parallel_group,
                # depth_process_group=depth_parallel_group,
                in_channels=channels,
                out_channels=self.out_channels,
                kernel_size=1,
                # padding=0,
                bias=False
                # transpose_groups=False,
                # async_comm=async_comm,
                # method=method,
                # non_transposed_output=True
                )
            #self.skip_connection_2 = Conv2d(
            #    inner_process_group=inner_parallel_group,
            #    outer_process_group=outer_parallel_group,
            #    depth_process_group=depth_parallel_group,
            #    in_channels=mid_channels,
            #    out_channels=self.out_channels,
            #    kernel_size=1,
            #    padding=0,
            #    bias=False,
            #    transpose_groups=transpose_condition(method),
            #    async_comm=async_comm,
            #    method=method
            #    )
        self.async_comm = async_comm
            #conv_nd(dims, channels, self.out_channels, 1)

    #def forward(self, x, emb):
    #    """
    #    Apply the block to a Tensor, conditioned on a timestep embedding.
#
 #       :param x: an [N x C x ...] Tensor of features.
 #       :param emb: an [N x emb_channels] Tensor of timestep embeddings.
 #       :return: an [N x C x ...] Tensor of outputs.
 #       """
 #       return checkpoint(
 #           self._forward, (x, emb), self.parameters(), self.use_checkpoint
 #       )

    def forward(self, x, emb):
        # if self.async_comm:
        # x = residual_prologue(x)
        h = checkpoint_wrapper(self.in_layers, x, use_checkpoint=self.use_checkpoint)
        h = self.in_conv(h)
        emb_out = self.emb_layers_silu(emb).type(h.dtype)
        emb_out = self.emb_layers_linear(emb_out)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            raise NotImplementedError
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            def _pre_out_conv(h, bias, emb, norm):
                h = fused_residual_bias_add(h, bias, emb)
                return norm(h)
            h = checkpoint_wrapper(_pre_out_conv, h, self.in_conv_bias, emb_out, self.out_layers, use_checkpoint=self.use_checkpoint)
            h = self.out_conv(h)
        """
        if isAsync(self.skip_connection.forward):
            skip_output = await self.skip_connection(x)
            #skip_output = await self.skip_connection_2(skip_output)
        else:
        """
        skip_output = self.skip_connection(x)
        return fused_residual_bias_add(skip_output, h, self.out_conv_bias)


class _AllReduce(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_, process_group=None):
        dist.all_reduce(input_.contiguous(), group=process_group)
        return input_

    @staticmethod
    def forward(ctx, input_, process_group=None):
        dist.all_reduce(input_.contiguous(), group=process_group)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False, handle=None, use_flash_attention=True, async_comm=False):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint

        inner_parallel_group = handle.get_row_parallel_group()
        inner_parallel_size = dist.get_world_size(inner_parallel_group)
        inner_parallel_rank = dist.get_rank(inner_parallel_group)

        outer_parallel_group = handle.get_column_parallel_group()
        outer_parallel_size = dist.get_world_size(outer_parallel_group)
        depth_parallel_group = handle.get_depth_parallel_group()

        self.num_heads = divide(num_heads, outer_parallel_size) 
        self.inner_parallel_group = inner_parallel_group
        self.outer_parallel_group = outer_parallel_group

        # self.method = method

        """
        if self.method in [Method.DRYDEN, Method.CAI_3D]:
            self.norm = normalization(divide(channels, inner_parallel_size * outer_parallel_size)) #normalization(channels)
        else:
        """
        self.norm = normalization(divide(channels, inner_parallel_size)) #normalization(channels)
        
        #assert method == Method.AGARWAL, "only implemented for AGARWAL"
        self.qkv = Conv2d(
                    # inner_process_group=inner_parallel_group,
                    # outer_process_group=outer_parallel_group,
                    # depth_process_group=depth_parallel_group,
                    in_channels=channels,
                    out_channels=channels * 3,
                    kernel_size=1,
                    # padding=0,
                    bias=False
                    # transpose_groups=False,
                    # async_comm=async_comm,
                    # method=method
                )

        """
        if self.method in [Method.DRYDEN, Method.CAI_3D]:       
            self.qkv_bias = nn.Parameter(
                torch.zeros(divide(channels * 3, outer_parallel_size * inner_parallel_size), 1, 1), requires_grad=True
            )
        else:
        """
        self.qkv_bias = nn.Parameter(
            torch.zeros(divide(channels * 3, outer_parallel_size), 1, 1), requires_grad=True
        )
        #conv_nd(1, channels, channels * 3, 1)
        if use_flash_attention:
            raise NotImplementedError
            self.attention = FlashAttention()
        else:
            self.attention = QKVAttention(inner_parallel_group, outer_parallel_group)
        self.use_flash_attention = use_flash_attention

        self.proj_out = zero_module(
                    Conv2d(
                        # inner_process_group=inner_parallel_group,
                        # outer_process_group=outer_parallel_group,
                        # depth_process_group=depth_parallel_group,
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=1,
                        transpose=True,
                        # padding=0,
                        bias=False
                        # transpose_groups=method != Method.DRYDEN,
                        # transpose_groups=True # because agarwal method by default?
                        # async_comm=async_comm,
                        # method=method
                    )
                )

        """
        if self.method in [Method.DRYDEN, Method.CAI_3D]:
            self.proj_out_bias = nn.Parameter(
                torch.zeros(divide(channels, inner_parallel_size * outer_parallel_size), 1, 1), requires_grad=True
            )
        else:
        """
        self.proj_out_bias = nn.Parameter(
            torch.zeros(divide(channels, inner_parallel_size), 1, 1), requires_grad=True
        )
        
        self.async_comm = async_comm

        #def forward(self, x):
        #return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _internal_compute(self, qkv, b, c):
        qkv = qkv + self.qkv_bias
        if self.use_flash_attention:
            qkv_reshaped = qkv.reshape(b, -1, qkv.shape[2]*qkv.shape[3])
            qkv_reshaped = torch.transpose(qkv_reshaped, 1, 2)
            B, S = qkv_reshaped.shape[0], qkv_reshaped.shape[1]
            qkv_reshaped = qkv_reshaped.reshape(B, S, 3, self.num_heads, -1)
            h = self.attention(qkv_reshaped)[0]
            h = h.reshape(B, S, -1)
            h = torch.transpose(h, 1, 2)
            h = h.reshape(B, -1, qkv.shape[2], qkv.shape[3])
        else:
            qkv_reshaped = qkv.reshape(b * self.num_heads, -1, qkv.shape[2]*qkv.shape[3])
            h = self.attention(qkv_reshaped)
            h = h.reshape(b, -1, qkv.shape[2], qkv.shape[3])
        return h

    def forward(self, x):
        # if self.async_comm:
        # x = residual_prologue(x)
        h = self.norm(x)
        b, c, *spatial = h.shape
        qkv = self.qkv(h)
        h = checkpoint_wrapper(self._internal_compute, qkv, b, c, use_checkpoint=self.use_checkpoint) 
        h = self.proj_out(h)
        return fused_residual_bias_add(x, h, self.proj_out_bias)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def __init__(self, inner_pg, outer_pg):
        super(QKVAttention, self).__init__()
        self.inner_pg = inner_pg
        self.outer_pg = outer_pg
        # self.method = method

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        """
        if self.method in [Method.DRYDEN, Method.CAI_3D]:
           weight = _AllReduce.apply(weight, self.inner_pg)
        """
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial**2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])

def transpose_condition():
    # return method != Method.DRYDEN
    return True  # default is agarwal?

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        async_comm=False,
        use_flash_attention=False,
        handle=None,
        # method=Method.AGARWAL,
        mid_channels=32
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
       
        assert dims == 2, "only supported 2D diffusion"
        self.handle = handle
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        # self.method = method

        # previously this was getting row group and in axonn row is outer?
        inner_parallel_group = handle.outer_intra_layer_parallel_group
        inner_parallel_size = dist.get_world_size(inner_parallel_group)
        inner_parallel_rank = dist.get_rank(inner_parallel_group)

        # previously this was getting column and in axonn column is inner?
        outer_parallel_group = handle.inner_intra_layer_parallel_group
        outer_parallel_size = dist.get_world_size(outer_parallel_group)

        depth_parallel_group = handle.inner_intra_layer_parallel_group

        self.inner_parallel_rank = inner_parallel_rank
        self.inner_parallel_size = inner_parallel_size
        self.inner_parallel_group = inner_parallel_group


        time_embed_dim = model_channels * 4
        self.time_embed_l1 = Linear(
                # inner_parallel_group,
                # outer_parallel_group,
                # depth_parallel_group,
                in_features=model_channels,
                out_features=time_embed_dim
                # transpose_groups=False,
                # async_comm=async_comm,
                # bias=False,
                # method=method
                )
        self.time_embed_l2 = nn.SiLU()
            
        self.time_embed_l3 = Linear(
                # inner_parallel_group,
                # outer_parallel_group,
                # depth_parallel_group,
                in_features=time_embed_dim,
                out_features=time_embed_dim,
                transpose=True
                # transpose_groups=transpose_condition(),
                # async_comm=async_comm,
                # bias=False,
                # method=method,
                )
            
            

        if self.num_classes is not None:
            raise NotImplementedError
            self.label_emb = nn.Embedding(num_classes, divide(time_embed_dim, inner_parallel_size))
        """
        if self.method in [Method.CAI_3D, Method.DRYDEN]:
            first_conv = conv_nd(dims, in_channels, divide(model_channels, inner_parallel_size * outer_parallel_size), 3, padding=1)
        else:
        """
        first_conv = conv_nd(dims, in_channels, divide(model_channels, inner_parallel_size), 3, padding=1)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    first_conv,
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        conv_flops = 0
        #async_comm=False
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        handle=handle,
                        async_comm=async_comm,
                        # method=method,
                        mid_channels=mid_channels
                    )
                ]
                first_conv_flops = conv3x3_flops(
                    1,
                    ch,
                    mult * model_channels,
                    1 / ds,
                    1 / ds,
                    act_ckp=self.use_checkpoint,
                )
                
                if ch != mult * model_channels:
                    first_conv_flops += conv3x3_flops(
                        1,
                        ch,
                        mult * model_channels,
                        1 / ds,
                        1 / ds,
                        act_ckp=self.use_checkpoint,
                    )/9



                second_conv_flops = conv3x3_flops(
                    1,
                    mult * model_channels,
                    mult * model_channels,
                    1 / ds,
                    1 / ds,
                    act_ckp=self.use_checkpoint,
                )

                conv_flops += first_conv_flops + second_conv_flops

                ch = mult * model_channels
                if ds in attention_resolutions:
                    raise NotImplementedError
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads, handle=handle, use_flash_attention=use_flash_attention, async_comm=async_comm
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims, handle=handle))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                handle=handle,
                async_comm=async_comm,
                # method=method,
                mid_channels=mid_channels
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, handle=handle, use_flash_attention=use_flash_attention, async_comm=async_comm),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                handle=handle,
                async_comm=async_comm,
                # method=method,
                mid_channels=mid_channels
            ),
        )

        conv_flops += 4 * conv3x3_flops(
            1, ch, ch, 1 / ds, 1 / ds, act_ckp=self.use_checkpoint
        )

        conv_flops +=  conv3x3_flops(
            1, ch, 3*ch, 1 / ds, 1 / ds, act_ckp=self.use_checkpoint
        )/9

        conv_flops +=  conv3x3_flops(
            1, ch, ch, 1 / ds, 1 / ds, act_ckp=self.use_checkpoint
        )/9

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ch_res = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ch_res, 
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        handle=handle,
                        async_comm=async_comm,
                        # method=method,
                        mid_channels=mid_channels
                    )
                ]

                conv_flops += conv3x3_flops(
                    1,
                    ch + ch_res,
                    mult * model_channels,
                    1 / ds,
                    1 / ds,
                    act_ckp=self.use_checkpoint,
                )
                if (ch + ch_res) != mult * model_channels:
                    ##skip connection flops
                    conv_flops += conv3x3_flops(
                        1,
                        ch+ch_res,
                        mult * model_channels,
                        1 / ds,
                        1 / ds,
                        act_ckp=self.use_checkpoint)/9
                conv_flops += conv3x3_flops(
                    1,
                    mult * model_channels,
                    mult * model_channels,
                    1 / ds,
                    1 / ds,
                    act_ckp=self.use_checkpoint,
                )


                ch = model_channels * mult
                if ds in attention_resolutions:
                    raise NotImplementedError
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            handle=handle,
                            use_flash_attention=use_flash_attention,
                            async_comm=async_comm,
                            # method=method
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        """
        if method in [Method.CAI_3D, Method.DRYDEN]:
            self.out = nn.Sequential(
                normalization(divide(ch, inner_parallel_size * outer_parallel_size)),
                nn.SiLU(),
            )
        else:
        """
        self.out = nn.Sequential(
            normalization(divide(ch, inner_parallel_size)),
            nn.SiLU(),
        )
        
        """
        if method in [Method.CAI_3D, Method.DRYDEN]:
            self.out_conv = zero_module(conv_nd(dims, divide(model_channels, inner_parallel_size * outer_parallel_size), out_channels, 3, padding=1, bias=False))
        else:
        """
        self.out_conv = zero_module(conv_nd(dims, divide(model_channels, inner_parallel_size), out_channels, 3, padding=1, bias=False))
        
        self.out[-1].dont_touch = True
        self.conv_flops = conv_flops
        self.async_comm = async_comm
        #self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'dont_touch') and m.dont_touch:
                    continue
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def flops(self, N, H, W):
        return N * H * W * self.conv_flops 

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        handle = self.handle 
        """
        if self.method == Method.CAI_3D:
            emb_partition = ax.config.intra_layer_parallel_rank
            num_partitions = ax.config.G_intra
        else:
        """
        emb_partition = self.inner_parallel_rank
        num_partitions = self.inner_parallel_size

        emb = timestep_embedding(timesteps, 
                self.model_channels,
                partition=emb_partition,
                num_partitions=num_partitions, 
                ).to(x.dtype)
        
        # emb = await self.time_embed_l1(emb)
        emb = self.time_embed_l1(emb)
        emb = self.time_embed_l2(emb)
        # emb = await self.time_embed_l3(emb)
        emb = self.time_embed_l3(emb)
        
        if self.num_classes is not None:
            raise NotImplementedError
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        h = x.type(self.inner_dtype)
        iters = 0
        # if self.async_comm:
        # emb = residual_prologue(emb)

        #log_dist(f'First five elements of emb - {emb.reshape(-1)[:5]}', [0])
        #log_dist(f'Before input blocks First five elements of h - {h.reshape(-1)[10:15]}', [0])
        
        for module in self.input_blocks:
            # h = await module(h, emb)
            h = module(h, emb)
            # if self.async_comm:
            # h = residual_prologue(h)
            hs.append(h)
            iters += 1
        #log_dist(f'After input blocks First five elements of h - {h.reshape(-1)[10:15]}', [0])
        h = self.middle_block(h,emb)
       # log_dist(f'After middle blocks First five elements of h - {h.reshape(-1)[10:15]}', [0])
        step=0
        for module in self.output_blocks:
            prev = hs.pop()#.detach()#.detach()
            cat_in = th.cat([h, prev], dim=1)
            h = module(cat_in, emb)
            #log_dist(f'After output block number {step} {module.__class__} First five elements of h - {h.reshape(-1)[10:15]}', [0])
            step += 1
        h = h.type(x.dtype)
        h = self.out(h)

        """
        if self.method == Method.CAI_3D:
            out = await all_reduce(self.out_conv(h), ax.comm_handle.intra_layer_group, async_comm=False)
        else:
            out = await all_reduce(self.out_conv(h), self.inner_parallel_group, async_comm=self.async_comm) 
        """
        # self.async_comm maybe nees to be replaced with overlap_comm
        out = _all_reduce(self.out_conv(h), self.inner_parallel_group, self.async_comm)
        
        # if self.async_comm:
        # backward_scheduler.change_context()
        #log_dist(f'Finished FW pass and changed context to {backward_scheduler.get_context()}', [0])
        return out

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)
