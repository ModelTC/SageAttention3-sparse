"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa
import fp4attn_cuda
import fp4quant_cuda
from .sparge import get_block_map_meansim


@triton.jit
def group_mean_kernel(
    q_ptr,          
    q_out_ptr,      
    qm_out_ptr,     
    B, H, L, D: tl.constexpr,    
    stride_qb, stride_qh, stride_ql, stride_qd,  
    stride_qmb, stride_qmh, stride_qml, stride_qmd,  
    GROUP_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)
    
    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    
    # 添加边界检查
    mask = offsets < L

    q_offsets = pid_b * stride_qb + pid_h * stride_qh + offsets[:, None] * stride_ql + tl.arange(0, D)[None, :] * stride_qd
    q_group = tl.load(q_ptr + q_offsets, mask=mask[:, None], other=0.0)

    # 计算实际元素数量
    actual_group_size = tl.minimum(GROUP_SIZE, L - group_start)
    qm_group = tl.sum(q_group, axis=0) / actual_group_size # use actual_group_size is more acurate.

    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group, mask=mask[:, None])

    qm_offset = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + tl.arange(0, D) * stride_qmd
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor):
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = (L + GROUP_SIZE - 1) // GROUP_SIZE
    
    q_out = torch.empty_like(q)  # [B, H, L, D]
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype) 
    
    grid = (B, H, num_groups)
    
    group_mean_kernel[grid](
        q, q_out, qm,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        GROUP_SIZE=GROUP_SIZE
    )
    return q_out, qm


def pad_128(x):
    L = x.size(3)
    pad_len = (128 - L % 128) % 128
    if pad_len == 0:
        return x
    return F.pad(x, (0, pad_len), value=0)


def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    pad_N = int((N + 128 - 1) // 128 * 128)
    packed_fp4 = torch.empty((B, H, pad_N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, pad_N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    pad_N = int((N + 128 - 1) // 128 * 128)
    packed_fp4 = torch.empty((B, H, pad_N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, pad_N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    pad_N = int((N + 128 - 1) // 128 * 128)
    packed_fp4 = torch.empty((B, H, D, pad_N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, pad_N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def blockscaled_fp4_attn(qlist: Tuple, 
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         KL: int,
                         lut: torch.Tensor = None,
                         valid_block_num: torch.Tensor = None,
                         is_causal: bool = False, 
                         is_sparse: bool = False,
                         per_block_mean: bool = True,
                         is_bf16: bool = True
                        ):
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return fp4attn_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, lut, valid_block_num, KL, None, softmax_scale, is_causal, is_sparse, per_block_mean, is_bf16)


def sageattn3_blackwell(q, k, v, attn_mask = None, is_causal = False, is_sparse = False, per_block_mean = True, **kwargs):
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)

    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16

    km = k.mean(dim=-2, keepdim=True)
    k -= km

    if is_sparse:
        lut, valid_block_num = get_block_map_meansim(q, k, is_causal = is_causal, cdfthreshd=None, topk=0.2, return_lut=True, BLKQ=128, BLKK=128)
    else:
        lut, valid_block_num = None, None

    if per_block_mean:
        q, qm = triton_group_mean(q)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm

    delta_s = torch.matmul(qm, k.transpose(-2, -1))
    delta_s = pad_128(delta_s)

    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)

    o_fp4 = blockscaled_fp4_attn(
    qlist_from_cuda,
    klist_from_cuda, 
    vlist_from_cuda,
    delta_s,
    KL,
    lut,
    valid_block_num,
    is_causal,
    is_sparse,
    per_block_mean,
    is_bf16
    )[0][:, :, :QL, :].contiguous()

    return o_fp4
