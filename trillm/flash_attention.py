import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import math

@triton.jit
def flash_attn_kernel(Q, K, V, Out, m_m, l, B, H, M, N, stride_qb, stride_qh, stride_qm, stride_qd,
                      stride_kb, stride_kh, stride_kn, stride_kd, stride_vb, stride_vh, stride_vn, stride_vd,
                      stride_ob, stride_oh, stride_om, stride_od, stride_mm, stride_l, sm_scale,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, D: tl.constexpr):


    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)
    # print(pid,batch_id)

    off_b = batch_id // H
    off_h = batch_id % H

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    block_id_m = pid % num_pid_m
    block_id_n = pid // num_pid_m

    m = block_id_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    d_range = tl.arange(0, D)

    q = Q + off_b * stride_qb + off_h * stride_qh + m[:, None] * stride_qm + d_range[None, :] * stride_qd
    k = K + off_b * stride_kb + off_h * stride_kh + n[:, None] * stride_kn + d_range[None, :] * stride_kd
    v = V + off_b * stride_vb + off_h * stride_vh + n[:, None] * stride_vn +d_range[None, :] * stride_vd
    o = Out + off_b * stride_ob + off_h * stride_oh + m[:, None] * stride_om + d_range[None, :] * stride_od
    m_m = m_m + off_b * stride_mm + off_h * M + m[:, None]
    l = l + off_b * stride_l + off_h * M + m[:, None]

    acc = tl.zeros((BLOCK_SIZE_M, D), dtype=tl.float32)

    q_matrix = tl.load(q, mask=(m[:, None] < M) & (d_range[None, :] < D), other=0.0)
    m_i = tl.load(m_m, mask=(m[:, None] < M), other=0.0)
    l_i = tl.load(l, mask=(m[:, None] < M), other=0.0)
    # print(q_matrix)

    for i in range(0, block_id_m + 1):
        k_block = k + i * BLOCK_SIZE_N * stride_kn
        v_block = v + i * BLOCK_SIZE_N * stride_vn
        k_matrix = tl.load(k_block, mask=(n[:, None] < N) & (d_range[None, :] < D), other=float("-inf"))
        v_matrix = tl.load(v_block)
        # print(k_matrix)
        # print(v_matrix)
        product = tl.dot(q_matrix, tl.trans(k_matrix))
        product += tl.where(m[:, None] >= ((i * BLOCK_SIZE_N) + n)[None, :], 0, float("-inf"))
        # print(m[:, None] >= ((i * BLOCK_SIZE_N) + n)[None, :], 0, float("-inf"))
        # print(product)
        product *= sm_scale
        # print(product)


        m_telda_ij = tl.max(product, axis=1, keep_dims=True)
        p_telda_ij = tl.exp(product - m_telda_ij).to(v_matrix.dtype)
        l_telda_ij = tl.sum(p_telda_ij, axis=1, keep_dims=True)
        m_i_new = tl.maximum(m_i, m_telda_ij)

        f_1 = tl.exp(m_i - m_i_new)
        f_2 = tl.exp(m_telda_ij - m_i_new)
        l_i_new = f_1 * l_i + f_2 * l_telda_ij

        acc = (l_i * f_1 * acc) + (f_2  * tl.dot(p_telda_ij,v_matrix))
        acc = acc / l_i_new
        l_i = l_i_new
        m_i = m_i_new

    tl.store(o, acc, mask=(m[:, None] < M) & (d_range[None, :] < D))
    tl.store(l, l_i)
    tl.store(m_m, m_i)

def flash_attn_triton(Q, K, V):
    assert Q.shape[-1] == K.shape[-1] == V.shape[-1], "incompatible dimensions"
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape
    Output = torch.empty_like(Q, device=Q.device, dtype=torch.float16)
    m_m = torch.full((B, H, M, 1), float("-inf"), device=Q.device, dtype=torch.float32)
    l = torch.zeros((B, H, M, 1), device=Q.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),B*H)
    sm_scale = 1 / math.sqrt(D)
    if M < 32:
        Q,K,V = F.pad(Q, (0, 0, 0, 32-M)), F.pad(K, (0, 0, 0, 32-M)), F.pad(V, (0, 0, 0, 32-M))
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 32

    flash_attn_kernel[grid](
        Q, K, V, Output,
        m_m, l, B, H, M, N,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Output.stride(0), Output.stride(1), Output.stride(2), Output.stride(3),
        m_m.stride(0), l.stride(0), sm_scale,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, D=D)

    return Output
