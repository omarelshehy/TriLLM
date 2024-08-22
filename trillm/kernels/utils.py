import torch
import triton
import triton.language as tl

@triton.jit
def triton_cosine_kernel(A, B, M, N, stride_ax, stride_ay, BLOCK_SIZE_A: tl.constexpr):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    num_pid_x = tl.cdiv(M, BLOCK_SIZE_A)

    pid = pid_x + pid_y * num_pid_x
    first_pid_x = (pid % num_pid_x) * BLOCK_SIZE_A
    first_pid_y = (pid // num_pid_x) * BLOCK_SIZE_A

    rm = first_pid_x + tl.arange(0, BLOCK_SIZE_A)
    rn = first_pid_y + tl.arange(0, BLOCK_SIZE_A)

    # Calculate addresses for A and B
    addr_A = A + (rm[:, None] * stride_ax + rn[None, :] * stride_ay)
    addr_B = B + (rm[:, None] * stride_ax + rn[None, :] * stride_ay)

    # Load from A
    acc = tl.zeros((BLOCK_SIZE_A, BLOCK_SIZE_A), dtype=tl.float32)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    acc += tl.load(addr_A, mask=mask)

    # Compute cosine
    acc_cos = tl.cos(acc)

    # Store to B
    tl.store(addr_B, acc_cos, mask=mask)

def triton_cos(A):
    B = torch.zeros(A.shape,dtype=A.dtype).to('cuda')
    grid = lambda META: (triton.cdiv(A.shape[0], META['BLOCK_SIZE_A']) * triton.cdiv(A.shape[1], META['BLOCK_SIZE_A']))
    M, N = A.shape
    triton_cosine_kernel[grid](A, B, M, N,stride_ax=A.stride(0), stride_ay=A.stride(1), BLOCK_SIZE_A=8)
    return B

@triton.jit
def triton_sine_kernel(A, B, M, N, stride_ax, stride_ay, BLOCK_SIZE_A: tl.constexpr):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    num_pid_x = tl.cdiv(M, BLOCK_SIZE_A)

    pid = pid_x + pid_y * num_pid_x
    first_pid_x = (pid % num_pid_x) * BLOCK_SIZE_A
    first_pid_y = (pid // num_pid_x) * BLOCK_SIZE_A

    rm = first_pid_x + tl.arange(0, BLOCK_SIZE_A)
    rn = first_pid_y + tl.arange(0, BLOCK_SIZE_A)

    # Calculate addresses for A and B
    addr_A = A + (rm[:, None] * stride_ax + rn[None, :] * stride_ay)
    addr_B = B + (rm[:, None] * stride_ax + rn[None, :] * stride_ay)

    # Load from A
    acc = tl.zeros((BLOCK_SIZE_A, BLOCK_SIZE_A), dtype=tl.float32)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    acc += tl.load(addr_A, mask=mask)

    # Compute sine
    acc_cos = tl.sin(acc)

    # Store to B
    tl.store(addr_B, acc_cos, mask=mask)

def triton_sin(A):
    B = torch.zeros(A.shape,dtype=A.dtype).to('cuda')
    grid = lambda META: (triton.cdiv(A.shape[0], META['BLOCK_SIZE_A']) * triton.cdiv(A.shape[1],META['BLOCK_SIZE_A']))
    M, N = A.shape
    triton_sine_kernel[triton.cdiv(A.shape[0], 8) , triton.cdiv(A.shape[1], 8)](A, B, M, N,stride_ax=A.stride(0), stride_ay=A.stride(1), BLOCK_SIZE_A=8)
    return B


@triton.jit
def ewm(A, B, C, M, ACTIVATION: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    # extract metaparameters
    pid = tl.program_id(axis=0)

    pid_m = pid * BLOCK_SIZE_M
    offsets = pid_m + tl.arange(0, BLOCK_SIZE_M)
    ptr_a = A
    ptr_b = B
    ptr_c = C
    mask = offsets < M
    a = tl.load(ptr_b + offsets,mask=mask)
    b = tl.load(ptr_a + offsets,mask=mask)
    acc = a * b
    if ACTIVATION == "silu":
        acc = acc.to(tl.float32) * tl.sigmoid(acc.to(tl.float32))
    tl.store(ptr_c + offsets, acc, mask=mask)

def ewm_triton(a, b, activation=""):
    # Check if matices are brodcastable
    if a.shape != b.shape:
        try:
            b = b.broadcast_to(a.shape)
            final_shape = a.shape
        except Exception as e:
            try:
                a = a.broadcast_to(b.shape)
                final_shape = b.shape
            except:
                raise ValueError("Matrices are not brodcastable")
    else:
        final_shape = a.shape
    #flatten to 1d
    a_flattend = a.flatten()
    b_flattend = b.flatten()
    M = a_flattend.shape[0]
    c = torch.empty_like(a_flattend,device=a.device, dtype=a.dtype)
    if activation:
        assert activation == 'silu', "only silu is available"  
    grid = lambda META: (triton.cdiv(a_flattend.shape[0], 16), )
    ewm[grid](a_flattend, b_flattend, c, M, ACTIVATION = activation, BLOCK_SIZE_M = 16)

    return c.reshape(final_shape).to(a.dtype)
