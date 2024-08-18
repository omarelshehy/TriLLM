import torch
import triton
import triton.language as tl


@triton.jit
def add_matmul_2d(A, B, Bias, C, M, N, K, stride_am, stride_ak,
                  stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr):
    # extract metaparameters
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Number of programs in group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # Id of the group this program is in
    group_id = pid // num_pid_in_group

    # Row-id of the first program in the group
    first_pid_m = group_id * GROUP_SIZE_M
    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # *Within groups*, programs are ordered in a column-major order
    # Row-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % group_size_m)
    # Col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // group_size_m

    # # rm (resp. rn) denotes a range of indices
    # # for rows (resp. col) of C
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # # rk denotes a range of indices for columns
    # # (resp. rows) of A (resp. B)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # # the memory addresses of elements in the first block of
    # # A and B can be computed using numpy-style broadcasting
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    Bias = Bias + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_SIZE_K):
        a = tl.load(A)
        b = tl.load(B)
        # block level matrix multiplication
        acc += tl.dot(a, b)
        # increment pointers so that the next blocks of A and B
        # are loaded during the next iteration
        A += BLOCK_SIZE_K * stride_ak
        B += BLOCK_SIZE_K * stride_bk
    acc += tl.load(Bias)

    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C, acc, mask=mask)


@triton.jit
def matmul_2d(A, B, C, M, N, K, stride_am, stride_ak,
              stride_bk, stride_bn, stride_cm, stride_cn,
              BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
              GROUP_SIZE_M: tl.constexpr):
    # extract metaparameters
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Number of programs in group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # Id of the group this program is in
    group_id = pid // num_pid_in_group

    # Row-id of the first program in the group
    first_pid_m = group_id * GROUP_SIZE_M
    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # *Within groups*, programs are ordered in a column-major order
    # Row-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % group_size_m)
    # Col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // group_size_m

    # # rm (resp. rn) denotes a range of indices
    # # for rows (resp. col) of C
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # # rk denotes a range of indices for columns
    # # (resp. rows) of A (resp. B)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # # the memory addresses of elements in the first block of
    # # A and B can be computed using numpy-style broadcasting
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_SIZE_K):
        a = tl.load(A)
        b = tl.load(B)
        # block level matrix multiplication
        acc += tl.dot(a, b)
        # increment pointers so that the next blocks of A and B
        # are loaded during the next iteration
        A += BLOCK_SIZE_K * stride_ak
        B += BLOCK_SIZE_K * stride_bk

    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C, acc, mask=mask)


def add_mm_triton(a, b, Bias):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    assert c.shape[0] == Bias.shape[0]
    Bias = torch.broadcast_to(Bias, (M, N))
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    # print(grid((triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )))
    add_matmul_2d[grid](
        a, b, Bias, c,
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, GROUP_SIZE_M=4
    )
    return c


def mm_triton_2d(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    # print(grid((triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )))
    matmul_2d[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1)
    )
    return c


@triton.jit
def matmul_3d(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
              ACTIVATION: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
              GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group

    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    ra = pid_batch * M * K
    rb = pid_batch * K * N
    rc = pid_batch * M * N

    A = A + ra + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + rb + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_SIZE_K):
        a = tl.load(A, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_SIZE_K * stride_ak
        B += BLOCK_SIZE_K * stride_bk
    if ACTIVATION == "silu":
        acc = acc.to(tl.float32) * tl.sigmoid(acc.to(tl.float32))

    C = C + rc + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C, acc, mask=mask)


# prompt: run the code in the cell above for some example matrices

# Generate random matrices
def mm_triton_3d(a, b, activation = ""):
    # Check constraints.
    assert a.shape[2] == b.shape[1], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == b.dtype , "Matrices must be the same type"
    _, M, K = a.shape
    _, K, N = b.shape
    L = a.shape[0]
    # Allocates output.
    c = torch.zeros((L, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, 16) * triton.cdiv(N, 16), L)
    # print(grid((triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )))

    matmul_3d[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(1), a.stride(2),  #
        b.stride(1), b.stride(2),  #
        c.stride(1), c.stride(2),
        ACTIVATION= activation, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=16, GROUP_SIZE_M=4
    )
    return c
