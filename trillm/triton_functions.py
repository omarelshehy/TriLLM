from torch.autograd import Function
from trillm.kernels.flash_attention import flash_attn_triton
from trillm.kernels.utils import ewm_triton, triton_cos, triton_sin
from trillm.kernels.gemm import mm_triton_3d


class _sdpa_flash_attention(Function):
    @staticmethod
    def forward(ctx, query, key, value):
        out = flash_attn_triton(query, key, value)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        pass


class _bmm(Function):
    @staticmethod
    def forward(ctx, x, y, activation=''):
        out = mm_triton_3d(x, y, activation)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        pass


class _element_wise_multiplication(Function):
    @staticmethod
    def forward(ctx, x, y):
        out = ewm_triton(x, y)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        pass


class _cos_func(Function):
    @staticmethod
    def forward(ctx, x):
        out = triton_cos(x)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        pass


class _sin_func(Function):
    @staticmethod
    def forward(ctx, x):
        out = triton_sin(x)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        pass


triton_sdpa_flash_attention = _sdpa_flash_attention.apply
triton_bmm = _bmm.apply
triton_element_wise_multiplication = _element_wise_multiplication.apply
triton_cos_func = _cos_func.apply
triton_sin_func = _sin_func.apply
