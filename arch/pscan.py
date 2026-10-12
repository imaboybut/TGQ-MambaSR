"""Blelloch parallel scan used by selective_scan_ref:
pscan(A, X) returns H where H[t] = sum_{s<=t} (prod_{s<k<=t} A[k]) * X[s]."""

import math
import torch
from torch.autograd import Function
import torch.nn.functional as F


def npo2(length: int) -> int:
    return 2 ** math.ceil(math.log2(length))


def pad_npo2(X: torch.Tensor) -> torch.Tensor:
    L = X.shape[1]
    L2 = npo2(L)
    if L2 == L:
        return X
    pad_len = L2 - L
    pad = X.new_zeros(X.shape[0], pad_len, X.shape[2], X.shape[3])
    return torch.cat([X, pad], dim=1)


class PScan(Function):
    @staticmethod
    def _pscan_inplace(A: torch.Tensor, X: torch.Tensor):
        # A, X : (B, D, L, N)
        B, D, L, _ = A.shape
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.shape[2]
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        if Xa.shape[2] == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])
            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2] * Xa[:, :, 1]))
        elif Xa.shape[2] == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        step = 2 ** (num_steps - 2)
        Aa = A[:, :, step - 1:L:step]
        Xa = X[:, :, step - 1:L:step]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2**k - 1:L:2**k]
            Xa = X[:, :, 2**k - 1:L:2**k]

            T = Xa.shape[2]
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def _pscan_rev_inplace(A: torch.Tensor, X: torch.Tensor):
        B, D, L, _ = A.shape
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.shape[2]
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        if Xa.shape[2] == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1] + Aa[:, :, 1] * Xa[:, :, 2]))
        elif Xa.shape[2] == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        Aa = A[:, :, 0:L:2**(num_steps - 2)]
        Xa = X[:, :, 0:L:2**(num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]

            T = Xa.shape[2]
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in: torch.Tensor, X_in: torch.Tensor):
        # A_in, X_in : (B, L, D, N)
        L = X_in.shape[1]

        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in).clone()
            X = pad_npo2(X_in).clone()

        A = A.transpose(1, 2)  # (B, D, L, N)
        X = X.transpose(1, 2)

        PScan._pscan_inplace(A, X)

        ctx.save_for_backward(A_in, X)

        out = X.transpose(1, 2)[:, :L]
        return out

    @staticmethod
    def backward(ctx, grad_output_in: torch.Tensor):
        A_in, X_scanned = ctx.saved_tensors

        L = grad_output_in.shape[1]

        if L == npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in)
            A_in = pad_npo2(A_in)

        grad_output = grad_output.transpose(1, 2)  # (B, D, L, N)
        A_in_t = A_in.transpose(1, 2)

        # A shifted: append zero at last timestep
        B_, D_, Lpad, N_ = A_in_t.shape
        A_shift = A_in_t[:, :, 1:, :]
        zero_tail = A_in_t.new_zeros(B_, D_, 1, N_)
        A = torch.cat([A_shift, zero_tail], dim=2).contiguous()

        PScan._pscan_rev_inplace(A, grad_output)

        # grad wrt A : Q[:, :, 1:] += X[:-1] * grad_output[1:]
        X = X_scanned
        Q = torch.zeros_like(X)
        inter = X[:, :, :-1, :] * grad_output[:, :, 1:, :]
        inter = torch.nan_to_num(inter)
        Q[:, :, 1:, :] = Q[:, :, 1:, :] + inter

        gradA = Q.transpose(1, 2)[:, :L]
        gradX = grad_output.transpose(1, 2)[:, :L]
        return gradA, gradX


pscan = PScan.apply
