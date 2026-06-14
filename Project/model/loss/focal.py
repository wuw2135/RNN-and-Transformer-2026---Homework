"""
https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
"""

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange


class FocalLoss(nn.Module):
    def __init__(
        self, alpha=[1.0 / 14] * 14, gamma=4.0, ignore_index=255, reduction="mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index    
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.to(torch.float32))
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor([alpha], dtype=torch.float32))
        else:
            raise TypeError("alpha should be None | float | list/tuple | 1D tensor")

        self.set_alpha(alpha)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        N, C, H, W = input.shape
        assert target.shape == (
            N,
            H,
            W,
        ), f"target shape must be (N,H,W), got {target.shape}"
        assert target.dtype in (
            torch.int64,
            torch.long,
        ), "target must be class indices (LongTensor)."

        # log-softmax & flatten to (N*H*W, C)
        logpt = F.log_softmax(input, dim=1).permute(0, 2, 3, 1).reshape(-1, C)
        tgt = target.view(-1)

        # optional ignore_index mask
        if self.ignore_index is not None:
            mask = tgt != self.ignore_index
            if mask.sum() == 0:
                return input.new_zeros(())
            logpt = logpt[mask]
            tgt = tgt[mask]

        idx = torch.arange(logpt.size(0), device=input.device)
        logpt_t = logpt[idx, tgt]  # (M,)
        pt = logpt_t.exp()  # (M,)

        if self.alpha is not None:
            alpha = self.alpha.to(input.device)
            if alpha.numel() == 1 and C == 2:
                # two classes
                alpha_vec = torch.tensor(
                    [alpha.item(), 1.0 - alpha.item()],
                    dtype=torch.float32,
                    device=input.device,
                )
                alpha_t = alpha_vec[tgt]
            else:
                assert (
                    alpha.numel() == C
                ), f"alpha length must be C={C}, got {alpha.numel()}"
                alpha_t = alpha[tgt]  # (M,)
        else:
            alpha_t = 1.0

        # focal loss
        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * logpt_t  # (M,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def set_alpha(self, alpha):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        alpha = alpha / (torch.mean(alpha) + 1e-5)
        self.alpha = alpha
