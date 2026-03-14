"""Loss functions for multi-label chest X-ray classification."""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.register_buffer(
            "alpha", alpha if isinstance(alpha, torch.Tensor) else None
        )
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-bce)
        if self.alpha is not None:
            bce = self.alpha * bce
        loss = (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class AsymmetricLossOptimized(nn.Module):
    """Optimized ASL: minimizes memory allocation and favors inplace operations."""

    def __init__(
        self,
        gamma_neg=5,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
    ):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.targets = (
            self.anti_targets
        ) = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        self.targets = y
        self.anti_targets = 1 - y
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(
            self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps))
        )
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets
                + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w
        return -self.loss.sum()


class APLLoss(nn.Module):
    """Taylor-expansion-based asymmetric loss."""

    def __init__(
        self,
        gamma_neg=5,
        gamma_pos=0,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(APLLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.epsilon_pos = 1.0
        self.epsilon_neg = 0.0
        self.epsilon_pos_pow = -2.5

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * (
            torch.log(xs_pos.clamp(min=self.eps))
            + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps))
            + self.epsilon_pos_pow
            * 0.5
            * torch.pow(1 - xs_pos.clamp(min=self.eps), 2)
        )
        los_neg = (1 - y) * (
            torch.log(xs_neg.clamp(min=self.eps))
            + self.epsilon_neg * (xs_neg.clamp(min=self.eps))
        )
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        return -loss.sum()
