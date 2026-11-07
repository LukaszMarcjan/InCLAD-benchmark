from math import exp

import torch
import torch.nn.functional as F
from torch import nn


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float | list[float] | None = None,
        gamma: float = 2.0,
        balance_index: int = 0,
        smooth: float = 1e-5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = probabilities.shape[1]

        if probabilities.ndim > 2:
            probabilities = probabilities.view(probabilities.size(0), probabilities.size(1), -1)
            probabilities = probabilities.permute(0, 2, 1).contiguous()
            probabilities = probabilities.view(-1, probabilities.size(-1))

        targets = torch.squeeze(targets, 1).view(-1, 1).long()

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_classes, 1, device=probabilities.device, dtype=probabilities.dtype)
        elif isinstance(alpha, list):
            if len(alpha) != num_classes:
                raise ValueError(f"alpha length must equal num_classes={num_classes}")
            alpha = torch.tensor(alpha, dtype=probabilities.dtype, device=probabilities.device).view(num_classes, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha_tensor = torch.ones(num_classes, 1, device=probabilities.device, dtype=probabilities.dtype)
            alpha_tensor = alpha_tensor * (1.0 - alpha)
            alpha_tensor[self.balance_index] = alpha
            alpha = alpha_tensor
        else:
            raise TypeError("alpha must be None, float, or list[float]")

        one_hot_key = torch.zeros(targets.size(0), num_classes, device=probabilities.device, dtype=probabilities.dtype)
        one_hot_key = one_hot_key.scatter_(1, targets, 1.0)

        if self.smooth > 0:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / max(num_classes - 1, 1), 1.0 - self.smooth)

        pt = (one_hot_key * probabilities).sum(dim=1) + self.smooth
        log_pt = pt.log()
        alpha_factor = torch.squeeze(alpha[targets])
        loss = -1.0 * alpha_factor * ((1.0 - pt) ** self.gamma) * log_pt

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [exp(-((value - window_size // 2) ** 2) / float(2 * sigma**2)) for value in range(window_size)],
        dtype=torch.float32,
    )
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int = 1) -> torch.Tensor:
    one_dim_window = _gaussian(window_size, sigma=1.5).unsqueeze(1)
    two_dim_window = one_dim_window.mm(one_dim_window.t()).float().unsqueeze(0).unsqueeze(0)
    return two_dim_window.expand(channel, 1, window_size, window_size).contiguous()


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    window: torch.Tensor | None = None,
    size_average: bool = True,
    val_range: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if val_range is None:
        max_val = 255.0 if torch.max(img1) > 128 else 1.0
        min_val = -1.0 if torch.min(img1) < -0.5 else 0.0
        dynamic_range = max_val - min_val
    else:
        dynamic_range = val_range

    padding = window_size // 2
    _, channel, height, width = img1.size()

    if window is None:
        real_size = min(window_size, height, width)
        window = _create_window(real_size, channel=channel).to(img1.device, dtype=img1.dtype)

    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2

    contrast_map = 2.0 * sigma12 + c2
    variance_map = sigma1_sq + sigma2_sq + c2
    ssim_map = ((2.0 * mu1_mu2 + c1) * contrast_map) / ((mu1_sq + mu2_sq + c1) * variance_map)

    if size_average:
        return ssim_map.mean(), ssim_map
    return ssim_map.mean(dim=1).mean(dim=1).mean(dim=1), ssim_map


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, size_average: bool = True, val_range: float | None = None):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = _create_window(window_size)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        _, channel, _, _ = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device)
        else:
            window = _create_window(self.window_size, channel=channel).to(img1.device, dtype=img1.dtype)
            self.window = window.detach().cpu()
            self.channel = channel

        score, _ = _ssim(
            img1,
            img2,
            window=window,
            window_size=self.window_size,
            size_average=self.size_average,
            val_range=self.val_range,
        )
        return 1.0 - score


class DRAEMLoss(nn.Module):
    def __init__(
        self,
        reconstruction_l2_weight: float,
        reconstruction_ssim_weight: float,
        reconstruction_ssim_window: int,
        segmentation_loss_weight: float,
        focal_alpha: float | None,
        focal_gamma: float,
    ):
        super().__init__()
        self.reconstruction_l2_weight = reconstruction_l2_weight
        self.reconstruction_ssim_weight = reconstruction_ssim_weight
        self.segmentation_loss_weight = segmentation_loss_weight
        self.ssim_loss = SSIMLoss(window_size=reconstruction_ssim_window)
        self.focal_loss = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        segmentation_probabilities: torch.Tensor,
        anomaly_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l2_loss = F.mse_loss(reconstruction, target)
        ssim_loss = self.ssim_loss(reconstruction, target)
        reconstruction_loss = self.reconstruction_l2_weight * l2_loss + self.reconstruction_ssim_weight * ssim_loss
        segmentation_loss = self.focal_loss(segmentation_probabilities, anomaly_mask)
        total_loss = reconstruction_loss + self.segmentation_loss_weight * segmentation_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "segmentation_loss": segmentation_loss,
            "l2_loss": l2_loss,
            "ssim_loss": ssim_loss,
        }
