from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ProjectionSummary:
    layers: int
    total_rank: int
    dimensions: Dict[str, int]
    ranks: Dict[str, int]


class IterativeSubspaceProjector:
    """Memory-bounded iSVD representation and GPM gradient projection.

    Each basis contains the significant input-feature directions of a layer.
    A new activation block is first residualized against the stored basis, then
    only enough left-singular vectors to reach ``energy_threshold`` are added.
    This is Eq. (6), Eq. (7), and Algorithm 1 from One-for-More expressed as a
    reusable, device-agnostic pyCLAD component.
    """

    def __init__(
        self,
        energy_threshold: float = 0.999,
        max_columns: Optional[int] = 50_000,
        random_seed: int = 0,
    ):
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("energy_threshold must be in (0, 1]")
        if max_columns is not None and max_columns <= 0:
            raise ValueError("max_columns must be positive or None")
        self.energy_threshold = float(energy_threshold)
        self.max_columns = max_columns
        self.random_seed = int(random_seed)
        self._bases: Dict[str, torch.Tensor] = {}
        self._updates = 0

    @property
    def bases(self) -> Dict[str, torch.Tensor]:
        return {name: basis.clone() for name, basis in self._bases.items()}

    def has_basis(self, name: str) -> bool:
        return name in self._bases and self._bases[name].shape[1] > 0

    def _sample_columns(self, matrix: torch.Tensor) -> torch.Tensor:
        if self.max_columns is None or matrix.shape[1] <= self.max_columns:
            return matrix
        generator = torch.Generator(device="cpu").manual_seed(self.random_seed + self._updates)
        indices = torch.randperm(matrix.shape[1], generator=generator)[: self.max_columns]
        return matrix[:, indices]

    @staticmethod
    def _rank_for_energy(singular_values: torch.Tensor, target_energy: torch.Tensor) -> int:
        if singular_values.numel() == 0 or float(target_energy) <= 0.0:
            return 0
        cumulative = torch.cumsum(singular_values.square(), dim=0)
        rank = int(torch.searchsorted(cumulative, target_energy, right=False).item()) + 1
        return min(rank, singular_values.numel())

    @torch.no_grad()
    def update(self, name: str, activation_matrix: torch.Tensor) -> None:
        matrix = activation_matrix.detach().to(device="cpu", dtype=torch.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Activation matrix for '{name}' must be 2D, got {matrix.shape}")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return
        if not torch.isfinite(matrix).all():
            raise ValueError(f"Activation matrix for '{name}' contains non-finite values")

        matrix = self._sample_columns(matrix)
        total_energy = matrix.square().sum()
        if float(total_energy) <= torch.finfo(matrix.dtype).eps:
            return

        previous = self._bases.get(name)
        if previous is None:
            left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
            target = self.energy_threshold * total_energy
            rank = self._rank_for_energy(singular_values, target)
            self._bases[name] = left[:, :rank].contiguous()
            self._updates += 1
            return

        if previous.shape[0] != matrix.shape[0]:
            raise ValueError(
                f"Feature dimension for '{name}' changed from {previous.shape[0]} to {matrix.shape[0]}"
            )

        residual = matrix - previous @ (previous.transpose(0, 1) @ matrix)
        residual_energy = residual.square().sum()
        captured_energy = (total_energy - residual_energy).clamp_min(0.0)
        target_energy = self.energy_threshold * total_energy
        additional_target = target_energy - captured_energy
        if float(additional_target) <= torch.finfo(matrix.dtype).eps:
            self._updates += 1
            return

        left, singular_values, _ = torch.linalg.svd(residual, full_matrices=False)
        rank = self._rank_for_energy(singular_values, additional_target)
        if rank == 0:
            self._updates += 1
            return

        augmented = torch.cat([previous, left[:, :rank]], dim=1)
        # Numerical re-orthogonalization keeps repeated projections stable.
        orthogonal, _ = torch.linalg.qr(augmented, mode="reduced")
        self._bases[name] = orthogonal.contiguous()
        self._updates += 1

    @torch.no_grad()
    def project_gradient(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        basis = self._bases.get(name)
        if basis is None:
            return gradient
        flat = gradient.reshape(gradient.shape[0], -1)
        if flat.shape[1] != basis.shape[0]:
            raise ValueError(
                f"Gradient feature dimension for '{name}' is {flat.shape[1]}, expected {basis.shape[0]}"
            )
        device_basis = basis.to(device=gradient.device, dtype=gradient.dtype)
        projected = flat - (flat @ device_basis) @ device_basis.transpose(0, 1)
        return projected.reshape_as(gradient)

    @torch.no_grad()
    def apply_to_model(self, model: nn.Module, freeze_unprojected: bool = True) -> None:
        handled_parameters: set[int] = set()

        for name, module in model.named_modules():
            parameter = None
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameter = module.weight
            elif isinstance(module, nn.MultiheadAttention):
                parameter = module.in_proj_weight

            if parameter is None or parameter.grad is None:
                continue
            handled_parameters.add(id(parameter))
            if self.has_basis(name):
                parameter.grad.copy_(self.project_gradient(name, parameter.grad))
            elif freeze_unprojected:
                parameter.grad.zero_()

        if freeze_unprojected:
            for parameter in model.parameters():
                if parameter.grad is not None and id(parameter) not in handled_parameters:
                    parameter.grad.zero_()

    def summary(self) -> ProjectionSummary:
        dimensions = {name: int(basis.shape[0]) for name, basis in self._bases.items()}
        ranks = {name: int(basis.shape[1]) for name, basis in self._bases.items()}
        return ProjectionSummary(
            layers=len(self._bases),
            total_rank=sum(ranks.values()),
            dimensions=dimensions,
            ranks=ranks,
        )


def activation_matrix(module: nn.Module, inputs: Iterable[torch.Tensor]) -> Optional[torch.Tensor]:
    """Convert a protected layer input to the feature matrix used by iSVD."""

    inputs = tuple(inputs)
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return None
    value = inputs[0].detach()

    if isinstance(module, nn.Conv2d):
        unfolded = F.unfold(
            value,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        return unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1]).transpose(0, 1).cpu()

    if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
        return value.reshape(-1, value.shape[-1]).transpose(0, 1).cpu()

    return None


def protected_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
            yield name, module
