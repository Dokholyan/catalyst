# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import List, Optional
from functools import partial

import torch
from torch import nn

from catalyst.metrics.functional import wrap_metric_fn_with_activation
from catalyst.metrics.region_base_metrics import trevsky


class TrevskyLoss(nn.Module):
    """The trevsky loss."""

    def __init__(
        self,
        alpha: float,
        beta: Optional[float] = None,
        class_dim: int = 1,
        activation: str = "Sigmoid",
        mode: str = "micro",
        weights: List[float] = None,
        eps: float = 1e-7,
    ):
        """
        Args:
            alpha: false negative coefficient, bigger alpha bigger penalty for
                false negative. Must be in (0, 1)
            beta: false positive coefficient, bigger alpha bigger penalty for
                false positive. Must be in (0, 1), if None beta = (1 - alpha)
            class_dim: indicates class dimention (K) for
                ``outputs`` and ``targets`` tensors (default = 1)
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax'``
            mode: class summation strategy. Must be one of ['micro', 'macro',
                'weighted']. If mode='micro', classes are ignored, and metric
                are calculated generally. If mode='macro', metric are
                calculated separately and than are averaged over all classes.
                If mode='weighted', metric are calculated separately and than
                summed over all classes with weights.
            weights: class weights(for mode="weighted")
            eps: epsilon to avoid zero division
        """
        super().__init__()
        assert mode in ["micro", "macro", "weighted"]
        metric_fn = wrap_metric_fn_with_activation(
            metric_fn=trevsky, activation=activation
        )
        self.loss_fn = partial(
            metric_fn,
            eps=eps,
            alpha=alpha,
            beta=beta,
            class_dim=class_dim,
            threshold=None,
            mode=mode,
            weights=weights,
        )

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculates loss between ``logits`` and ``target`` tensors."""
        trevsky = self.loss_fn(outputs, targets)  # [bs; num_classes]
        return 1 - trevsky


__all__ = ["TrevskyLoss"]