"""Matrix-only Muon with bounded same-shape batching and AdamW RMS scaling.

The update follows torch.optim.Muon and Fizgig's BatchedMuon (11c43efe).
Only 2D adapter parameters are supported; there is no implicit fallback optimizer.
"""

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch


def _validate_options(group: dict[str, Any]) -> None:
    for name in ("lr", "weight_decay", "eps"):
        value = group[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Muon {name} must be a finite number")
        if value < 0 or (name == "eps" and value == 0):
            raise ValueError(
                f"Muon {name} must be {'positive' if name == 'eps' else 'nonnegative'}"
            )
    momentum = group["momentum"]
    if (
        isinstance(momentum, bool)
        or not isinstance(momentum, (int, float))
        or not 0 <= momentum < 1
    ):
        raise ValueError("Muon momentum must be in [0, 1)")
    if not isinstance(group["nesterov"], bool):
        raise ValueError("Muon nesterov must be a boolean")
    if type(group["ns_steps"]) is not int or not 1 <= group["ns_steps"] < 100:
        raise ValueError("Muon ns_steps must be an integer in [1, 99]")
    coefficients = group["ns_coefficients"]
    if (
        not isinstance(coefficients, (tuple, list))
        or len(coefficients) != 3
        or any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in coefficients
        )
    ):
        raise ValueError("Muon ns_coefficients must contain three finite numbers")
    if group["adjust_lr_fn"] not in ("original", "match_rms_adamw"):
        raise ValueError("Muon adjust_lr_fn must be 'original' or 'match_rms_adamw'")
    if type(group["max_batch_bytes"]) is not int or group["max_batch_bytes"] < 1:
        raise ValueError("Muon max_batch_bytes must be a positive integer")


def _orthogonalize(
    matrices: torch.Tensor, coefficients: Sequence[float], steps: int, eps: float
) -> torch.Tensor:
    # Normalize each matrix separately; a shared batch norm changes the algorithm.
    x = matrices.to(torch.bfloat16)
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.transpose(-2, -1)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    a, b, c = coefficients
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, polynomial, x, beta=a)
    return x.transpose(-2, -1) if tall else x


class Muon(torch.optim.Optimizer):
    """Muon for 2D matrices, preserving incoming optimizer groups and options.

    ``max_batch_bytes`` limits the estimated temporary work per shape batch.
    One matrix is the minimum work unit and can exceed this budget. Momentum
    state itself is one tensor per parameter and is not part of that budget.
    Loss weighting changes gradient contributions; it is not a per-image LR.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: Sequence[float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str = "match_rms_adamw",
        max_batch_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
            max_batch_bytes=max_batch_bytes,
        )
        _validate_options(defaults)
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        group = dict(param_group)
        parameters = group["params"]
        group["params"] = (
            [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
        )
        _validate_options({**self.defaults, **group})
        identifiers = [id(p) for p in group["params"]]
        existing = {id(p) for g in self.param_groups for p in g["params"]}
        if len(set(identifiers)) != len(identifiers) or existing.intersection(
            identifiers
        ):
            raise ValueError(
                "Muon parameters must occur exactly once across optimizer groups"
            )
        for parameter in group["params"]:
            if (
                not isinstance(parameter, torch.Tensor)
                or parameter.ndim != 2
                or parameter.numel() == 0
            ):
                shape = (
                    tuple(parameter.shape)
                    if isinstance(parameter, torch.Tensor)
                    else None
                )
                raise ValueError(
                    "Muon requires nonempty 2D adapter parameters "
                    f"(got shape {shape}). Choose another optimizer or "
                    "target only dense matrix adapter factors."
                )
            if parameter.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError(
                    "Muon supports float32, float16, or bfloat16 parameters only"
                )
            if parameter.layout != torch.strided:
                raise ValueError("Muon requires dense strided parameters")
        super().add_param_group(group)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["muon_state_version"] = 1
        state["muon_param_shapes"] = [
            [list(p.shape) for p in g["params"]] for g in self.param_groups
        ]
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Check before Optimizer.load_state_dict mutates live groups or state.
        if not isinstance(state_dict, dict):
            raise ValueError("Muon checkpoint must be an optimizer state dictionary")
        shapes = [[list(p.shape) for p in g["params"]] for g in self.param_groups]
        if (
            state_dict.get("muon_state_version") != 1
            or state_dict.get("muon_param_shapes") != shapes
        ):
            raise ValueError(
                "Muon requires a compatible Muon checkpoint with matching parameter "
                "shapes/groups; start a new run when changing optimizer or adapter "
                "layout"
            )
        groups = state_dict.get("param_groups", [])
        if not isinstance(groups, list) or len(groups) != len(self.param_groups):
            raise ValueError("Muon checkpoint parameter groups do not match")
        saved_state = state_dict.get("state")
        if not isinstance(saved_state, dict):
            raise ValueError("Muon checkpoint has invalid optimizer state")
        expected_ids = set()
        for saved_group, live_group in zip(groups, self.param_groups):
            if not isinstance(saved_group, dict):
                raise ValueError("Muon checkpoint parameter group must be a dictionary")
            try:
                _validate_options(saved_group)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Muon checkpoint has invalid options: {error}"
                ) from error
            ids = saved_group.get("params", [])
            if not isinstance(ids, list) or len(ids) != len(live_group["params"]):
                raise ValueError("Muon checkpoint group size does not match")
            for identifier, parameter in zip(ids, live_group["params"]):
                if type(identifier) is not int or identifier in expected_ids:
                    raise ValueError(
                        "Muon checkpoint contains invalid parameter identifiers"
                    )
                expected_ids.add(identifier)
                entry = saved_state.get(identifier, {})
                if not isinstance(entry, dict) or set(entry) - {"momentum_buffer"}:
                    raise ValueError(
                        "Muon checkpoint contains incompatible parameter state"
                    )
                if entry:
                    buffer = entry["momentum_buffer"]
                    if (
                        not isinstance(buffer, torch.Tensor)
                        or buffer.shape != parameter.shape
                        or buffer.layout != torch.strided
                        or buffer.dtype
                        not in (torch.float16, torch.bfloat16, torch.float32)
                        or not torch.isfinite(buffer).all()
                    ):
                        raise ValueError(
                            "Muon checkpoint contains invalid momentum state"
                        )
        if set(saved_state) - expected_ids:
            raise ValueError("Muon checkpoint contains unexpected parameter state")
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(
        self, closure: Callable[[], torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Reject unsupported gradients before updating any group.
        for group in self.param_groups:
            _validate_options(group)
            for parameter in group["params"]:
                if (
                    parameter.grad is not None
                    and parameter.grad.layout != torch.strided
                ):
                    raise RuntimeError("Muon does not support sparse gradients")
        for group in self.param_groups:
            by_shape = defaultdict(list)
            for parameter in group["params"]:
                if parameter.grad is not None:
                    by_shape[
                        (tuple(parameter.shape), parameter.device, parameter.dtype)
                    ].append(parameter)
            for (shape, _device, _dtype), parameters in by_shape.items():
                # Include update stack/conversion and NS polynomial temporaries.
                # Chunking prevents hundreds of full LoKr matrices being stacked.
                m, n = shape
                work_bytes = (
                    m * n * (3 * parameters[0].element_size() + 6) + 8 * min(m, n) ** 2
                )
                chunk_size = max(1, group["max_batch_bytes"] // work_bytes)
                for start in range(0, len(parameters), chunk_size):
                    chunk = parameters[start : start + chunk_size]
                    buffers = []
                    gradients = [p.grad for p in chunk]
                    for parameter, gradient in zip(chunk, gradients):
                        state = self.state[parameter]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(gradient)
                        buffers.append(state["momentum_buffer"])
                    torch._foreach_lerp_(buffers, gradients, 1 - group["momentum"])
                    updates = (
                        [
                            g.lerp(b, group["momentum"])
                            for g, b in zip(gradients, buffers)
                        ]
                        if group["nesterov"]
                        else buffers
                    )
                    orthogonal = _orthogonalize(
                        torch.stack(updates),
                        group["ns_coefficients"],
                        group["ns_steps"],
                        group["eps"],
                    )
                    lr = group["lr"]
                    scale = (
                        0.2 * math.sqrt(max(shape))
                        if group["adjust_lr_fn"] == "match_rms_adamw"
                        else math.sqrt(max(1, m / n))
                    )
                    if group["weight_decay"]:
                        torch._foreach_mul_(chunk, 1 - lr * group["weight_decay"])
                    torch._foreach_add_(
                        chunk,
                        list(orthogonal.to(chunk[0].dtype).unbind(0)),
                        alpha=-lr * scale,
                    )
        return loss
