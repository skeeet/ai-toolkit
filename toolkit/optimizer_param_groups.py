from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import DistributedType
from torch import nn

if TYPE_CHECKING:
    from accelerate import Accelerator
    from torch.optim import Optimizer

# Used only when the model has no explicit block-name declaration. In particular,
# numeric children such as attn.to_out.0 and ff.net.0 are not transformer blocks.
_BLOCK_COLLECTIONS = (
    "transformer_blocks",
    "single_transformer_blocks",
    "joint_transformer_blocks",
    "blocks",
    "layers",
    "single_blocks",
    "double_blocks",
    "joint_blocks",
    "single_stream_blocks",
    "double_stream_blocks",
    "noise_refiner",
    "context_refiner",
)


def group_network_params_by_transformer_block(
    param_groups: list[dict[str, Any]],
    denoiser: nn.Module,
    denoiser_loras: Iterable[nn.Module],
    block_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Split existing adapter parameter groups without changing their options.

    Resolve blocks from original module paths, not serialized LoRA names: those
    names may replace dots with underscores or $$ and lose path boundaries.
    Text-encoder adapters and non-block parameters retain their original grouping.
    """
    collections = _BLOCK_COLLECTIONS if block_names is None else block_names
    if not collections:
        raise ValueError(
            "transformer_block optimizer grouping found no block collections"
        )
    pattern = re.compile(
        r"(?:^|\.)(?:"
        + "|".join(re.escape(name) for name in collections)
        + r")\.\d+(?=\.|$)"
    )
    module_blocks = {}
    for path, module in denoiser.named_modules():
        match = pattern.search(path)
        if match is not None:
            # Include the full prefix so separate stacks and nested U-Net
            # transformers with the same block index remain separate groups.
            module_blocks[id(module)] = path[: match.end()]

    parameter_blocks = {}
    for lora in denoiser_loras:
        original_ref = getattr(lora, "orig_module_ref", None)
        if original_ref is not None:
            original = original_ref()
        else:
            original = lora.org_module[0]
        block = module_blocks.get(id(original))
        if block is not None:
            for parameter in lora.parameters():
                parameter_id = id(parameter)
                if (
                    parameter_id in parameter_blocks
                    and parameter_blocks[parameter_id] != block
                ):
                    raise ValueError(
                        "An adapter parameter is shared across transformer blocks"
                    )
                parameter_blocks[parameter_id] = block

    result = []
    matched = False
    for group in param_groups:
        partitions = {}
        for parameter in group["params"]:
            block = parameter_blocks.get(id(parameter))
            matched = matched or block is not None
            partitions.setdefault(block, []).append(parameter)
        for parameters in partitions.values():
            result.append(
                {
                    **group,
                    "params": parameters,
                    "optimizer_grouping": "transformer_block",
                }
            )

    if not matched:
        raise ValueError(
            "transformer_block optimizer grouping found no trainable transformer-block "
            "adapter parameters. Check the model's get_transformer_block_names() "
            "or use network.optimizer_grouping: default."
        )
    return result


def clip_optimizer_groups(
    accelerator: Accelerator,
    param_groups: list[dict[str, Any]],
    max_grad_norm: float,
    optimizer: Optimizer,
) -> None:
    """Preserve per-group clipping while unscaling ordinary AMP updates once."""
    raw_optimizer = optimizer
    while isinstance(raw_optimizer, AcceleratedOptimizer):
        raw_optimizer = raw_optimizer.optimizer
    if getattr(raw_optimizer, "independent_groups", False) and getattr(
        raw_optimizer, "is_stochastic_rounding_accumulation", False
    ):
        # BF16 Prodigy8bit stores backward results outside .grad until its
        # pre-step hook. Materialize them before norm clipping or AMP unscale.
        raw_optimizer.step_hook()
    ordinary = (
        DistributedType.NO,
        DistributedType.MULTI_CPU,
        DistributedType.MULTI_GPU,
    )
    if (
        getattr(accelerator, "scaler", None) is not None
        and accelerator.distributed_type in ordinary
    ):
        # Accelerator.clip_grad_norm_ unscales all groups on every call. AMP
        # permits one unscale per update, so clip these already-unscaled groups.
        accelerator.unscale_gradients(optimizer)
        clip = torch.nn.utils.clip_grad_norm_
    else:
        # FSDP/DeepSpeed have specialized norm handling. Keep their public API,
        # as well as the existing path when no scaler is active.
        clip = accelerator.clip_grad_norm_
    for group in param_groups:
        clip(group["params"], max_grad_norm)
