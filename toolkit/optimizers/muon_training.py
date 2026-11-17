"""Process checks and gradient accumulation for denoiser adapter Muon training."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


def validate_process(process: BaseSDTrainProcess) -> None:
    model, train, network = (
        process.model_config,
        process.train_config,
        process.network_config,
    )
    adapter_types = {"lora", "lokr", "locon", "lycoris", "lorm", "dora"}
    if (
        network is None
        or network.type.lower() not in adapter_types
        or not train.train_unet
        or getattr(process.sd, "is_llm", False)
    ):
        raise ValueError(
            "Muon requires denoiser adapter training with train_unet enabled"
        )
    raw_arch = process.raw_process_config.get("model", {}).get("arch") or model.arch
    arch = raw_arch.lower()
    family, _, variant = arch.partition(":")
    if family == "moss_image" and (
        variant not in ("", "base") or "sft" in str(model.name_or_path).lower()
    ):
        raise ValueError(
            "Muon requires MOSS-Image Base for MOSS-Image adapter training"
        )
    if (
        model.model_kwargs.get("edit", False)
        or getattr(process.sd, "is_edit", False)
        or re.search(r"(^|[_:-])edit($|[_:-])", arch)
        or any(
            dataset.control_path or dataset.control_from_same_folder
            for dataset in process.dataset_configs
        )
    ):
        raise ValueError(
            "Muon does not support edit training or dataset control images"
        )
    if getattr(process.accelerator, "num_processes", 1) != 1 or getattr(
        process.accelerator.state, "fsdp_plugin", None
    ):
        raise ValueError("Muon currently requires single-process training without FSDP")
    if (
        train.train_text_encoder
        or process.adapter_config is not None
        or process.embed_config is not None
        or process.decorator_config is not None
    ):
        raise ValueError(
            "Muon does not support text encoder, auxiliary adapter, "
            "embedding or decorator training"
        )
    if train.do_paramiter_swapping or train.merge_network_on_save:
        raise ValueError(
            "Muon does not support parameter swapping or merging the adapter on save"
        )

    # Check the actual optimizer groups after trainer hooks have added parameters.
    if any(
        not isinstance(group, dict) or "params" not in group for group in process.params
    ):
        raise ValueError(
            "Muon requires denoiser adapter parameter groups. "
            "Disable extra trainable parameters or choose another optimizer."
        )
    parameters = {
        id(parameter) for group in process.params for parameter in group["params"]
    }
    if not parameters:
        raise ValueError(
            "Muon found no denoiser adapter parameters. "
            "Check the selected target layers."
        )
    adapter_parameters = {
        id(parameter)
        for module in getattr(process.network, "unet_loras", ())
        for parameter in module.parameters()
    }
    if not parameters.issubset(adapter_parameters):
        raise ValueError(
            "Muon's optimizer groups contain extra trainable parameters. "
            "Disable training outside the denoiser adapter or choose another optimizer."
        )


def _parameters(process: BaseSDTrainProcess) -> Iterator[torch.Tensor]:
    optimizer = process.optimizer
    for group in optimizer.param_groups:
        yield from group["params"]


def restore_gradients(process: BaseSDTrainProcess) -> None:
    # The upstream outer hook clears gradients with set_to_none=True. Holding
    # references avoids copying full LoKr gradients between accumulation calls.
    for parameter, gradient in getattr(process, "_muon_accumulated_grads", ()):
        parameter.grad = gradient
    process._muon_accumulated_grads = []


def retain_gradients(process: BaseSDTrainProcess) -> None:
    process._muon_accumulated_grads = (
        [(p, p.grad) for p in _parameters(process) if p.grad is not None]
        if process.is_grad_accumulation_step
        else []
    )


def discard_gradients(process: BaseSDTrainProcess) -> None:
    process._muon_accumulated_grads = []
    process._muon_invalid_loss = False


def skip_invalid_update(process: BaseSDTrainProcess) -> bool:
    import torch

    invalid = getattr(process, "_muon_invalid_loss", False) or not all(
        torch.isfinite(p.grad).all().item()
        for p in _parameters(process)
        if p.grad is not None
    )
    if not invalid:
        return False
    process.optimizer.zero_grad(set_to_none=True)
    discard_gradients(process)
    adaptive = getattr(process, "per_image_training", None)
    if adaptive is not None:
        adaptive.finish(stepped=True, skipped=True)
    sampler = getattr(process, "infonoise_sampler", None)
    if sampler is not None:
        sampler.discard_pending()
    return True
