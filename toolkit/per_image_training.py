"""Trainer bridge for per-image loss weighting; model imports stay in the caller."""

from __future__ import annotations

import copy
import json
import math
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from toolkit.per_image_adaptive import PerImageAdaptive

if TYPE_CHECKING:
    import torch

    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO


def validate_process(process: BaseSDTrainProcess) -> None:
    train, model, network = (
        process.train_config,
        process.model_config,
        process.network_config,
    )
    if not getattr(process, "supports_per_image_adaptation", False):
        raise ValueError(
            "Per-image adaptation requires the SDTrainer loss and optimizer hooks"
        )
    if (
        not train.train_unet
        or network is None
        or network.type.lower()
        not in (
            "lora",
            "lokr",
            "locon",
            "lycoris",
            "lorm",
            "dora",
        )
    ):
        raise ValueError(
            "Per-image adaptation requires a denoiser adapter training job"
        )
    # ModelConfig strips UI preset tags, so use the original choice for the
    # existing MOSS Base restriction.
    raw_model = getattr(process, "raw_process_config", {}).get("model", {})
    raw_arch = str(raw_model.get("arch", model.arch)).lower()
    arch, _, preset = raw_arch.partition(":")
    unsupported_moss = arch == "moss_image" and (
        preset not in ("", "base") or "sft" in str(model.name_or_path).lower()
    )
    if unsupported_moss:
        raise ValueError(
            "Per-image adaptation supports MOSS Base text-to-image training only"
        )
    if (
        model.model_kwargs.get("edit")
        or getattr(model, "in_context", False)
        or re.search(r"(^|[_:-])edit($|[_:-])", raw_arch)
        or any(
            getattr(ds, "control_path", None)
            or getattr(ds, "control_from_same_folder", False)
            for ds in process.dataset_configs
        )
    ):
        raise ValueError(
            "Per-image adaptation requires text-to-image training "
            "without edit or control inputs"
        )
    if getattr(process.accelerator, "num_processes", 1) != 1 or getattr(
        process.accelerator.state, "fsdp_plugin", None
    ):
        raise ValueError(
            "Per-image adaptation currently requires single-process training "
            "without FSDP"
        )
    if (
        train.loss_type != "mse"
        or train.loss_target != "noise"
        or train.train_turbo
        or train.do_guidance_loss
    ):
        raise ValueError(
            "Per-image adaptation requires the standard flow velocity MSE objective"
        )
    # Configuring a refiner changes timestep preparation even when it is frozen.
    if model.refiner_name_or_path is not None:
        raise ValueError(
            "Per-image adaptation requires model.refiner_name_or_path to be unset"
        )
    unsupported = (
        "train_text_encoder",
        "do_prior_divergence",
        "correct_pred_norm",
        "single_item_batching",
        "short_and_long_captions",
        "do_paramiter_swapping",
        "merge_network_on_save",
        "diff_output_preservation",
        "blank_prompt_preservation",
        "inverted_mask_prior",
        "do_signal_amplification",
        "do_signal_correction_noise",
        "do_batch_noise_correction",
        "do_norm_loss",
        "match_noise_norm",
        "t0_loss_target",
        "do_fft_loss",
        "target_norm_std",
        "do_differential_guidance",
    )
    if any(getattr(train, key, False) for key in unsupported):
        raise ValueError(
            "Per-image adaptation does not support altered targets, auxiliary "
            "preservation, single-item batching, paired captions or parameter swapping"
        )
    if (
        getattr(train, "pred_scaler", 1.0) != 1.0
        or getattr(train, "target_noise_multiplier", 1.0) != 1.0
        or getattr(train, "diffusion_feature_extractor_path", None)
    ):
        raise ValueError(
            "Per-image adaptation requires unmodified predictions and targets "
            "without feature loss"
        )
    if (
        process.adapter_config is not None
        or process.embed_config is not None
        or process.decorator_config is not None
        or getattr(process, "guidance_config", None) is not None
    ):
        raise ValueError(
            "Per-image adaptation requires denoiser adapter training "
            "without an auxiliary adapter"
        )
    if any(
        getattr(ds, "num_frames", 1) != 1
        or getattr(ds, "auto_frame_count", False)
        or getattr(ds, "prior_reg", False)
        or getattr(ds, "do_i2v", False)
        or getattr(ds, "do_audio", False)
        for ds in process.dataset_configs
    ):
        raise ValueError(
            "Per-image adaptation requires ordinary still-image datasets "
            "without prior_reg targets"
        )
    if any(getattr(ds, "unconditional_path", None) for ds in process.dataset_configs):
        raise ValueError(
            "Per-image adaptation does not support "
            "unconditional image guidance datasets"
        )
    optimizer = train.optimizer.lower()
    if optimizer == "automagic2":
        raise ValueError(
            "Per-image adaptation requires optimizer updates in step(); "
            "AutoMagic2 always updates during backward"
        )
    fused_default = optimizer in ("automagic3", "automagic4", "automagicexperiment")
    if optimizer in (
        "automagic3",
        "automagic4",
        "automagicexperiment",
        "adamconvrot",
    ) and train.optimizer_params.get("fused", fused_default):
        raise ValueError(
            "Per-image adaptation requires optimizer updates in step(); "
            "set optimizer_params.fused: false"
        )


def _dataset_signature(config: DatasetConfig) -> str:
    # Include the training role, captions, masks and preprocessing configuration.
    # Dataset membership is checked separately. This does not hash image pixels.
    return json.dumps(vars(config), sort_keys=True, default=str, allow_nan=False)


def _effective_resume_context(context: Any) -> Any:
    if not isinstance(context, dict):
        return context
    network, objective = context.get("network"), context.get("objective")
    if not isinstance(network, dict) or not isinstance(objective, dict):
        return context
    # Standalone checkpoints predate these optional settings. Compare their
    # effective defaults without rewriting the serialized checkpoint context.
    network = {"optimizer_grouping": "default", **network}
    objective = objective.copy()
    for key, default in (("timestep_shift", 0.0), ("timestep_scale", 1.0)):
        if objective.get(key) is None:
            objective[key] = default
    return {**context, "network": network, "objective": objective}


class PerImageTraining:
    VERSION = 1

    def __init__(self, process: BaseSDTrainProcess) -> None:
        validate_process(process)
        self.process = process
        self.watch = PerImageAdaptive(process.train_config.per_image_adaptive)
        self.pending = []
        self.gradients = []
        self.invalid = False
        self.active_epoch = int(process.epoch_num)
        self.configs = {}
        self.context = None
        self.raw_sum = self.weighted_sum = 0.0
        self.count = 0

    @staticmethod
    def validate_model(model: Any) -> None:
        if getattr(model, "is_edit", False):
            raise ValueError("Per-image adaptation requires a text-to-image model")
        if (
            not getattr(model, "is_flow_matching", False)
            or getattr(model, "prediction_type", None) == "v_prediction"
            or getattr(model, "x0_pred", False)
            or getattr(model, "dopsd_self_ref", False)
            or callable(getattr(model, "get_additional_loss", None))
        ):
            raise ValueError(
                "Per-image adaptation requires a model using standard flow velocity MSE"
            )

    @staticmethod
    def validate_batch(
        prediction: torch.Tensor,
        noise: torch.Tensor,
        batch: DataLoaderBatchDTO,
        timesteps: torch.Tensor,
    ) -> None:
        if (
            prediction.ndim != 4
            or prediction.shape != noise.shape
            or prediction.shape != batch.latents.shape
            or prediction.shape[0] != len(batch.file_items)
        ):
            raise ValueError(
                "Per-image adaptation requires matching still-image prediction, "
                "noise and clean latent tensors shaped [B, C, H, W]"
            )
        if timesteps.shape != (len(batch.file_items),):
            raise ValueError(
                "Per-image adaptation requires one loss and timestep per image"
            )

    def validate_target(
        self, target: torch.Tensor, noise: torch.Tensor, batch: DataLoaderBatchDTO
    ) -> None:
        import torch

        if target.shape != noise.shape:
            raise ValueError(
                "Per-image adaptation requires unmodified flow velocity targets"
            )
        if all(item.is_reg for item in batch.file_items):
            return
        with torch.no_grad():
            expected = (noise - batch.latents).float()
            candidate = target.float()
            if torch.allclose(
                candidate, expected, rtol=1e-4, atol=1e-5
            ) or torch.allclose(candidate, -expected, rtol=1e-4, atol=1e-5):
                return
            if (
                not torch.isfinite(candidate).all()
                or not torch.isfinite(expected).all()
            ):
                self.invalid = True
                return
        raise ValueError(
            "Per-image adaptation requires unmodified flow velocity targets "
            "(noise minus clean latents, or the reverse convention)"
        )

    def bind(self, datasets: Iterable[AiToolkitDataset]) -> None:
        signatures = {}
        keys = set()
        for dataset in datasets:
            for item in dataset.file_list:
                if item.is_reg:
                    continue
                signature = _dataset_signature(item.dataset_config)
                if signature not in signatures:
                    signatures[signature] = len(signatures)
                self.configs[signature] = signatures[signature]
                keys.add(self.key(item))
        self.watch.bind_dataset(sorted(keys))
        p = self.process
        self.context = {
            "datasets": list(signatures),
            "model": {
                "arch": p.model_config.arch,
                "name_or_path": p.model_config.name_or_path,
                "model_kwargs": copy.deepcopy(p.model_config.model_kwargs),
            },
            "network": copy.deepcopy(vars(p.network_config)),
            "objective": {
                name: copy.deepcopy(getattr(p.train_config, name, None))
                for name in (
                    "loss_type",
                    "loss_target",
                    "timestep_type",
                    "timestep_shift",
                    "timestep_scale",
                    "linear_timesteps",
                    "linear_timesteps2",
                    "snr_gamma",
                    "min_snr_gamma",
                    "learnable_snr_gos",
                )
            },
        }
        # Match the JSON representation stored in safetensors metadata.
        self.context = json.loads(
            json.dumps(self.context, sort_keys=True, default=str, allow_nan=False)
        )
        state = getattr(p, "_per_image_resume_state", None)
        if state is not None:
            if state.get("version") != self.VERSION or _effective_resume_context(
                state.get("context")
            ) != _effective_resume_context(self.context):
                raise ValueError(
                    "Per-image checkpoint model, network, objective or dataset "
                    "configuration changed; start a new run"
                )
            epoch = state.get("active_epoch")
            if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
                raise ValueError("Invalid per-image checkpoint epoch")
            snapshot = state["watcher"]
            expected_epoch = snapshot.get("pending_epoch")
            if expected_epoch is None:
                expected_epoch = snapshot.get("last_epoch", -2) + 1
            if epoch != expected_epoch:
                raise ValueError(
                    "Per-image checkpoint epoch disagrees with watcher state"
                )
            if p.epoch_num < epoch:
                raise ValueError(
                    "Per-image trainer epoch is behind the saved watcher epoch. "
                    "Remove the train.start_step override "
                    "to resume using checkpoint metadata."
                )
            self.watch.load_state_dict(snapshot)
            self.active_epoch = epoch
            p._per_image_resume_state = None

    def key(self, item: FileItemDTO) -> str:
        signature = _dataset_signature(item.dataset_config)
        if signature not in self.configs:
            raise ValueError("Unknown per-image dataset role")
        root = item.dataset_config.dataset_path or item.dataset_config.folder_path
        root = os.path.abspath(root)
        if os.path.splitext(root)[1].lower() == ".json":
            root = os.path.dirname(root)
        relative = os.path.relpath(os.path.abspath(item.path), root).replace(
            os.sep, "/"
        )
        return f"{self.configs[signature]}:{relative}"

    def restore_gradients(self) -> None:
        for parameter, gradient in self.gradients:
            parameter.grad = gradient
        self.gradients = []

    def prepare_gradients(self) -> None:
        # Toolkit optimizers can accumulate low-precision gradients outside
        # .grad. Expose them before the trainer's finite check and clipping.
        for group in self.process.optimizer.param_groups:
            for parameter in group["params"]:
                if hasattr(parameter, "_accum_grad"):
                    parameter.grad = parameter._accum_grad
                    del parameter._accum_grad

    def apply(
        self,
        loss: torch.Tensor,
        batch: DataLoaderBatchDTO,
        timesteps: torch.Tensor,
        *,
        mask_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch

        items = batch.file_items
        if loss.shape != (len(items),) or timesteps.shape != (len(items),):
            raise ValueError(
                "Per-image adaptation requires one loss and timestep per image"
            )
        eligible = [not item.is_reg for item in items]
        weights = [
            self.watch.multiplier(self.key(item)) if keep else 1.0
            for item, keep in zip(items, eligible)
        ]
        weighted = loss * loss.new_tensor(weights)
        epoch = int(getattr(batch, "_per_image_epoch", self.process.epoch_num))
        raw = loss.detach().float().cpu().tolist()
        fractions = (timesteps.detach().float() / 1000.0).cpu().tolist()
        if not all(torch.isfinite(loss).tolist()) or not all(
            0.0 <= t <= 1.0 for t in fractions
        ):
            self.invalid = True
            return weighted
        masses = mask_mass.cpu().tolist() if mask_mass is not None else [1.0] * len(raw)
        for item, keep, value, t, mass in zip(items, eligible, raw, fractions, masses):
            if keep and math.isfinite(mass) and mass > 0:
                self.pending.append((self.key(item), epoch, t, value / mass))
        self.raw_sum += sum(raw)
        self.weighted_sum += sum(value * weight for value, weight in zip(raw, weights))
        self.count += len(raw)
        return weighted

    def discard(self) -> None:
        self.pending.clear()
        self.gradients.clear()
        for group in self.process.optimizer.param_groups:
            for parameter in group["params"]:
                if hasattr(parameter, "_accum_grad"):
                    del parameter._accum_grad
        self.invalid = False
        self.raw_sum = self.weighted_sum = 0.0
        self.count = 0

    def finish(self, *, stepped: bool, skipped: bool = False) -> None:
        if not stepped:
            self.gradients = [
                (p, p.grad)
                for group in self.process.optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            return
        if skipped or self.invalid:
            self.discard()
            return
        boundary = False
        for key, epoch, t, value in self.pending:
            while self.active_epoch < epoch:
                self.watch.end_epoch(self.active_epoch)
                self.active_epoch += 1
                boundary = True
            self.watch.record(key, epoch, t, value)
        # An update may end with only regularization samples in a new epoch.
        while self.active_epoch < self.process.epoch_num:
            self.watch.end_epoch(self.active_epoch)
            self.active_epoch += 1
            boundary = True
        if self.count:
            self.process.additional_logs.update(
                {
                    "per_image/raw_loss": self.raw_sum / self.count,
                    "per_image/weighted_loss": self.weighted_sum / self.count,
                }
            )
        self.discard()
        if boundary:
            self.write_report()

    def state_dict(self) -> dict[str, Any]:
        # Pending observations have no successful optimizer update and are not
        # authoritative state. Upstream does not checkpoint pending gradients.
        return {
            "version": self.VERSION,
            "context": copy.deepcopy(self.context),
            "active_epoch": self.active_epoch,
            "watcher": self.watch.state_dict(),
        }

    def write_report(self) -> None:
        path = Path(self.process.save_root) / "per_image_adaptive.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(self.watch.report(), indent=2, allow_nan=False) + "\n"
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
