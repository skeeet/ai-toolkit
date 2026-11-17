"""Execute real trainer hooks without importing the model/plugin registry."""

import ast
import copy
import importlib.util
import json
import os
import unittest
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

from toolkit.dto import DTO
from toolkit.per_image_training import PerImageTraining, validate_process

try:
    from toolkit.samplers import infonoise
except ImportError:
    if importlib.util.find_spec("toolkit.samplers.infonoise"):
        raise
    infonoise = None

InfoNoiseSampler = infonoise.InfoNoiseSampler if infonoise is not None else None


ROOT = Path(__file__).resolve().parents[1]
SD_SOURCE = ROOT / "extensions_built_in/sd_trainer/SDTrainer.py"
BASE_SOURCE = ROOT / "jobs/process/BaseSDTrainProcess.py"
MODEL_SOURCE = ROOT / "extensions_built_in/diffusion_models"


def extract_method(path, name, namespace=None):
    tree = ast.parse(path.read_text())
    node = copy.deepcopy(
        next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
    )
    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    scope = {
        "torch": torch,
        "OrderedDict": OrderedDict,
        "DTO": DTO,
        "get_torch_dtype": lambda _: torch.float32,
        "sync_grad_transfers": lambda: None,
        "print_acc": lambda *a: None,
        "copy": copy,
        "os": os,
        **(namespace or {}),
    }
    if hasattr(infonoise, "validate_target"):
        scope["validate_infonoise_target"] = infonoise.validate_target
    if importlib.util.find_spec("toolkit.optimizer_param_groups"):
        from toolkit.optimizer_param_groups import clip_optimizer_groups

        scope["clip_optimizer_groups"] = clip_optimizer_groups
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope[name]


CALCULATE = extract_method(SD_SOURCE, "calculate_loss")
TRAIN = extract_method(SD_SOURCE, "hook_train_loop")


def dataset_config(root="/dataset/a", **extra):
    return NS(
        **{
            "dataset_path": root,
            "folder_path": root,
            "num_frames": 1,
            "auto_frame_count": False,
            "prior_reg": False,
            **extra,
        }
    )


def image(config, name="same.png", reg=False):
    return NS(dataset_config=config, path=f"{config.dataset_path}/{name}", is_reg=reg)


def process(
    mode="adaptive", items=None, *, model=None, network_type="lokr", **train_options
):
    config = {
        "per_image_adaptive": {"mode": mode},
        "loss_type": "mse",
        "loss_target": "noise",
        "train_turbo": False,
        "do_guidance_loss": False,
        "do_differential_guidance": False,
        "train_text_encoder": False,
        "train_unet": True,
        "do_prior_divergence": False,
        "correct_pred_norm": False,
        "single_item_batching": False,
        "do_paramiter_swapping": False,
        "merge_network_on_save": False,
        "diff_output_preservation": False,
        "blank_prompt_preservation": False,
        "inverted_mask_prior": False,
        "do_signal_amplification": False,
        "do_signal_correction_noise": False,
        "do_batch_noise_correction": False,
        "do_norm_loss": False,
        "optimizer": "adamw",
        "optimizer_params": {},
        "dtype": "float32",
        "match_noise_norm": False,
        "pred_scaler": 1.0,
        "target_noise_multiplier": 1.0,
        "t0_loss_target": False,
        "do_fft_loss": False,
        "linear_timesteps": False,
        "linear_timesteps2": False,
        "timestep_type": "sigmoid",
        "learnable_snr_gos": False,
        "snr_gamma": None,
        "min_snr_gamma": None,
        "target_norm_std": False,
        "max_loss_debug": False,
        "max_loss": None,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 100.0,
        "start_step": None,
        **train_options,
    }
    items = [image(dataset_config())] if items is None else items
    p = NS(
        train_config=NS(**config),
        model_config=NS(
            **{
                "arch": "moss_image",
                "name_or_path": "MOSS-Image-Base",
                "refiner_name_or_path": None,
                "model_kwargs": {},
                "low_vram": False,
                **(model or {}),
            }
        ),
        network_config=(
            None
            if network_type is None
            else NS(type=network_type, pretrained_lora_path=None)
        ),
        supports_per_image_adaptation=True,
        accelerator=NS(
            num_processes=1,
            state=NS(fsdp_plugin=None),
            is_main_process=True,
            optimizer_step_was_skipped=False,
            backward=lambda loss: loss.backward(),
            clip_grad_norm_=lambda parameters, maximum: None,
        ),
        adapter_config=None,
        embed_config=None,
        decorator_config=None,
        guidance_config=None,
        dataset_configs=[item.dataset_config for item in items],
        epoch_num=0,
        additional_logs={},
        dfe=None,
        adapter=None,
        device_torch=torch.device("cpu"),
        infonoise_sampler=None,
        _infonoise_target_checked=False,
        sd=NS(
            prediction_type="epsilon",
            is_flow_matching=True,
            is_multistage=False,
            scale_loss=lambda loss: loss,
            x0_pred=False,
        ),
        embedding=None,
        ema=None,
        steps_this_boundary=0,
        is_grad_accumulation_step=False,
        timer=lambda *a: nullcontext(),
        lr_scheduler=NS(step=Mock()),
        end_of_training_loop=Mock(),
        num_consecutive_oom=0,
    )
    p.per_image_training = PerImageTraining(p)
    p.per_image_training.bind([NS(file_list=items)])
    p.per_image_training.write_report = Mock()
    return p


def batch(items, *, epoch=0, multipliers=None, latents=None):
    n = len(items)
    return NS(
        file_items=items,
        latents=torch.zeros(n, 1, 2, 2) if latents is None else latents,
        loss_multiplier_list=[1.0] * n if multipliers is None else multipliers,
        mask_tensor=None,
        get_is_reg_list=lambda: [item.is_reg for item in items],
        _per_image_epoch=epoch,
    )


class PerImageTrainingTests(unittest.TestCase):
    def test_standard_flow_models_and_native_adapter_routes_use_actual_targets(self):
        cases = (
            ("krea2:turbo", "Krea-2-Turbo", "lora", "krea2/krea2.py"),
            ("qwen_image_2", "Qwen-Image-2.1", "locon", "qwen_image_2/qwen_image_2.py"),
            ("zimage:turbo", "Z-Image-Turbo", "lycoris", "z_image/z_image.py"),
            ("moss_image", "MOSS-Image-Base", "lokr", "moss_image/moss_image.py"),
            ("moss_image:base", "MOSS-Image-Base", "dora", "moss_image/moss_image.py"),
            ("flux2", "FLUX.2-SFT", "lorm", "flux2/flux2_model.py"),
        )
        cfg = dataset_config()
        items = [image(cfg, "a.png"), image(cfg, "b.png")]
        for arch, name, network, source in cases:
            with self.subTest(arch=arch, network=network):
                p = process(
                    items=items,
                    model={"arch": arch, "name_or_path": name},
                    network_type=network,
                )
                target_fn = extract_method(MODEL_SOURCE / source, "get_loss_target")
                p.sd.get_loss_target = lambda **kwargs: target_fn(p.sd, **kwargs)
                p.per_image_training.validate_model(p.sd)
                p.per_image_training.watch.multiplier = lambda key: (
                    0.5 if key == "0:a.png" else 1.0
                )
                latents = torch.full((2, 1, 2, 2), 3.0)
                noise = torch.full_like(latents, 5.0)
                pred = (
                    torch.tensor([4.0, 5.0])
                    .view(2, 1, 1, 1)
                    .expand_as(noise)
                    .clone()
                    .requires_grad_()
                )
                loss = CALCULATE(
                    p,
                    pred,
                    noise,
                    noise,
                    torch.tensor([250.0, 750.0]),
                    batch(items, latents=latents),
                )
                loss.backward()
                self.assertEqual(loss.item(), 5.5)
                torch.testing.assert_close(
                    pred.grad[:, 0, 0, 0], torch.tensor([0.25, 0.75])
                )
                self.assertEqual(
                    p.per_image_training.pending,
                    [
                        ("0:a.png", 0, 0.25, 4.0),
                        ("0:b.png", 0, 0.75, 9.0),
                    ],
                )

    def test_reverse_flow_velocity_uses_the_actual_f_light_target(self):
        item = image(dataset_config())
        p = process(items=[item], model={"arch": "f_light", "name_or_path": "F-Lite"})
        target_fn = extract_method(
            MODEL_SOURCE / "f_light/f_light.py", "get_loss_target"
        )
        p.sd.get_loss_target = lambda **kwargs: target_fn(p.sd, **kwargs)
        noise = torch.full((1, 1, 2, 2), 5.0)
        pred = torch.zeros_like(noise, requires_grad=True)
        b = batch([item], latents=torch.full_like(noise, 3.0))
        loss = CALCULATE(p, pred, noise, noise, torch.tensor([250.0]), b)
        loss.backward()
        self.assertEqual(loss.item(), 4.0)
        torch.testing.assert_close(pred.grad, torch.ones_like(noise))
        self.assertEqual(p.per_image_training.pending, [("0:same.png", 0, 0.25, 4.0)])

    def test_refiner_validation_depends_on_the_configured_model(self):
        for train_refiner in (True, False):
            with self.subTest(train_refiner=train_refiner):
                process(train_refiner=train_refiner)
                with self.assertRaisesRegex(ValueError, "refiner_name_or_path"):
                    process(
                        train_refiner=train_refiner,
                        model={"refiner_name_or_path": "separate-refiner"},
                    )

    def test_non_adapter_edit_and_non_image_configs_fail_before_model_load(self):
        for options in (
            {"network_type": None},
            {"network_type": "custom"},
            {"train_unet": False},
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "denoiser adapter"),
            ):
                process(**options)
        for options in (
            {"arch": "moss_image:sft", "name_or_path": "local-checkpoint"},
            {"name_or_path": "OpenMOSS-Team/MOSS-Image-SFT"},
            {"arch": "moss_image:base_edit"},
            {"arch": "qwen_image_edit_plus:2511"},
            {"arch": "krea2", "model_kwargs": {"edit": True}},
        ):
            with (
                self.subTest(model=options),
                self.assertRaisesRegex(ValueError, "text-to-image"),
            ):
                process(model=options)
        for options in (
            {"num_frames": 2},
            {"auto_frame_count": True},
            {"do_audio": True},
            {"do_i2v": True},
            {"prior_reg": True},
        ):
            with (
                self.subTest(dataset=options),
                self.assertRaisesRegex(ValueError, "still-image datasets"),
            ):
                process(items=[image(dataset_config(**options))])
        with self.assertRaisesRegex(ValueError, "text-to-image"):
            process(
                model={"arch": "qwen_image_2"},
                items=[image(dataset_config(control_path=["/refs"]))],
            )
        with self.assertRaisesRegex(ValueError, "text-to-image"):
            process(
                model={"arch": "qwen_image_2"},
                items=[image(dataset_config(control_from_same_folder=True))],
            )
        p = process()
        p.supports_per_image_adaptation = False
        with self.assertRaisesRegex(ValueError, "SDTrainer"):
            validate_process(p)

    def test_initialized_model_requires_standard_flow_prediction(self):
        p = process()
        for options in (
            {"is_flow_matching": False},
            {"prediction_type": "v_prediction"},
            {"x0_pred": True},
            {"dopsd_self_ref": True},
            {"get_additional_loss": lambda *args: torch.tensor(1.0)},
        ):
            with (
                self.subTest(model=options),
                self.assertRaisesRegex(ValueError, "flow velocity"),
            ):
                p.per_image_training.validate_model(NS(**(vars(p.sd) | options)))
        with self.assertRaisesRegex(ValueError, "text-to-image"):
            p.per_image_training.validate_model(NS(**(vars(p.sd) | {"is_edit": True})))

    def test_loss_rejects_non_image_audio_broadcast_and_non_vector_timesteps(self):
        items = [image(dataset_config(), name) for name in ("a.png", "b.png")]
        p = process(items=items)
        normal = torch.ones((2, 1, 2, 2))
        video = normal.unsqueeze(2)
        tokens = normal.flatten(2)
        timesteps = torch.tensor([250.0, 750.0])
        for pred, noise, times, message in (
            (video, video, timesteps, "matching still-image"),
            (tokens, tokens, timesteps, "matching still-image"),
            (normal[:1], normal, timesteps, "matching still-image"),
            (normal, normal, timesteps.unsqueeze(1), "one loss and timestep"),
            (
                DTO(normal, audio=normal, audio_target=normal),
                normal,
                timesteps,
                "joint audio",
            ),
        ):
            with self.subTest(pred=type(pred), shape=pred.shape, times=times.shape):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", UserWarning)
                    with self.assertRaisesRegex(ValueError, message):
                        CALCULATE(
                            p, pred, noise, noise, times, batch(items, latents=noise)
                        )
                self.assertFalse(p.per_image_training.pending)

    def test_real_calculate_loss_mask_weights_and_gradient_oracle(self):
        cfg = dataset_config()
        items = [image(cfg, "train.png"), image(cfg, "reg.png", reg=True)]
        p = process(items=items, snr_gamma=2.0)
        p.sd.scale_loss = lambda loss: loss * 2
        p.per_image_training.watch.multiplier = Mock(return_value=0.25)
        b = batch(items, multipliers=[2.0, 7.0])
        pred = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]], [[[2.0, 3.0], [4.0, 5.0]]]], requires_grad=True
        )
        mask = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]], [[[0.0, 1.0], [0.0, 1.0]]]])
        noise = torch.zeros_like(pred)
        times = torch.tensor([100.0, 900.0])
        p.sd.noise_scheduler = object()
        fn = extract_method(
            SD_SOURCE,
            "calculate_loss",
            {
                "apply_snr_weight": lambda loss, *a, **k: (
                    loss * torch.tensor([3.0, 5.0])
                ),
            },
        )
        loss = fn(p, pred, noise, noise, times, b, mask_multiplier=mask)
        loss.backward()
        # Independent pixel oracle: MSE * model factor * mask * image weight *
        # dataset multiplier * SNR factor / (batch size * pixels per image).
        factors = torch.tensor([0.25 * 2 * 3, 1.0 * 7 * 5]).view(2, 1, 1, 1)
        expected_gradient = 2 * pred.detach() * 2 * mask * factors / 8
        torch.testing.assert_close(pred.grad, expected_gradient)
        expected_loss = (pred.detach().square() * 2 * mask * factors).sum() / 8
        torch.testing.assert_close(loss, expected_loss)
        # Watcher sees only eligible raw masked MSE, before dataset/SNR/image factors.
        self.assertEqual(
            p.per_image_training.pending,
            [("0:train.png", 0, 0.10000000149011612, 2.75)],
        )
        p.per_image_training.watch.multiplier.assert_called_once_with("0:train.png")

    def test_observe_mode_matches_unmodified_gradient(self):
        items = [image(dataset_config(), "a.png"), image(dataset_config(), "b.png")]
        p = process("observe", items=items)
        b = batch(items, multipliers=[0.3, 2.0])
        values = torch.arange(1.0, 9.0).reshape(2, 1, 2, 2)
        mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]], [[[0.2, 0.3], [0.5, 1.0]]]])
        gradients, losses = [], []
        for enabled in (True, False):
            pred = values.clone().requires_grad_()
            bridge = p.per_image_training
            if not enabled:
                p.per_image_training = None
            loss = CALCULATE(
                p,
                pred,
                torch.zeros_like(pred),
                torch.zeros_like(pred),
                torch.tensor([200.0, 800.0]),
                b,
                mask_multiplier=mask,
            )
            loss.backward()
            gradients.append(pred.grad.clone())
            losses.append(loss.detach())
            p.per_image_training = bridge
        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)

    def test_coverage_compares_masked_error_per_unit_of_mask(self):
        cfg = dataset_config()
        items = [image(cfg, name) for name in ("small.png", "large.png", "empty.png")]
        p = process(
            items=items, per_image_adaptive={"mode": "observe", "policy": "coverage"}
        )
        pred = torch.full((3, 1, 2, 2), 2.0, requires_grad=True)
        zero = torch.zeros_like(pred)
        mask = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ]
        )
        times = torch.full((3,), 500.0)
        loss = CALCULATE(p, pred, zero, zero, times, batch(items), mask_multiplier=mask)
        loss.backward()
        self.assertAlmostEqual(loss.item(), 5.0 / 3)
        torch.testing.assert_close(pred.grad, mask / 3)
        self.assertEqual(
            p.per_image_training.pending,
            [("0:small.png", 0, 0.5, 4.0), ("0:large.png", 0, 0.5, 4.0)],
        )

    def test_coverage_weights_survive_batch_and_accumulation(self):
        cfg = dataset_config()
        items = [image(cfg, f"{i}.png", reg=i == 3) for i in range(5)]
        for size in (1, 2, 4):
            for count in (1, 2, 4):
                for outer in (False, True):
                    with self.subTest(size=size, accumulation=count, outer=outer):
                        p = process(
                            items=items,
                            per_image_adaptive={
                                "mode": "adaptive",
                                "policy": "coverage",
                            },
                            gradient_accumulation_steps=count if outer else 1,
                        )
                        bridge = p.per_image_training
                        for e in range(18):
                            for idx, value in (
                                (0, 5 + 12 * 0.9**e),
                                (1, 0.2),
                                (2, 4),
                                (4, 1),
                            ):
                                for sigma in (0.125, 0.875):
                                    bridge.watch.record(
                                        bridge.key(items[idx]), e, sigma, value
                                    )
                            bridge.watch.end_epoch(e)
                        bridge.active_epoch = p.epoch_num = 18
                        parameter = torch.nn.Parameter(torch.ones(4))
                        p.params = [parameter]
                        p.optimizer = torch.optim.SGD([parameter], lr=0.1)
                        routes = []
                        expected = torch.ones(4)
                        for i in range(count):
                            indices = [(i + j) % 4 for j in range(size)]
                            routes.append(
                                batch([items[idx] for idx in indices], epoch=18)
                            )
                            for idx in indices:
                                expected[idx] -= (
                                    0.2 * (1.1, 0.95, 0.75, 1.0)[idx] / (count * size)
                                )

                        def single(b, accum_scale):
                            indices = [items.index(item) for item in b.file_items]
                            pred = parameter[indices, None, None, None].expand(
                                -1, 1, 2, 2
                            )
                            zero = torch.zeros_like(pred)
                            loss = CALCULATE(
                                p, pred, zero, zero, torch.full((size,), 500.0), b
                            )
                            (loss * accum_scale).backward()
                            return loss.detach()

                        p.train_single_accumulation = single
                        if outer:
                            for i, microbatch in enumerate(routes):
                                p.is_grad_accumulation_step = i < count - 1
                                TRAIN(p, microbatch)
                                if p.is_grad_accumulation_step:
                                    torch.testing.assert_close(parameter, torch.ones(4))
                        else:
                            TRAIN(p, routes)
                        torch.testing.assert_close(parameter, expected)
                        state = bridge.watch.state_dict()
                        self.assertEqual(state["pending_epoch"], 18)
                        for idx in (0, 1, 2, 4):
                            visits = sum(
                                item is items[idx]
                                for b in routes
                                for item in b.file_items
                            )
                            self.assertEqual(
                                state["images"][f"0:{idx}.png"]["observations"],
                                36 + visits,
                            )
                        self.assertNotIn("0:3.png", state["images"])

    def test_coverage_resume_confirms_only_after_a_successful_update(self):
        cfg = dataset_config()
        items = [image(cfg, f"{i}.png") for i in range(4)]
        settings = {"mode": "adaptive", "policy": "coverage"}
        p = process(items=items, per_image_adaptive=settings)
        bridge = p.per_image_training
        key = bridge.key(items[0])
        for e in range(16):
            for i, item in enumerate(items):
                for sigma in (0.125, 0.875):
                    bridge.watch.record(bridge.key(item), e, sigma, 4 if i == 0 else 1)
            bridge.watch.end_epoch(e)
        bridge.active_epoch = 16
        metadata = json.loads(
            json.dumps(
                {
                    "per_image_adaptive": bridge.state_dict(),
                    "training_info": {"step": 32, "epoch": 16},
                }
            )
        )
        resumed = process(items=items, per_image_adaptive=settings)
        load = extract_method(
            BASE_SOURCE,
            "load_training_state_from_metadata",
            {"load_metadata_from_safetensors": lambda path: metadata},
        )
        load(resumed, "checkpoint.safetensors")
        bridge = resumed.per_image_training
        bridge.bind([NS(file_list=items)])
        self.assertEqual(bridge.watch.report()["images"][key]["votes"], 1)
        self.assertEqual(bridge.watch.multiplier(key), 1.0)
        parameter = self._install_train(resumed)

        def single(b, accum_scale):
            pred = parameter * torch.tensor([2.0, 1.0, 1.0, 1.0]).view(-1, 1, 1, 1)
            pred = pred.expand(-1, 1, 2, 2)
            zero = torch.zeros_like(pred)
            loss = CALCULATE(resumed, pred, zero, zero, b.times, b)
            (loss * accum_scale).backward()
            return loss.detach()

        resumed.train_single_accumulation = single
        bad = batch(items, epoch=16)
        bad.times = torch.tensor([float("nan")] * len(items))
        valid = batch(items, epoch=16)
        valid.times = torch.full((len(items),), 125.0)
        before = bridge.state_dict()
        TRAIN(resumed, [valid, bad])
        self.assertEqual(bridge.state_dict(), before)
        self.assertEqual(parameter.item(), 1.0)
        self.assertFalse(bridge.pending)

        high = batch(items, epoch=16)
        high.times = torch.full((len(items),), 875.0)
        resumed.epoch_num = 17
        TRAIN(resumed, [valid, high])
        self.assertEqual(bridge.watch.report()["images"][key]["votes"], 2)
        self.assertEqual(bridge.watch.multiplier(key), 1.0)
        self.assertEqual(bridge.watch.report()["images"][key]["observations"], 34)

    def test_duplicate_basenames_roles_and_repeats_have_stable_keys(self):
        a, b = dataset_config("/dataset/a"), dataset_config("/dataset/b")
        role = dataset_config("/dataset/a", caption_dropout_rate=0.2)
        items = [image(a), image(b), image(role), image(a)]
        p = process(items=items)
        keys = [p.per_image_training.key(item) for item in items]
        self.assertEqual(keys, ["0:same.png", "1:same.png", "2:same.png", "0:same.png"])
        self.assertEqual(
            len(p.per_image_training.watch.state_dict()["dataset_keys"]), 3
        )

    def _install_train(self, p, *, invalid=False):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        p.params = [parameter]
        p.optimizer = torch.optim.SGD([parameter], lr=0.1)

        def single(b, accum_scale):
            pred = parameter.expand(len(b.file_items), 1, 2, 2)
            if invalid:
                pred = pred * float("nan")
            noise = torch.zeros_like(pred)
            times = torch.full((len(b.file_items),), 500.0)
            loss = CALCULATE(p, pred, noise, noise, times, b)
            (loss * accum_scale).backward()
            return loss.detach()

        p.train_single_accumulation = single
        return parameter

    def test_real_hook_accumulates_across_epoch_and_commits_only_after_update(self):
        item = image(dataset_config())
        p = process(items=[item], gradient_accumulation_steps=2)
        parameter = self._install_train(p)
        p.is_grad_accumulation_step = True
        TRAIN(p, batch([item], epoch=0))
        self.assertEqual(float(parameter.detach()), 1.0)
        self.assertEqual(
            p.per_image_training.watch.report()["images"]["0:same.png"]["observations"],
            0,
        )
        self.assertEqual(len(p.per_image_training.pending), 1)
        self.assertEqual(len(p.per_image_training.gradients), 1)
        p.epoch_num = 1
        p.is_grad_accumulation_step = False
        TRAIN(p, batch([item], epoch=1))
        self.assertAlmostEqual(float(parameter.detach()), 0.8, places=6)
        state = p.per_image_training.watch.state_dict()
        self.assertEqual(state["images"]["0:same.png"]["observations"], 2)
        self.assertEqual(state["last_epoch"], 0)
        self.assertEqual(state["pending_epoch"], 1)
        self.assertEqual(p.per_image_training.pending, [])
        p.per_image_training.write_report.assert_called_once()

    def test_real_hook_invalid_update_discards_whole_accumulation(self):
        item = image(dataset_config())
        p = process(items=[item], gradient_accumulation_steps=2)
        parameter = self._install_train(p)
        before = p.per_image_training.watch.state_dict()
        p.is_grad_accumulation_step = True
        TRAIN(p, batch([item]))
        p.per_image_training.invalid = True
        p.is_grad_accumulation_step = False
        TRAIN(p, batch([item]))
        self.assertEqual(float(parameter.detach()), 1.0)
        self.assertEqual(p.per_image_training.watch.state_dict(), before)
        self.assertEqual(p.per_image_training.pending, [])
        self.assertEqual(p.per_image_training.gradients, [])
        self.assertIsNone(parameter.grad)

    def test_real_hook_nonfinite_gradients_discard_finite_observations(self):
        item = image(dataset_config())
        p = process(items=[item])
        parameter = self._install_train(p)
        original = p.train_single_accumulation

        def single(b, accum_scale):
            result = original(b, accum_scale)
            parameter.grad.fill_(float("inf"))
            return result

        p.train_single_accumulation = single
        before = p.per_image_training.watch.state_dict()
        TRAIN(p, batch([item]))
        self.assertEqual(float(parameter.detach()), 1.0)
        self.assertEqual(p.per_image_training.watch.state_dict(), before)

    def test_backward_updating_optimizers_fail_before_training(self):
        for name, options in (
            ("automagic2", {}),
            ("automagic2", {"fused": False}),
            ("automagic3", {}),
            ("automagic4", {}),
            ("automagicexperiment", {}),
            ("adamconvrot", {"fused": True}),
        ):
            with self.subTest(optimizer=name, options=options):
                with self.assertRaisesRegex(ValueError, "updates in step"):
                    process(optimizer=name, optimizer_params=options)
        # Original AutoMagic always updates in step and has no fused option.
        process(optimizer="automagic")
        process(optimizer="adamconvrot")

    def test_real_low_precision_optimizers_expose_nonfinite_buffers_before_step(self):
        from toolkit.optimizers.automagic3 import Automagic3
        from toolkit.optimizers.automagic4 import Automagic4

        for name, optimizer in (("automagic3", Automagic3), ("automagic4", Automagic4)):
            with self.subTest(optimizer=name):
                item = image(dataset_config())
                p = process(
                    items=[item], optimizer=name, optimizer_params={"fused": False}
                )
                parameter = self._install_train(p)
                parameter.data = parameter.data.bfloat16()
                p.optimizer = optimizer([parameter], lr=0.001, fused=False)
                original = p.train_single_accumulation

                def single(b, accum_scale):
                    loss = original(b, accum_scale)
                    self.assertIsNone(parameter.grad)
                    parameter._accum_grad.fill_(float("inf"))
                    return loss

                p.train_single_accumulation = single
                before = p.per_image_training.watch.state_dict()
                TRAIN(p, batch([item]))
                self.assertEqual(float(parameter.detach()), 1.0)
                self.assertEqual(p.per_image_training.watch.state_dict(), before)
                self.assertEqual(p.optimizer.state, {})
                self.assertFalse(hasattr(parameter, "_accum_grad"))

    def test_real_low_precision_optimizer_clips_buffer_before_step(self):
        from toolkit.optimizers.automagic3 import Automagic3

        item = image(dataset_config())
        p = process(
            items=[item],
            optimizer="automagic3",
            optimizer_params={"fused": False},
            max_grad_norm=0.5,
            gradient_accumulation_steps=2,
        )
        parameter = self._install_train(p)
        parameter.data = parameter.data.bfloat16()
        p.optimizer = Automagic3([parameter], lr=0.001, fused=False)
        norms = []

        def clip(parameters, limit):
            norms.append(torch.nn.utils.clip_grad_norm_(parameters, limit).item())
            self.assertEqual(parameter.grad.item(), 0.5)

        p.accelerator.clip_grad_norm_ = clip
        p.is_grad_accumulation_step = True
        TRAIN(p, batch([item]))
        self.assertEqual(p.optimizer.state, {})
        p.is_grad_accumulation_step = False
        TRAIN(p, batch([item]))
        self.assertEqual(norms, [2.0])
        self.assertEqual(p.optimizer.state[parameter]["step"], 1)
        self.assertEqual(
            p.per_image_training.watch.report()["images"]["0:same.png"]["observations"],
            2,
        )

    def test_oom_discard_clears_real_low_precision_optimizer_buffer(self):
        from toolkit.optimizers.automagic3 import Automagic3

        item = image(dataset_config())
        p = process(
            items=[item], optimizer="automagic3", optimizer_params={"fused": False}
        )
        parameter = self._install_train(p)
        parameter.data = parameter.data.bfloat16()
        p.optimizer = Automagic3([parameter], lr=0.001, fused=False)
        p.is_grad_accumulation_step = True
        TRAIN(p, batch([item]))
        self.assertTrue(hasattr(parameter, "_accum_grad"))
        p.per_image_training.discard()
        self.assertFalse(hasattr(parameter, "_accum_grad"))
        self.assertFalse(p.per_image_training.pending)

    def test_accelerator_skipped_step_discards_observations(self):
        item = image(dataset_config())
        p = process(items=[item])
        parameter = self._install_train(p)
        p.optimizer.step = Mock()  # The accelerator's skipped optimizer step.
        p.accelerator.optimizer_step_was_skipped = True
        before = p.per_image_training.watch.state_dict()
        TRAIN(p, batch([item]))
        self.assertEqual(float(parameter.detach()), 1.0)
        self.assertEqual(p.per_image_training.watch.state_dict(), before)
        self.assertFalse(p.per_image_training.pending)

    def test_real_oom_recovery_branch_discards_pending_observations_and_gradients(self):
        item = image(dataset_config())
        p = process(items=[item])
        parameter = self._install_train(p)
        p.is_grad_accumulation_step = True
        TRAIN(p, batch([item]))
        before = p.per_image_training.watch.state_dict()
        tree = ast.parse(BASE_SOURCE.read_text())
        branch = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name)
            and n.test.id == "did_oom"
        )
        scope = {
            "self": p,
            "optimizer": p.optimizer,
            "flush": lambda: None,
            "torch": NS(cuda=NS(ipc_collect=lambda: None)),
            "print_acc": lambda *a: None,
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=branch.body, type_ignores=[])
                ),
                str(BASE_SOURCE),
                "exec",
            ),
            scope,
        )
        self.assertIsNone(parameter.grad)
        self.assertFalse(p.per_image_training.pending)
        self.assertFalse(p.per_image_training.gradients)
        self.assertEqual(p.per_image_training.watch.state_dict(), before)

    def test_metadata_delays_load_until_binding_and_rejects_context_drift(self):
        item = image(dataset_config())
        p = process(items=[item])
        bridge = p.per_image_training
        bridge.watch.record("0:same.png", 0, 0.5, 1.0)
        bridge.watch.end_epoch(0)
        bridge.active_epoch = 1
        metadata = {
            "per_image_adaptive": json.loads(json.dumps(bridge.state_dict())),
            "training_info": {"step": 8, "epoch": 1},
        }
        load = extract_method(
            BASE_SOURCE,
            "load_training_state_from_metadata",
            {
                "load_metadata_from_safetensors": lambda path: copy.deepcopy(metadata),
            },
        )
        resumed = process(items=[item])
        before = resumed.per_image_training.watch.state_dict()
        load(resumed, "checkpoint.safetensors")
        self.assertEqual(resumed.per_image_training.watch.state_dict(), before)
        self.assertIsNotNone(resumed._per_image_resume_state)
        resumed.per_image_training.bind([NS(file_list=[item])])
        self.assertEqual(resumed.per_image_training.state_dict(), bridge.state_dict())
        self.assertIsNone(resumed._per_image_resume_state)
        self.assertEqual(resumed.epoch_num, 1)
        bad = process(items=[item])
        load(bad, "checkpoint.safetensors")
        bad.model_config.name_or_path = "different-base"
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            bad.per_image_training.bind([NS(file_list=[item])])

    def test_missing_checkpoint_watcher_state_is_rejected(self):
        p = process()
        load = extract_method(
            BASE_SOURCE,
            "load_training_state_from_metadata",
            {
                "load_metadata_from_safetensors": lambda path: {},
            },
        )
        with self.assertRaisesRegex(ValueError, "no per-image adaptive state"):
            load(p, "checkpoint.safetensors")

    def test_standalone_checkpoint_resumes_with_default_grouping_and_sigmoid_patches(
        self,
    ):
        item = image(dataset_config())
        source = process(items=[item])
        bridge = source.per_image_training
        bridge.watch.record("0:same.png", 0, 0.5, 1.0)
        snapshot = bridge.state_dict()
        for drop_keys in (False, True):
            with self.subTest(drop_keys=drop_keys):
                previous = copy.deepcopy(snapshot)
                if drop_keys:
                    del previous["context"]["objective"]["timestep_shift"]
                    del previous["context"]["objective"]["timestep_scale"]
                resumed = process(items=[item], timestep_shift=0.0, timestep_scale=1.0)
                resumed.network_config.optimizer_grouping = "default"
                resumed._per_image_resume_state = previous
                resumed.per_image_training.bind([NS(file_list=[item])])
                self.assertEqual(
                    resumed.per_image_training.watch.state_dict(), snapshot["watcher"]
                )
                self.assertIsNone(resumed._per_image_resume_state)
        for field, value in (
            ("timestep_shift", -0.5),
            ("timestep_scale", 0.75),
            ("optimizer_grouping", "transformer_block"),
        ):
            with self.subTest(field=field):
                resumed = process(items=[item], timestep_shift=0.0, timestep_scale=1.0)
                resumed.network_config.optimizer_grouping = "default"
                config = (
                    resumed.network_config
                    if field == "optimizer_grouping"
                    else resumed.train_config
                )
                setattr(config, field, value)
                resumed._per_image_resume_state = copy.deepcopy(snapshot)
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    resumed.per_image_training.bind([NS(file_list=[item])])

    def test_explicit_start_step_with_stale_trainer_epoch_fails_during_bind(self):
        item = image(dataset_config())
        source = process(items=[item])
        source.per_image_training.watch.end_epoch(0)
        source.per_image_training.active_epoch = 1
        snapshot = source.per_image_training.state_dict()
        resumed = process(items=[item], start_step=8)
        load = extract_method(
            BASE_SOURCE,
            "load_training_state_from_metadata",
            {
                "load_metadata_from_safetensors": lambda path: {
                    "per_image_adaptive": copy.deepcopy(snapshot),
                    "training_info": {"step": 8, "epoch": 1},
                },
            },
        )
        load(resumed, "checkpoint.safetensors")
        self.assertEqual(resumed.epoch_num, 0)
        before = resumed.per_image_training.watch.state_dict()
        with self.assertRaisesRegex(ValueError, "trainer epoch.*watcher epoch"):
            resumed.per_image_training.bind([NS(file_list=[item])])
        self.assertEqual(resumed.per_image_training.watch.state_dict(), before)

    def test_resume_rejects_inconsistent_active_epoch_before_watcher_load(self):
        p = process()
        snapshot = p.per_image_training.state_dict()
        snapshot["active_epoch"] = 3
        p._per_image_resume_state = snapshot
        before = p.per_image_training.watch.state_dict()
        with self.assertRaisesRegex(ValueError, "epoch disagrees"):
            p.per_image_training.bind([NS(file_list=[image(dataset_config())])])
        self.assertEqual(p.per_image_training.watch.state_dict(), before)
        self.assertEqual(p._per_image_resume_state, snapshot)

    def test_altered_targets_are_rejected_without_pending_observations(self):
        item = image(dataset_config())
        p = process(items=[item])
        b = batch([item])
        times = torch.tensor([500.0])
        for target in (torch.ones_like(b.latents), torch.zeros(1)):
            with self.subTest(target=target.shape):
                p.sd.get_loss_target = lambda **kwargs: target
                with warnings.catch_warnings():
                    warnings.simplefilter("error", UserWarning)
                    with self.assertRaisesRegex(ValueError, "unmodified flow velocity"):
                        CALCULATE(p, b.latents, b.latents, b.latents, times, b)
                self.assertFalse(p.per_image_training.pending)

    def test_mixed_velocity_signs_in_one_batch_are_rejected(self):
        items = [image(dataset_config(), name) for name in ("a.png", "b.png")]
        p = process(items=items)
        noise = torch.ones((2, 1, 2, 2))
        p.sd.get_loss_target = lambda **kwargs: (
            noise * torch.tensor([1.0, -1.0]).view(2, 1, 1, 1)
        )
        with self.assertRaisesRegex(ValueError, "unmodified flow velocity"):
            CALCULATE(
                p, noise, noise, noise, torch.tensor([250.0, 750.0]), batch(items)
            )
        self.assertFalse(p.per_image_training.pending)

    def test_nonfinite_targets_skip_optimizer_update_and_observations(self):
        item = image(dataset_config())
        p = process(items=[item])
        parameter = self._install_train(p)
        p.sd.get_loss_target = lambda **kwargs: torch.full_like(
            kwargs["noise"], float("nan")
        )
        before = p.per_image_training.watch.state_dict()
        TRAIN(p, batch([item]))
        self.assertEqual(float(parameter.detach()), 1.0)
        self.assertEqual(p.per_image_training.watch.state_dict(), before)
        self.assertFalse(p.per_image_training.pending)

    def test_nonfinite_loss_or_sigma_discards_entire_pending_update(self):
        item = image(dataset_config())
        for loss_value, timestep in (
            (float("nan"), 500.0),
            (1.0, float("inf")),
            (1.0, -1.0),
        ):
            with self.subTest(loss=loss_value, timestep=timestep):
                p = process(items=[item])
                p.optimizer = NS(param_groups=[])
                b = batch([item])
                before = p.per_image_training.watch.state_dict()
                p.per_image_training.apply(torch.ones(1), b, torch.tensor([500.0]))
                p.per_image_training.apply(
                    torch.tensor([loss_value]),
                    b,
                    torch.tensor([timestep]),
                )
                p.per_image_training.finish(stepped=True)
                self.assertEqual(p.per_image_training.watch.state_dict(), before)
                self.assertFalse(p.per_image_training.pending)

    def test_real_save_metadata_hook_serializes_authoritative_snapshot(self):
        p = process()
        bridge = p.per_image_training
        item = image(dataset_config())
        b = batch([item])
        bridge.apply(torch.ones(1), b, torch.tensor([500.0]))
        tree = ast.parse(BASE_SOURCE.read_text())
        branch = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and any(
                isinstance(x, ast.Assign)
                and any(
                    isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "save_meta"
                    and isinstance(t.slice, ast.Constant)
                    and t.slice.value == "per_image_adaptive"
                    for t in x.targets
                )
                for x in n.body
            )
        )
        metadata = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])),
                str(BASE_SOURCE),
                "exec",
            ),
            {"self": p, "save_meta": metadata},
        )
        self.assertEqual(metadata["per_image_adaptive"], bridge.state_dict())
        self.assertNotIn("pending", metadata["per_image_adaptive"])
        self.assertEqual(
            sum(metadata["per_image_adaptive"]["watcher"]["bin_counts"]), 0
        )
        bridge.write_report.assert_called_once()

    def test_unsupported_prediction_and_auxiliary_settings_fail_early(self):
        for options in (
            {"pred_scaler": 0.5},
            {"target_noise_multiplier": 0.5},
            {"match_noise_norm": True},
            {"t0_loss_target": True},
            {"do_fft_loss": True},
            {"target_norm_std": True},
            {"diffusion_feature_extractor_path": "/feature/model"},
            {"single_item_batching": True},
            {"inverted_mask_prior": True},
            {"do_differential_guidance": True},
            {"short_and_long_captions": True},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    process(**options)
        with self.assertRaisesRegex(ValueError, "unconditional image"):
            process(items=[image(dataset_config(unconditional_path="/unconditional"))])
        p = process()
        p.guidance_config = NS()
        with self.assertRaises(ValueError):
            validate_process(p)

    @unittest.skipIf(
        InfoNoiseSampler is None, "standalone patch has no InfoNoise sampler"
    )
    def test_real_infonoise_and_watcher_share_raw_masked_loss_before_weighting(self):
        self.assertIn("infonoise_sampler", CALCULATE.__code__.co_names)
        self.assertIn("infonoise_sampler", TRAIN.__code__.co_names)
        cfg = dataset_config()
        items = [
            image(cfg, "a.png"),
            image(cfg, "b.png"),
            image(cfg, "reg.png", reg=True),
        ]
        for skipped in (False, True):
            with self.subTest(skipped=skipped):
                p = process(items=items, timestep_type="infonoise")
                p.infonoise_sampler = InfoNoiseSampler(
                    {"bins": 4, "warmup_steps": 100}, seed=13
                )
                p.sd.scale_loss = lambda loss: loss * 2
                p.per_image_training.watch.multiplier = Mock(
                    side_effect=lambda key: 0.25 if key.endswith("a.png") else 1.1
                )
                parameter = torch.nn.Parameter(torch.tensor(2.0))
                p.params = [parameter]
                p.optimizer = torch.optim.SGD([parameter], lr=0.01)
                p.accelerator.optimizer_step_was_skipped = skipped
                if skipped:
                    p.optimizer.step = Mock()
                b = batch(items, multipliers=[7.0, 11.0, 13.0])
                times = torch.tensor([250.0, 750.0, 500.0])
                masks = (
                    torch.tensor([1.0, 0.5, 0.25]).view(3, 1, 1, 1).expand(3, 1, 2, 2)
                )
                captured = {}

                def single(actual_batch, accum_scale):
                    pred = parameter.expand(3, 1, 2, 2)
                    noise = torch.zeros_like(pred)
                    loss = CALCULATE(
                        p,
                        pred,
                        noise,
                        noise,
                        times,
                        actual_batch,
                        mask_multiplier=masks,
                    )
                    captured["info"] = list(p.infonoise_sampler.pending)
                    captured["images"] = list(p.per_image_training.pending)
                    (loss * accum_scale).backward()
                    captured["gradient"] = parameter.grad.detach().clone()
                    return loss.detach()

                p.train_single_accumulation = single
                TRAIN(p, b)
                # InfoNoise converts the same raw velocity MSE to clean-data MSE
                # with squared flow time. Regularization is excluded from both estimates.
                self.assertEqual([row[-1] for row in captured["images"]], [8.0, 4.0])
                self.assertEqual([row[1] for row in captured["info"]], [0.5, 2.25])
                expected_gradient = 8 * (0.25 * 7 + 0.5 * 1.1 * 11 + 0.25 * 13) / 3
                self.assertAlmostEqual(
                    float(captured["gradient"]), expected_gradient, places=5
                )
                self.assertAlmostEqual(
                    float(parameter.detach()),
                    2.0 if skipped else 2.0 - 0.01 * expected_gradient,
                    places=6,
                )
                self.assertFalse(p.infonoise_sampler.pending)
                self.assertFalse(p.per_image_training.pending)
                self.assertEqual(
                    p.infonoise_sampler.optimizer_steps, 0 if skipped else 1
                )
                self.assertEqual(
                    int(p.infonoise_sampler.total_counts.sum()), 0 if skipped else 2
                )
                self.assertEqual(
                    sum(p.per_image_training.watch.state_dict()["bin_counts"]),
                    0 if skipped else 2,
                )


if __name__ == "__main__":
    unittest.main()
