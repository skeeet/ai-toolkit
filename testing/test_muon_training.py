"""Exercise the real trainer hooks with tiny matrices, without model imports."""

import ast
import copy
import importlib.util
import unittest
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

from toolkit.optimizer import get_optimizer
from toolkit.optimizers.muon import Muon
from toolkit.optimizers.muon_training import validate_process

ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE_MODES = (
    (False, True)
    if importlib.util.find_spec("toolkit.per_image_training")
    else (False,)
)


def actual_train_hook():
    source = ast.parse(
        (ROOT / "extensions_built_in/sd_trainer/SDTrainer.py").read_text()
    )
    method = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef) and node.name == "hook_train_loop"
    )
    method = copy.deepcopy(method)
    method.decorator_list = []
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = dict(
        torch=torch,
        OrderedDict=OrderedDict,
        sync_grad_transfers=lambda: None,
        CustomAdapter=type("CustomAdapter", (), {}),
        ClipVisionAdapter=type("ClipVisionAdapter", (), {}),
    )
    if importlib.util.find_spec("toolkit.optimizer_param_groups"):
        from toolkit.optimizer_param_groups import clip_optimizer_groups

        namespace["clip_optimizer_groups"] = clip_optimizer_groups
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            "SDTrainer.hook_train_loop",
            "exec",
        ),
        namespace,
    )
    return namespace["hook_train_loop"]


def process(*, accumulation=1, adaptive=False):
    parameter = torch.nn.Parameter(torch.tensor([[0.3, -0.1], [0.5, 0.7]]))
    optimizer = Muon([parameter], lr=0.01, weight_decay=0.01)
    train = SimpleNamespace(
        optimizer="muon",
        optimizer_params={},
        gradient_accumulation_steps=accumulation,
        max_grad_norm=1000.0,
        train_unet=True,
        train_text_encoder=False,
        do_paramiter_swapping=False,
        merge_network_on_save=False,
        loss_type="mse",
        loss_target="noise",
        train_turbo=False,
        do_guidance_loss=False,
        per_image_adaptive={"mode": "observe"},
    )
    accelerator = SimpleNamespace(
        num_processes=1,
        state=SimpleNamespace(fsdp_plugin=None),
        optimizer_step_was_skipped=False,
        clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
    )
    result = SimpleNamespace(
        raw_process_config={"model": {}},
        supports_per_image_adaptation=True,
        train_config=train,
        model_config=SimpleNamespace(
            arch="moss_image",
            name_or_path="MOSS-Image-Base",
            model_kwargs={},
            low_vram=False,
        ),
        network_config=SimpleNamespace(type="lokr"),
        network=SimpleNamespace(unet_loras=[torch.nn.ParameterList([parameter])]),
        accelerator=accelerator,
        adapter_config=None,
        embed_config=None,
        decorator_config=None,
        optimizer=optimizer,
        params=[{"params": [parameter]}],
        sd=SimpleNamespace(is_multistage=False),
        steps_this_boundary=0,
        is_grad_accumulation_step=False,
        timer=lambda _: nullcontext(),
        adapter=None,
        ema=None,
        embedding=None,
        lr_scheduler=SimpleNamespace(step=lambda: None),
        end_of_training_loop=lambda: None,
        epoch_num=0,
        additional_logs={},
        dataset_configs=[],
        num_consecutive_oom=0,
        per_image_training=None,
        infonoise_sampler=None,
    )

    def backward(batch, accum_scale=1):
        # Each batch specifies a different gradient direction. Losing earlier
        # gradients cannot be hidden by Muon's gradient-scale invariance.
        loss = (parameter * batch).sum()
        (loss * accum_scale).backward()
        return loss.detach()

    result.train_single_accumulation = backward
    result.hook_train_loop = MethodType(actual_train_hook(), result)
    if adaptive:
        from toolkit.per_image_training import PerImageTraining

        result.per_image_training = PerImageTraining(result)
    return result, parameter


class MuonTrainingTest(unittest.TestCase):
    batches = [
        torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[0.0, 2.0], [0.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [-3.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [0.0, 4.0]]),
    ]

    def reference(self, parameter, batches, *, average=True):
        result = torch.nn.Parameter(parameter.detach().clone())
        optimizer = Muon([result], lr=0.01, weight_decay=0.01)
        gradient = torch.stack(batches).sum(0)
        result.grad = gradient / len(batches) if average else gradient
        optimizer.step()
        return result, optimizer.state[result]["momentum_buffer"]

    def test_batch_list_matches_one_combined_update(self):
        for adaptive in ADAPTIVE_MODES:
            with self.subTest(adaptive=adaptive):
                trainer, parameter = process(adaptive=adaptive)
                expected, momentum = self.reference(parameter, self.batches)
                trainer.hook_train_loop(self.batches)
                torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    trainer.optimizer.state[parameter]["momentum_buffer"],
                    momentum,
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(trainer._muon_accumulated_grads, [])

    def test_repeated_calls_match_combined_update_with_and_without_watcher(self):
        for adaptive in ADAPTIVE_MODES:
            with self.subTest(adaptive=adaptive):
                trainer, parameter = process(accumulation=4, adaptive=adaptive)
                initial = parameter.detach().clone()
                expected, momentum = self.reference(parameter, self.batches)
                for index, batch in enumerate(self.batches):
                    trainer.is_grad_accumulation_step = index < 3
                    trainer.hook_train_loop(batch)
                    if index < 3:
                        torch.testing.assert_close(parameter, initial, rtol=0, atol=0)
                        self.assertIs(
                            trainer._muon_accumulated_grads[0][1], parameter.grad
                        )
                torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    trainer.optimizer.state[parameter]["momentum_buffer"],
                    momentum,
                    rtol=0,
                    atol=0,
                )
                self.assertIsNone(parameter.grad)
                self.assertEqual(trainer._muon_accumulated_grads, [])

    def test_legacy_whole_epoch_preserves_summed_gradients(self):
        trainer, parameter = process(accumulation=-1)
        expected, momentum = self.reference(parameter, self.batches, average=False)
        for index, batch in enumerate(self.batches):
            trainer.is_grad_accumulation_step = index < 3
            trainer.hook_train_loop(batch)
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            trainer.optimizer.state[parameter]["momentum_buffer"],
            momentum,
            rtol=0,
            atol=0,
        )

    def test_each_group_is_clipped_after_accumulating_all_microbatches(self):
        trainer, first = process(accumulation=2)
        second = torch.nn.Parameter(torch.ones_like(first))
        trainer.params.append({"params": [second], "lr": 0.02})
        trainer.optimizer = get_optimizer(trainer.params, "muon")
        trainer.train_config.max_grad_norm = 1.0

        def backward(batch, accum_scale=1):
            left, right = batch
            loss = (first * left).sum() + (second * right).sum()
            (loss * accum_scale).backward()
            return loss.detach()

        trainer.train_single_accumulation = backward
        trainer.is_grad_accumulation_step = True
        trainer.hook_train_loop(
            (
                torch.tensor([[6.0, 0.0], [0.0, 0.0]]),
                torch.tensor([[0.0, 0.0], [12.0, 0.0]]),
            )
        )
        self.assertEqual(trainer.optimizer.state, {})
        trainer.is_grad_accumulation_step = False
        trainer.hook_train_loop(
            (
                torch.tensor([[0.0, 8.0], [0.0, 0.0]]),
                torch.tensor([[0.0, 0.0], [0.0, -16.0]]),
            )
        )

        # The averaged gradients have norms 5 and 10. Each clips to 1 before
        # Muon stores 5% of that gradient in its first momentum buffer.
        torch.testing.assert_close(
            trainer.optimizer.state[first]["momentum_buffer"],
            torch.tensor([[0.03, 0.04], [0.0, 0.0]]),
        )
        torch.testing.assert_close(
            trainer.optimizer.state[second]["momentum_buffer"],
            torch.tensor([[0.0, 0.0], [0.03, -0.04]]),
        )

    def test_actual_oom_cleanup_drops_accumulated_gradient_references(self):
        for adaptive in ADAPTIVE_MODES:
            with self.subTest(adaptive=adaptive):
                trainer, parameter = process(accumulation=2, adaptive=adaptive)
                trainer.is_grad_accumulation_step = True
                trainer.hook_train_loop(self.batches[0])
                trainer._muon_invalid_loss = True
                source = ast.parse(
                    (ROOT / "jobs/process/BaseSDTrainProcess.py").read_text()
                )
                handler = next(
                    node
                    for node in ast.walk(source)
                    if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name)
                    and node.test.id == "did_oom"
                    and any(
                        isinstance(child, ast.Attribute)
                        and child.attr == "num_consecutive_oom"
                        for child in ast.walk(node)
                    )
                )
                namespace = dict(
                    self=trainer,
                    did_oom=True,
                    optimizer=trainer.optimizer,
                    flush=lambda: None,
                    print_acc=lambda *args: None,
                    torch=SimpleNamespace(
                        cuda=SimpleNamespace(ipc_collect=lambda: None)
                    ),
                )
                exec(
                    compile(
                        ast.fix_missing_locations(
                            ast.Module(body=[copy.deepcopy(handler)], type_ignores=[])
                        ),
                        "BaseSDTrainProcess.OOM_cleanup",
                        "exec",
                    ),
                    namespace,
                )
                self.assertIsNone(parameter.grad)
                self.assertEqual(trainer._muon_accumulated_grads, [])
                self.assertFalse(trainer._muon_invalid_loss)
                if adaptive:
                    self.assertEqual(trainer.per_image_training.gradients, [])
                # New accumulation starts from scratch after the discarded call.
                expected, momentum = self.reference(parameter, self.batches[1:3])
                trainer.hook_train_loop(self.batches[1])
                trainer.is_grad_accumulation_step = False
                trainer.hook_train_loop(self.batches[2])
                torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    trainer.optimizer.state[parameter]["momentum_buffer"],
                    momentum,
                    rtol=0,
                    atol=0,
                )

    def test_denoiser_adapter_eligibility_is_architecture_independent(self):
        for arch, name, kind in (
            ("krea2", "Krea-2-Raw", "lora"),
            ("krea2:turbo", "Krea-2-Turbo", "lokr"),
            ("qwen_image_2", "Qwen-Image-2.1", "locon"),
            ("qwen_image_2:turbo", "Qwen-Image-2.1", "lycoris"),
            ("zimage:turbo", "Z-Image-Turbo", "lorm"),
            ("moss_image", "MOSS-Image-Base", "lora"),
            ("moss_image:base", "MOSS-Image-Base", "lokr"),
            ("sdxl", "example/sdxl-sft", "DORA"),
        ):
            with self.subTest(arch=arch, kind=kind):
                trainer, parameter = process()
                trainer.model_config.arch = arch
                trainer.model_config.name_or_path = name
                trainer.network_config.type = kind
                validate_process(trainer)
                optimizer = get_optimizer(trainer.params, "muon")
                self.assertIs(optimizer.param_groups[0]["params"][0], parameter)

    def test_rejects_other_training_modes(self):
        cases = [
            ("network_config", "type", "fullrank"),
            ("model_config", "name_or_path", "MOSS-Image-SFT"),
            ("model_config", "arch", "moss_image:base_edit"),
            ("model_config", "arch", "krea2:o_edit"),
            ("model_config", "arch", "qwen_image_edit_plus"),
            ("sd", "is_edit", True),
            ("sd", "is_llm", True),
            ("accelerator", "num_processes", 2),
            ("train_config", "train_unet", False),
            ("train_config", "train_text_encoder", True),
            ("train_config", "do_paramiter_swapping", True),
            ("train_config", "merge_network_on_save", True),
        ]
        for component, name, value in cases:
            with self.subTest(component=component, name=name):
                trainer, _ = process()
                setattr(getattr(trainer, component), name, value)
                with self.assertRaisesRegex(ValueError, "Muon"):
                    validate_process(trainer)
        for attr in ("adapter_config", "embed_config", "decorator_config"):
            with self.subTest(extra_config=attr):
                trainer, _ = process()
                setattr(trainer, attr, object())
                with self.assertRaisesRegex(ValueError, "Muon"):
                    validate_process(trainer)
        trainer, _ = process()
        trainer.network_config = None
        with self.assertRaisesRegex(ValueError, "denoiser adapter"):
            validate_process(trainer)
        trainer, _ = process()
        trainer.model_config.model_kwargs["edit"] = True
        with self.assertRaisesRegex(ValueError, "Muon"):
            validate_process(trainer)
        trainer, _ = process()
        trainer.accelerator.state.fsdp_plugin = object()
        with self.assertRaisesRegex(ValueError, "Muon"):
            validate_process(trainer)

    def test_qwen_image_21_with_dataset_controls_is_edit_training(self):
        for dataset in (
            SimpleNamespace(
                control_path=["references"], control_from_same_folder=False
            ),
            SimpleNamespace(control_path=None, control_from_same_folder=True),
        ):
            with self.subTest(dataset=dataset):
                trainer, _ = process()
                trainer.model_config.arch = "qwen_image_2"
                trainer.model_config.name_or_path = "Qwen-Image-2.1"
                trainer.dataset_configs = [dataset]
                with self.assertRaisesRegex(ValueError, "edit"):
                    validate_process(trainer)

    def test_raw_model_tag_is_available_after_model_config_normalization(self):
        for arch in ("moss_image:base_edit", "moss_image:sft"):
            with self.subTest(arch=arch):
                trainer, _ = process()
                trainer.raw_process_config["model"]["arch"] = arch
                with self.assertRaisesRegex(ValueError, "Muon"):
                    validate_process(trainer)

    def test_optimizer_groups_must_contain_only_denoiser_adapter_parameters(self):
        extra = torch.nn.Parameter(torch.ones(2, 2))
        for entry in ({"params": [extra]}, extra):
            trainer, _ = process()
            trainer.params.append(entry)
            with self.assertRaisesRegex(ValueError, "extra trainable parameters"):
                validate_process(trainer)
        trainer, _ = process()
        trainer.params = [{"params": []}]
        with self.assertRaisesRegex(ValueError, "no denoiser adapter parameters"):
            validate_process(trainer)

    def test_nonmatrix_factors_have_actionable_optimizer_errors(self):
        for kind, shape in (
            ("locon", (2, 1, 3, 3)),
            ("lycoris", ()),
            ("dora", (2,)),
            ("lorm", (2,)),
        ):
            with self.subTest(kind=kind, shape=shape):
                trainer, parameter = process()
                trainer.network_config.type = kind
                factor = torch.nn.Parameter(torch.ones(shape))
                trainer.network.unet_loras[0].append(factor)
                trainer.params = [{"params": [parameter, factor]}]
                validate_process(trainer)
                with self.assertRaisesRegex(
                    ValueError, "nonempty 2D.*Choose another optimizer"
                ):
                    get_optimizer(trainer.params, "muon")

    def test_nonfinite_accumulation_skips_update_and_next_update_recovers(self):
        for adaptive in ADAPTIVE_MODES:
            with self.subTest(adaptive=adaptive):
                trainer, parameter = process(accumulation=2, adaptive=adaptive)
                initial = parameter.detach().clone()
                skipped = []
                trainer.infonoise_sampler = SimpleNamespace(
                    discard_pending=lambda: skipped.append(True)
                )
                trainer.is_grad_accumulation_step = True
                trainer.hook_train_loop(self.batches[0])
                trainer.is_grad_accumulation_step = False
                result = trainer.hook_train_loop(
                    torch.full_like(parameter, float("nan"))
                )
                self.assertEqual(result, {"loss": 0.0})
                self.assertEqual(skipped, [True])
                trainer.infonoise_sampler = None
                self.assertEqual(trainer.optimizer.state, {})
                self.assertEqual(trainer._muon_accumulated_grads, [])
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, initial, rtol=0, atol=0)
                if adaptive:
                    self.assertEqual(trainer.per_image_training.gradients, [])
                expected, momentum = self.reference(parameter, self.batches[1:3])
                trainer.is_grad_accumulation_step = True
                trainer.hook_train_loop(self.batches[1])
                trainer.is_grad_accumulation_step = False
                trainer.hook_train_loop(self.batches[2])
                torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    trainer.optimizer.state[parameter]["momentum_buffer"],
                    momentum,
                    rtol=0,
                    atol=0,
                )

    def test_replaced_nonfinite_loss_still_marks_optimizer_update_invalid(self):
        trainer, parameter = process(accumulation=2)
        source = ast.parse(
            (ROOT / "extensions_built_in/sd_trainer/SDTrainer.py").read_text()
        )
        guard = next(
            node
            for node in ast.walk(source)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and any(
                isinstance(child, ast.Attribute) and child.attr == "_muon_invalid_loss"
                for child in ast.walk(node)
            )
        )
        initial = parameter.detach().clone()
        namespace = dict(
            self=trainer,
            loss=torch.tensor(float("nan"), requires_grad=True),
            torch=torch,
            print_acc=lambda *args: None,
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[copy.deepcopy(guard)], type_ignores=[])
                ),
                "SDTrainer.nonfinite_loss",
                "exec",
            ),
            namespace,
        )
        self.assertTrue(torch.isfinite(namespace["loss"]))
        self.assertTrue(trainer._muon_invalid_loss)
        trainer.is_grad_accumulation_step = True
        trainer.hook_train_loop(self.batches[0])
        self.assertTrue(trainer._muon_invalid_loss)
        trainer.is_grad_accumulation_step = False
        trainer.hook_train_loop(self.batches[1])
        torch.testing.assert_close(parameter, initial, rtol=0, atol=0)
        self.assertEqual(trainer.optimizer.state, {})
        self.assertFalse(trainer._muon_invalid_loss)

    def test_real_resume_guard_rejects_optimizer_switch_in_both_directions(self):
        source = ast.parse((ROOT / "jobs/process/BaseSDTrainProcess.py").read_text())
        guard = next(
            node
            for node in ast.walk(source)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "load_optimizer"
            and any(
                isinstance(child, ast.Name) and child.id == "loaded_muon_state"
                for child in ast.walk(node)
            )
        )
        code = compile(
            ast.fix_missing_locations(
                ast.Module(body=[copy.deepcopy(guard)], type_ignores=[])
            ),
            "BaseSDTrainProcess.optimizer_resume",
            "exec",
        )
        matrix = torch.nn.Parameter(torch.ones(2, 2))
        muon = Muon([matrix])
        adam = torch.optim.AdamW([matrix])
        for optimizer, name, saved in (
            (muon, "muon", adam.state_dict()),
            (adam, "adamw", muon.state_dict()),
        ):
            with self.subTest(optimizer=name):
                namespace = dict(
                    load_optimizer=True,
                    optimizer=optimizer,
                    optimizer_type=name,
                    optimizer_state_file_path="unused",
                    flush=lambda: None,
                    print_acc=lambda *args: None,
                    torch=SimpleNamespace(
                        load=lambda *args, **kwargs: copy.deepcopy(saved)
                    ),
                )
                with self.assertRaisesRegex(ValueError, "Incompatible Muon"):
                    exec(code, namespace)
        # Existing non-Muon malformed-state behavior remains log-and-continue.
        messages = []
        namespace = dict(
            load_optimizer=True,
            optimizer=adam,
            optimizer_type="adamw",
            optimizer_state_file_path="unused",
            flush=lambda: None,
            print_acc=lambda message: messages.append(str(message)),
            torch=SimpleNamespace(load=lambda *args, **kwargs: 42),
        )
        exec(code, namespace)
        self.assertTrue(
            any("Failed to load optimizer state" in message for message in messages)
        )


if __name__ == "__main__":
    unittest.main()
