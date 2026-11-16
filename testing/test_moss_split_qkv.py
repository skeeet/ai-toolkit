"""Offline split-QKV MOSS training tests using tiny models, not downloaded weights."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel
from extensions_built_in.diffusion_models.moss_image.src.transformer import (
    MossImageParams,
    MossImageSplitQKVTransformer,
    MossImageTransformer,
)
from toolkit.config_modules import ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork


def holder(split=False):
    return MossImageModel(
        "cpu",
        ModelConfig(
            name_or_path="unused",
            arch="moss_image",
            model_kwargs={"split_qkv": split},
        ),
        dtype="float32",
    )


def tiny_model(dtype=torch.float32):
    return MossImageTransformer(
        MossImageParams(
            dim=16,
            ffn_dim=32,
            num_heads=2,
            num_layers=1,
            text_dim=16,
            freq_dim=16,
            latent_channels=4,
        )
    ).to(dtype=dtype)


def attach(model, owner, full_rank=True):
    model.requires_grad_(False)
    config = NetworkConfig(
        type="lokr", lokr_factor=4, lokr_full_rank=full_rank, linear=1, linear_alpha=1
    )
    with contextlib.redirect_stdout(io.StringIO()):
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=model,
            lora_dim=config.linear,
            alpha=config.linear_alpha,
            network_config=config,
            network_type="lokr",
            is_transformer=True,
            transformer_only=True,
            train_text_encoder=False,
            base_model=owner,
            target_lin_modules=owner.target_lora_modules,
        )
        network.apply_to(None, model, False, True)
    network._update_torch_multiplier()
    return network


def inputs(dtype=torch.float32):
    return dict(
        hidden_states=torch.randn(1, 4, 2, 2, dtype=dtype),
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=torch.randn(1, 3, 16, dtype=dtype),
    )


def qkv_adapters(network):
    return {
        p: next(m for m in network.unet_loras if m.lora_name.endswith("$$proj$$" + p))
        for p in ("q", "k", "v")
    }


class MossSplitQKVTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)

    def assert_state_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertEqual(actual[key].dtype, expected[key].dtype)
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_configuration_defaults_and_validation(self):
        default = MossImageModel(
            "cpu", ModelConfig(name_or_path="unused", arch="moss_image")
        )
        self.assertFalse(default.split_qkv)
        self.assertEqual(default.target_lora_modules, ["MossImageTransformer"])
        self.assertTrue(holder(True).split_qkv)
        self.assertEqual(
            holder(True).target_lora_modules, ["MossImageSplitQKVTransformer"]
        )
        for invalid in ("false", "true", 0, 1, None):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "must be a boolean"),
            ):
                MossImageModel(
                    "cpu",
                    ModelConfig(
                        name_or_path="unused",
                        arch="moss_image",
                        model_kwargs={"split_qkv": invalid},
                    ),
                )

    def test_holder_loads_both_file_and_directory_with_selected_layout(self):
        source = tiny_model().state_dict()
        with tempfile.TemporaryDirectory(prefix="moss-split-") as directory:
            path = Path(directory) / "base.safetensors"
            save_file(source, str(path))
            for split in (False, True):
                for name in (str(path), directory):
                    with self.subTest(split=split, name=name):
                        owner = holder(split)
                        owner.model_config.name_or_path = name
                        model = owner._load_transformer()
                        self.assertIs(
                            type(model),
                            MossImageSplitQKVTransformer
                            if split
                            else MossImageTransformer,
                        )
                        self.assertEqual("qkv" in model.layers[0].proj, not split)

    def test_frozen_base_and_gradients_match_for_both_layouts(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                fused = tiny_model(dtype)
                split = MossImageSplitQKVTransformer.load_from_state_dict(
                    fused.state_dict(), dtype
                )
                data = inputs(dtype)
                x1 = data["hidden_states"].clone().requires_grad_()
                x2 = data["hidden_states"].clone().requires_grad_()
                y1 = fused(**{**data, "hidden_states": x1})
                y2 = split(**{**data, "hidden_states": x2})
                tolerance = (
                    dict(rtol=2e-5, atol=2e-6)
                    if dtype == torch.float32
                    else dict(rtol=0.04, atol=0.004)
                )
                torch.testing.assert_close(y1, y2, **tolerance)
                y1.float().square().mean().backward()
                y2.float().square().mean().backward()
                torch.testing.assert_close(x1.grad, x2.grad, **tolerance)
                for p, rows in zip(
                    ("q", "k", "v"), fused.layers[0].proj["qkv"].weight.grad.chunk(3)
                ):
                    torch.testing.assert_close(
                        split.layers[0].proj[p].weight.grad, rows, **tolerance
                    )
                # The small text refiner retains its original fused projections.
                self.assertTrue(hasattr(split.text_embedder.blocks[0], "qkv"))

    def test_native_split_base_and_full_model_export(self):
        fused = tiny_model()
        split = MossImageSplitQKVTransformer.load_from_state_dict(
            fused.state_dict(), torch.float32
        )
        state = split.state_dict()
        self.assertNotIn("layers.0.proj.qkv.weight", state)
        loaded = MossImageSplitQKVTransformer.load_from_state_dict(state, torch.float32)
        self.assert_state_equal(loaded.state_dict(), state)
        exported = split.convert_state_dict_on_save(state)
        self.assertIn("layers.0.proj.qkv.weight", exported)
        loaded_fused = MossImageTransformer.load_from_state_dict(
            exported, torch.float32
        )
        self.assert_state_equal(loaded_fused.state_dict(), fused.state_dict())
        self.assertNotIn("layers.0.proj.qkv.weight", state)

    def test_quantized_fused_base_splits_without_requantization(self):
        prefix = "layers.0.proj.qkv"
        for fmt in ("int8", "float8"):
            with self.subTest(fmt=fmt):
                state = dict(tiny_model().state_dict())
                if fmt == "int8":
                    weight = torch.randint(-32, 32, (48, 16), dtype=torch.int8)
                    scale = torch.linspace(0.01, 0.03, 48)
                    conf = {
                        "format": "int8_tensorwise",
                        "convrot": True,
                        "convrot_groupsize": 4,
                    }
                else:
                    weight = torch.randn(48, 16).to(torch.float8_e4m3fn)
                    scale = torch.tensor(0.1)
                    conf = {"format": "float8_e4m3fn"}
                state[prefix + ".weight"] = weight
                state[prefix + ".weight_scale"] = scale
                state[prefix + ".comfy_quant"] = torch.tensor(
                    list(json.dumps(conf).encode()), dtype=torch.uint8
                )
                fused = MossImageTransformer.load_from_state_dict(
                    dict(state), torch.float32
                )
                split = MossImageSplitQKVTransformer.load_from_state_dict(
                    dict(state), torch.float32
                )
                for i, p in enumerate(("q", "k", "v")):
                    part = split.layers[0].proj[p]
                    self.assertTrue(part.is_ostris_quantized)
                    torch.testing.assert_close(
                        part.weight,
                        fused.layers[0].proj["qkv"].weight[i * 16 : (i + 1) * 16],
                        rtol=0,
                        atol=0,
                    )
                    if fmt == "int8":
                        torch.testing.assert_close(
                            part.cr8_qdata,
                            weight[i * 16 : (i + 1) * 16],
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            part.cr8_scales.view(torch.float32),
                            scale[i * 16 : (i + 1) * 16],
                            rtol=0,
                            atol=0,
                        )
                owner = holder(True)
                network = attach(split, owner)
                network.train()
                x = torch.randn(1, 3, 16, requires_grad=True)
                with network:
                    loss = sum(
                        split.layers[0].proj[p](x).square().mean()
                        for p in ("q", "k", "v")
                    )
                loss.backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for adapter in qkv_adapters(network).values():
                    self.assertGreater(adapter.lokr_w2.grad.abs().sum().item(), 0)

    def test_legacy_scaled_fp8_fused_base_splits_exactly(self):
        state = dict(tiny_model().state_dict())
        prefix = "layers.0.proj.qkv"
        weight = torch.randn(48, 16).to(torch.float8_e4m3fn).view(torch.uint8)
        scale = torch.tensor(0.25)
        state["scaled_fp8"] = torch.tensor(1)
        state[prefix + ".weight"] = weight
        state[prefix + ".scale_weight"] = scale
        state[prefix + ".scale_input"] = torch.tensor(0.125)

        split = MossImageSplitQKVTransformer.load_from_state_dict(state, torch.float32)

        for i, name in enumerate(("q", "k", "v")):
            projection = split.layers[0].proj[name]
            rows = weight[i * 16 : (i + 1) * 16]
            torch.testing.assert_close(projection.f8_qdata.view(torch.uint8), rows)
            torch.testing.assert_close(
                projection.f8_scale.view(torch.float32)[0], scale, rtol=0, atol=0
            )
            torch.testing.assert_close(
                projection.weight,
                rows.view(torch.float8_e4m3fn).float() * scale,
                rtol=0,
                atol=0,
            )

    def test_independent_training_checkpointing_and_resume(self):
        for full_rank in (True, False):
            with self.subTest(full_rank=full_rank):
                base = {k: v.clone() for k, v in tiny_model().state_dict().items()}
                model = MossImageSplitQKVTransformer.load_from_state_dict(
                    base, torch.float32
                )
                owner = holder(True)
                network = attach(model, owner, full_rank)
                adapters = qkv_adapters(network)
                self.assertEqual(
                    len({next(a.parameters()).data_ptr() for a in adapters.values()}), 3
                )
                model.enable_gradient_checkpointing()
                model.train()
                network.train()

                # Full-rank LoKr uses an integer alpha sentinel (9999999999).
                # Match the trainer's save dtype before comparing restored
                # buffers; float32 cannot represent that integer exactly.
                def saved_state():
                    return {
                        k: (v.float().to(v.dtype) if k.endswith(".alpha") else v)
                        for k, v in network.state_dict().items()
                    }

                optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
                data = inputs()
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with network:
                        loss = model(**data).square().mean()
                        loss.backward()
                    for adapter in adapters.values():
                        grads = [p.grad for p in adapter.parameters()]
                        self.assertTrue(
                            all(
                                g is not None and torch.isfinite(g).all() for g in grads
                            )
                        )
                        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
                    optimizer.step()
                for parameter in model.parameters():
                    self.assertIsNone(parameter.grad)
                with tempfile.TemporaryDirectory(prefix="moss-split-") as directory:
                    path = str(Path(directory) / "adapter.safetensors")
                    network.save_weights(path, dtype=torch.float32)
                    state = load_file(path)
                    for p in ("q", "k", "v"):
                        self.assertIn(f"diffusion_model.layers.0.proj.{p}.alpha", state)
                        self.assertTrue(
                            any(
                                k.startswith(f"diffusion_model.layers.0.proj.{p}.lokr_")
                                for k in state
                            )
                        )
                    self.assertFalse(any(".proj.qkv." in k for k in state))
                    resumed = MossImageSplitQKVTransformer.load_from_state_dict(
                        base, torch.float32
                    )
                    resumed_owner = holder(True)
                    resumed_net = attach(resumed, resumed_owner, full_rank)
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertIsNone(resumed_net.load_weights(path))
                    self.assertNotIn("forced keymap", output.getvalue())
                    self.assert_state_equal(resumed_net.state_dict(), saved_state())
                    resumed_optimizer = torch.optim.AdamW(
                        resumed_net.parameters(), lr=0.01
                    )
                    resumed_optimizer.load_state_dict(
                        copy.deepcopy(optimizer.state_dict())
                    )
                    # One further step must agree with uninterrupted adapter training.
                    for current_model, current_net, current_opt in (
                        (model, network, optimizer),
                        (resumed, resumed_net, resumed_optimizer),
                    ):
                        current_opt.zero_grad(set_to_none=True)
                        with current_net:
                            current_model(**data).square().mean().backward()
                        current_opt.step()
                    self.assert_state_equal(resumed_net.state_dict(), saved_state())

    def test_comfyui_export_applies_independent_updates_to_fused_rows(self):
        for full_rank in (True, False):
            with self.subTest(full_rank=full_rank):
                base = tiny_model()
                split = MossImageSplitQKVTransformer.load_from_state_dict(
                    {k: v.clone() for k, v in base.state_dict().items()}, torch.float32
                )
                owner = holder(True)
                network = attach(split, owner, full_rank)
                with torch.no_grad():
                    for adapter in qkv_adapters(network).values():
                        for parameter in adapter.parameters():
                            parameter.normal_(std=0.1)
                exported = network.get_state_dict(dtype=torch.float32)
                deltas = []
                for p in ("q", "k", "v"):
                    prefix = f"diffusion_model.layers.0.proj.{p}"
                    a = exported[prefix + ".lokr_w1"]
                    if full_rank:
                        b = exported[prefix + ".lokr_w2"]
                        scale = 1.0
                    else:
                        b1, b2 = (
                            exported[prefix + ".lokr_w2_a"],
                            exported[prefix + ".lokr_w2_b"],
                        )
                        b = b1 @ b2
                        scale = exported[prefix + ".alpha"] / b2.shape[0]
                    deltas.append(torch.kron(a, b) * scale)
                # ComfyUI maps .q/.k/.v to successive row thirds of the native
                # .qkv weight and applies each reconstructed LoKr delta there.
                with torch.no_grad():
                    base.layers[0].proj["qkv"].weight.add_(torch.cat(deltas))
                data = inputs()
                with network:
                    actual = split(**data)
                torch.testing.assert_close(actual, base(**data), rtol=2e-5, atol=2e-6)

    def test_wrong_lokr_layout_is_rejected_before_network_loading(self):
        for split, projection in ((True, "qkv"), (False, "q")):
            with self.subTest(split=split):
                owner = holder(split)
                state = {
                    f"diffusion_model.layers.0.proj.{projection}.lokr_w1": torch.ones(
                        4, 4
                    )
                }
                with self.assertRaisesRegex(ValueError, "checkpoint QKV layout"):
                    owner.convert_lora_weights_before_load(state)
                self.assertEqual(len(state), 1)

    def test_standard_lora_split_keys_pass_through_in_split_mode(self):
        owner = holder(True)
        state = {
            f"transformer.layers.0.proj.{p}.{suffix}": torch.randn(2, 2)
            for p in ("q", "k", "v")
            for suffix in ("lora_A.weight", "lora_B.weight")
        }
        self.assert_state_equal(
            owner.convert_lora_weights_before_load(
                owner.convert_lora_weights_before_save(state)
            ),
            state,
        )

    def test_bad_fused_and_mixed_base_layouts_are_rejected(self):
        state = dict(tiny_model().state_dict())
        state["layers.0.proj.q.weight"] = torch.ones(16, 16)
        with self.assertRaisesRegex(ValueError, "Mixed fused and split"):
            MossImageSplitQKVTransformer.convert_state_dict_on_load(state)
        state = {"layers.0.proj.qkv.weight": torch.ones(47, 16)}
        with self.assertRaisesRegex(ValueError, "does not divide"):
            MossImageSplitQKVTransformer.convert_state_dict_on_load(state)


if __name__ == "__main__":
    unittest.main()
