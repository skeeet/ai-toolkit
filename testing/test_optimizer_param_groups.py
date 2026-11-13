import copy
import unittest

import torch
from torch import nn

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.optimizer import get_optimizer
from toolkit.optimizer_param_groups import group_network_params_by_transformer_block


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.ModuleDict({'to_out': nn.Sequential(nn.Linear(8, 8))})
        self.ff = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        self.norm = nn.LayerNorm(8)


class TinyTransformer(nn.Module):
    def __init__(self, stacks):
        super().__init__()
        for stack in stacks:
            parent = self
            parts = stack.split('.')
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, nn.Module())
                parent = getattr(parent, part)
            parent.add_module(parts[-1], nn.ModuleList([TinyBlock(), TinyBlock()]))
        self.proj_out = nn.Linear(8, 8)


def make_network(stacks, network_type='lora', peft=False, all_layers=False):
    model = TinyTransformer(stacks)
    model.requires_grad_(False)
    config = NetworkConfig(type=network_type, linear=2, lokr_full_rank=False, all_layers=all_layers)
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=model,
        lora_dim=2,
        train_text_encoder=False,
        target_lin_modules=['TinyTransformer'],
        network_config=config,
        network_type=network_type,
        peft_format=peft,
    )
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    return model, network


def param_ids(parameters):
    return [id(parameter) for parameter in parameters]


class OptimizerParamGroupingTest(unittest.TestCase):
    def test_config_default_and_validation(self):
        self.assertEqual(NetworkConfig().optimizer_grouping, 'default')
        self.assertEqual(NetworkConfig(optimizer_grouping='default').optimizer_grouping, 'default')
        self.assertEqual(
            NetworkConfig(optimizer_grouping='transformer_block').optimizer_grouping,
            'transformer_block',
        )
        for value in ('block', '', None, ['default']):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'optimizer_grouping'):
                NetworkConfig(optimizer_grouping=value)

    def test_model_paths_and_adapter_formats(self):
        cases = (
            # MOSS, Qwen Image 2.1, Krea 2, and multiple/dotted block stacks.
            (['layers'], ['layers']),
            (['transformer_blocks'], ['transformer_blocks']),
            (['blocks'], ['blocks']),
            (['double_blocks', 'single_blocks'], None),
            (['model.language_model.layers'], ['model.language_model.layers']),
            (['left.transformer_blocks', 'right.transformer_blocks'], ['transformer_blocks']),
            (['custom.stages'], ['custom.stages']),
        )
        for stacks, declaration in cases:
            for network_type, peft in (('lora', False), ('lora', True), ('lokr', False), ('dora', True)):
                with self.subTest(stacks=stacks, network_type=network_type, peft=peft):
                    model, network = make_network(stacks, network_type, peft)
                    original = network.prepare_optimizer_params(1e-4, 2e-4, 3e-4)
                    self.assertEqual(len(original), 1)
                    original_ids = param_ids(original[0]['params'])
                    grouped = group_network_params_by_transformer_block(
                        original, model, network.unet_loras, declaration,
                    )
                    self.assertEqual(len(grouped), len(stacks) * 2 + 1)
                    actual_ids = [id(p) for group in grouped for p in group['params']]
                    self.assertCountEqual(actual_ids, original_ids)
                    self.assertEqual(len(actual_ids), len(set(actual_ids)))
                    self.assertTrue(all(group['lr'] == 2e-4 for group in grouped))
                    self.assertEqual(param_ids(original[0]['params']), original_ids)
                    # Three linear adapters per block. Internal ff.0/ff.1 and
                    # attn.to_out.0 must stay together; proj_out is the remainder.
                    sizes = sorted(len(group['params']) for group in grouped)
                    params_per_adapter = len(list(network.unet_loras[0].parameters()))
                    self.assertEqual(sizes, [params_per_adapter] + [3 * params_per_adapter] * (len(stacks) * 2))
                    base_ids = set(param_ids(model.parameters()))
                    self.assertTrue(base_ids.isdisjoint(actual_ids))

    def test_preserves_text_encoder_remainders_and_existing_group_options(self):
        model, network = make_network(['layers'], all_layers=True)
        network.prepare_optimizer_params(None, None, None)
        blocks = [[], []]
        remainder = []
        for lora in network.unet_loras:
            if '_layers_0_' in lora.lora_name:
                blocks[0].extend(lora.parameters())
            elif '_layers_1_' in lora.lora_name:
                blocks[1].extend(lora.parameters())
            else:
                remainder.extend(lora.parameters())
        text_params = list(nn.Linear(8, 8).parameters())
        original = [
            {'params': text_params, 'lr': 3e-4},
            {'params': blocks[0][:1] + blocks[1][:1], 'lr': 1e-4, 'weight_decay': 0.2},
            {'params': blocks[0][1:] + blocks[1][1:] + remainder, 'lr': 2e-4, 'weight_decay': 0.0},
        ]
        grouped = group_network_params_by_transformer_block(original, model, network.unet_loras, ['layers'])
        self.assertEqual(len(grouped), 6)
        expected = [text_params, blocks[0][:1], blocks[1][:1], blocks[0][1:], blocks[1][1:], remainder]
        for group, parameters in zip(grouped, expected):
            self.assertEqual(param_ids(group['params']), param_ids(parameters))
        self.assertEqual([group['lr'] for group in grouped], [3e-4, 1e-4, 1e-4, 2e-4, 2e-4, 2e-4])
        self.assertEqual([group.get('weight_decay') for group in grouped], [None, 0.2, 0.2, 0, 0, 0])
        # Each block includes full-weight LayerNorm adapters as well as LoRA.
        self.assertEqual(len(blocks[0]), 8)
        self.assertEqual(len(blocks[1]), 8)

    def test_declared_collections_do_not_pick_up_other_stacks(self):
        model, network = make_network(['layers', 'text_embedder.blocks'])
        original = network.prepare_optimizer_params(None, None, None)
        grouped = group_network_params_by_transformer_block(original, model, network.unet_loras, ['layers'])
        self.assertEqual(len(grouped), 3)
        remainder = [p for lora in network.unet_loras if '_layers_' not in lora.lora_name for p in lora.parameters()]
        self.assertEqual(param_ids(grouped[-1]['params']), param_ids(remainder))

    def test_no_matching_trainable_blocks_is_an_error(self):
        model, network = make_network(['layers'])
        original = network.prepare_optimizer_params(None, None, None)
        for declaration in ([], ['not_layers'], ['layers_extra']):
            with self.subTest(declaration=declaration), self.assertRaisesRegex(ValueError, 'transformer_block'):
                group_network_params_by_transformer_block(original, model, network.unet_loras, declaration)
        outside = [p for lora in network.unet_loras if 'proj_out' in lora.lora_name for p in lora.parameters()]
        with self.assertRaisesRegex(ValueError, 'no trainable'):
            group_network_params_by_transformer_block([{'params': outside}], model, network.unet_loras, ['layers'])

    def test_actual_transformer_constructors(self):
        from extensions_built_in.diffusion_models.moss_image.src.transformer import MossImageTransformer, MossImageParams
        from extensions_built_in.diffusion_models.krea2.src.mmdit import SingleStreamDiT, SingleMMDiTConfig
        from extensions_built_in.diffusion_models.qwen_image_2.src.transformer import QwenImage21Transformer2DModel

        # Small CPU models exercise the real module structures without loading
        # pretrained weights or allocating production-sized transformers.
        constructors = (
            (lambda: MossImageTransformer(MossImageParams(
                dim=32, ffn_dim=64, num_heads=2, num_layers=2,
                text_dim=16, freq_dim=16, latent_channels=8,
            )), ['layers']),
            (lambda: SingleStreamDiT(SingleMMDiTConfig(
                features=32, tdim=16, txtdim=32, heads=2, multiplier=2,
                layers=2, patch=1, channels=8, txtheads=2, txtkvheads=2,
            )), ['blocks']),
            (lambda: QwenImage21Transformer2DModel(
                in_channels=8, out_channels=8, num_layers=2,
                attention_head_dim=16, num_attention_heads=2,
                context_in_dim=16, axes_dims_rope=(4, 6, 6),
            ), ['transformer_blocks']),
        )
        for constructor, block_names in constructors:
            class ModelSpec:
                use_old_lokr_format = False

                def get_transformer_block_names(self):
                    return block_names

            base_model = ModelSpec()
            for adapter_type in ('lora', 'lokr'):
                model = constructor()
                with self.subTest(model=type(model).__name__, adapter_type=adapter_type):
                    model.requires_grad_(False)
                    network = LoRASpecialNetwork(
                        text_encoder=None, unet=model, train_text_encoder=False,
                        lora_dim=2, target_lin_modules=[type(model).__name__],
                        network_type=adapter_type, is_transformer=True,
                        network_config=NetworkConfig(type=adapter_type, lokr_full_rank=False),
                        transformer_only=True, base_model=base_model,
                    )
                    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
                    original = network.prepare_optimizer_params(None, 1e-6, None)
                    grouped = group_network_params_by_transformer_block(
                        original, model, network.unet_loras, block_names,
                    )
                    self.assertEqual(len(original), 1)
                    before = [id(p) for group in original for p in group['params']]
                    after = [id(p) for group in grouped for p in group['params']]
                    self.assertCountEqual(after, before)
                    self.assertEqual(len(after), len(set(after)))
                    expected = {}
                    for lora in network.unet_loras:
                        original_module = lora.org_module[0]
                        for index, block in enumerate(getattr(model, block_names[0])):
                            if any(original_module is child for child in block.modules()):
                                expected.setdefault(index, []).extend(lora.parameters())
                    actual = [set(param_ids(group['params'])) for group in grouped]
                    expected_groups = [set(param_ids(expected[index])) for index in range(2)]
                    # Krea's current substring-based transformer_only targeting
                    # also includes text-fusion adapters outside its main blocks.
                    remainder = set(before) - set.union(*expected_groups)
                    if remainder:
                        expected_groups.append(remainder)
                    self.assertEqual(len(actual), len(expected_groups))
                    for expected_group in expected_groups:
                        self.assertIn(expected_group, actual)

    def test_z_image_blocks_and_refiner_remainders(self):
        from toolkit.models.v2.diffusion_models.z_image import ZImageTransformer2DModel

        class ModelSpec:
            use_old_lokr_format = False

            def get_transformer_block_names(self):
                return ZImageTransformer2DModel.get_transformer_block_names()

        base_model = ModelSpec()
        for adapter_type in ('lora', 'lokr', 'dora'):
            for transformer_only in (True, False):
                with self.subTest(adapter_type=adapter_type, transformer_only=transformer_only):
                    model = ZImageTransformer2DModel(
                        in_channels=4, dim=32, n_layers=2, n_refiner_layers=2,
                        n_heads=2, n_kv_heads=2, cap_feat_dim=16,
                        axes_dims=[4, 6, 6], axes_lens=[16, 16, 16],
                    )
                    model.requires_grad_(False)
                    network = LoRASpecialNetwork(
                        text_encoder=None, unet=model, train_text_encoder=False,
                        lora_dim=2, target_lin_modules=[type(model).__name__],
                        network_type=adapter_type, is_transformer=True,
                        network_config=NetworkConfig(type=adapter_type, lokr_full_rank=False),
                        transformer_only=transformer_only, base_model=base_model,
                    )
                    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
                    original = network.prepare_optimizer_params(None, 1e-6, None)
                    self.assertEqual(len(original), 1)
                    before = param_ids(original[0]['params'])
                    for declaration in (base_model.get_transformer_block_names(), None):
                        grouped = group_network_params_by_transformer_block(
                            original, model, network.unet_loras, declaration,
                        )
                        actual = [set(param_ids(group['params'])) for group in grouped]
                        collections = declaration or ['layers', 'noise_refiner', 'context_refiner']
                        expected = []
                        for collection in collections:
                            for block in getattr(model, collection):
                                module_ids = {id(module) for module in block.modules()}
                                parameters = {
                                    id(parameter) for lora in network.unet_loras
                                    if id(lora.org_module[0]) in module_ids
                                    for parameter in lora.parameters()
                                }
                                if parameters:
                                    expected.append(parameters)
                        remainder = set(before) - set.union(*expected)
                        if remainder:
                            expected.append(remainder)
                        self.assertEqual(len(actual), len(expected))
                        for partition in expected:
                            self.assertIn(partition, actual)
                        after = [id(p) for group in grouped for p in group['params']]
                        self.assertCountEqual(after, before)
                        self.assertEqual(len(after), len(set(after)))

    def test_automagic3_independent_rates_and_resume(self):
        for fused in (False, True):
            with self.subTest(fused=fused):
                model, network = make_network(['layers'])
                original = network.prepare_optimizer_params(None, 1e-6, None)
                grouped = group_network_params_by_transformer_block(original, model, network.unet_loras, ['layers'])
                optimizer = get_optimizer(grouped, 'automagic3', optimizer_params={'fused': fused, 'polarity_history': 2})
                initial = optimizer.get_learning_rates()
                for step in range(4):
                    # One block sees a persistent direction, the next an
                    # alternating direction; the remainder has no gradient.
                    terms = []
                    for index, group in enumerate(optimizer.param_groups[:2]):
                        sign = -1 if index == 1 and step % 2 else 1
                        terms.extend(sign * parameter.sum() for parameter in group['params'])
                    sum(terms).backward()
                    optimizer.step()
                    optimizer.zero_grad()
                rates = optimizer.get_learning_rates()
                self.assertGreater(rates[0], initial[0])
                self.assertLess(rates[1], initial[1])
                self.assertEqual(rates[2], initial[2])
                for group in optimizer.param_groups[:2]:
                    mirrors = [optimizer.state[p]['lr'].item() for p in group['params']]
                    self.assertTrue(all(value == mirrors[0] for value in mirrors))

                state = copy.deepcopy(optimizer.state_dict())
                restored_model, restored_network = make_network(['layers'])
                restored = group_network_params_by_transformer_block(
                    restored_network.prepare_optimizer_params(None, 1e-6, None),
                    restored_model, restored_network.unet_loras, ['layers'],
                )
                restored_optimizer = get_optimizer(restored, 'automagic3', optimizer_params={'fused': fused, 'polarity_history': 2})
                restored_optimizer.load_state_dict(state)
                # Automagic3 restores each group's geometric median via fp32
                # log/exp, which introduces a small relative rounding error.
                torch.testing.assert_close(
                    torch.tensor(restored_optimizer.get_learning_rates()),
                    torch.tensor(rates), rtol=1e-6, atol=0,
                )


if __name__ == '__main__':
    unittest.main()
