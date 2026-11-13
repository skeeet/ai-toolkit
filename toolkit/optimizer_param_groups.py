import re
from typing import Iterable, List, Optional

from torch import nn


# Used only when the model has no explicit block-name declaration. In particular,
# numeric children such as attn.to_out.0 and ff.net.0 are not transformer blocks.
_BLOCK_COLLECTIONS = (
    'transformer_blocks', 'single_transformer_blocks', 'joint_transformer_blocks',
    'blocks', 'layers', 'single_blocks', 'double_blocks', 'joint_blocks',
    'single_stream_blocks', 'double_stream_blocks', 'noise_refiner', 'context_refiner',
)


def group_network_params_by_transformer_block(
    param_groups: List[dict],
    denoiser: nn.Module,
    denoiser_loras: Iterable[nn.Module],
    block_names: Optional[List[str]] = None,
) -> List[dict]:
    """Split existing adapter parameter groups without changing their options.

    Resolve blocks from original module paths, not serialized LoRA names: those
    names may replace dots with underscores or $$ and lose path boundaries.
    Text-encoder adapters and non-block parameters retain their original grouping.
    """
    collections = _BLOCK_COLLECTIONS if block_names is None else block_names
    if not collections:
        raise ValueError("transformer_block optimizer grouping found no block collections")
    pattern = re.compile(
        r'(?:^|\.)(?:' + '|'.join(re.escape(name) for name in collections)
        + r')\.\d+(?=\.|$)'
    )
    module_blocks = {}
    for path, module in denoiser.named_modules():
        match = pattern.search(path)
        if match is not None:
            # Include the full prefix so separate stacks and nested U-Net
            # transformers with the same block index remain separate groups.
            module_blocks[id(module)] = path[:match.end()]

    parameter_blocks = {}
    for lora in denoiser_loras:
        original_ref = getattr(lora, 'orig_module_ref', None)
        if original_ref is not None:
            original = original_ref()
        else:
            original = lora.org_module[0]
        block = module_blocks.get(id(original))
        if block is not None:
            for parameter in lora.parameters():
                parameter_id = id(parameter)
                if parameter_id in parameter_blocks and parameter_blocks[parameter_id] != block:
                    raise ValueError("An adapter parameter is shared across transformer blocks")
                parameter_blocks[parameter_id] = block

    result = []
    matched = False
    for group in param_groups:
        partitions = {}
        for parameter in group['params']:
            block = parameter_blocks.get(id(parameter))
            matched = matched or block is not None
            partitions.setdefault(block, []).append(parameter)
        for parameters in partitions.values():
            result.append({**group, 'params': parameters, 'optimizer_grouping': 'transformer_block'})

    if not matched:
        raise ValueError(
            "transformer_block optimizer grouping found no trainable transformer-block "
            "adapter parameters. Check the model's get_transformer_block_names() "
            "or use network.optimizer_grouping: default."
        )
    return result
