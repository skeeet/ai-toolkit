"""Map model-facing MOSS layout boxes through a still-image bucket."""

import json
import math


def _parse_layout(caption):
    if not caption.lstrip().startswith("{"):
        return None
    try:
        layout = json.loads(caption)
    except json.JSONDecodeError:
        return None
    if layout.get("bbox_coord_space") == "abs_centralized" and layout.get("bbox_format") == "yxyx":
        return layout
    return None


def _dump_layout(layout, original):
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":")) + ("\n" if original.endswith("\n") else "")


def rewrite_moss_layout_descriptions(caption, rewrite, trigger_to_add=None):
    """Apply prompt substitutions to descriptions and preserve literal rendered text."""
    layout = _parse_layout(caption)
    if layout is None:
        return None
    summary = layout.get("summary")
    items = layout.get("items")
    if not isinstance(items, list):
        raise ValueError("items must be an array")

    descriptions = [
        part for part in [summary, *items]
        if isinstance(part, dict) and isinstance(part.get("caption"), str)
    ]
    changed = False
    for part in descriptions:
        text = rewrite(part["caption"])
        if text != part["caption"]:
            part["caption"] = text
            changed = True

    if trigger_to_add and trigger_to_add.strip() and not any(
        trigger_to_add in part["caption"] for part in descriptions
    ):
        if not isinstance(summary, dict) or not isinstance(summary.get("caption"), str):
            raise ValueError("summary.caption is required to insert a training trigger")
        summary["caption"] = f"{trigger_to_add} {summary['caption']}"
        changed = True

    return _dump_layout(layout, caption) if changed else caption


def _map_axis(low, high, source, resized, offset, target, flip):
    low = max(0, min(source, low + source // 2))
    high = max(0, min(source, high + source // 2))
    if flip:
        low, high = source - high, source - low

    low = max(0, low * resized / source - offset)
    high = min(target, high * resized / source - offset)
    if low >= high:
        return None
    return math.floor(low) - target // 2, math.ceil(high) - target // 2


def transform_moss_layout_boxes(caption, source_size, resized_size, crop, *, flip_x=False, flip_y=False):
    """Transform centered yxyx boxes from oriented source pixels into bucket pixels."""
    if not flip_x and not flip_y and resized_size == source_size and crop == (0, 0, *source_size):
        return caption
    layout = _parse_layout(caption)
    if layout is None:
        return caption

    items = layout.get("items")
    if not isinstance(items, list):
        raise ValueError("items must be an array")

    source_width, source_height = source_size
    resized_width, resized_height = resized_size
    crop_x, crop_y, bucket_width, bucket_height = crop
    mapped_items = []
    for index, item in enumerate(items):
        box = item.get("bbox") if isinstance(item, dict) else None
        if not isinstance(box, list) or len(box) != 4 or not all(
            type(edge) is int or (type(edge) is float and math.isfinite(edge)) for edge in box
        ):
            raise ValueError(f"items[{index}].bbox must contain four finite yxyx coordinates")
        y_min, x_min, y_max, x_max = box
        if y_min >= y_max or x_min >= x_max:
            raise ValueError(f"items[{index}].bbox must have positive width and height")

        x = _map_axis(x_min, x_max, source_width, resized_width, crop_x, bucket_width, flip_x)
        y = _map_axis(y_min, y_max, source_height, resized_height, crop_y, bucket_height, flip_y)
        if x is not None and y is not None:
            mapped_items.append({**item, "bbox": [y[0], x[0], y[1], x[1]]})

    if mapped_items == items:
        return caption
    if not mapped_items:
        summary = layout.get("summary")
        summary_caption = summary.get("caption") if isinstance(summary, dict) else None
        if not isinstance(summary_caption, str) or not summary_caption.strip():
            raise ValueError("summary.caption is required when the crop removes all layout items")
        return summary_caption

    layout["items"] = mapped_items
    return _dump_layout(layout, caption)
