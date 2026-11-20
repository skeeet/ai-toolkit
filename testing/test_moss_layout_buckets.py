"""CPU-only regression tests for bucket-aware MOSS captions."""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image

from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import DatasetConfig, ModelConfig
from toolkit.data_loader import get_dataloader_from_datasets


def layout(*boxes):
    return {
        "summary": {"caption": "A person beside a window.", "meta": ["photograph"]},
        "items": [
            {"label": "object", "caption": f"Object {index}.", "bbox": box, "tags": ["detail"]}
            for index, box in enumerate(boxes)
        ],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


class MossLayoutBucketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temporary = TemporaryDirectory(prefix="moss-layout-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = MossImageModel(
            "cpu",
            ModelConfig(arch="moss_image", name_or_path="data-loader-test"),
            dtype="float32",
        )
        self.model.model = self.model.vae = None

    def fixture(self, size, document):
        Image.new("RGB", size).save(self.root / "sample.png")
        prompt = document if isinstance(document, str) else json.dumps(document, indent=2) + "\n"
        (self.root / "sample.json").write_text(prompt, encoding="utf-8")
        return prompt

    def loader(self, **overrides):
        config = {
            "folder_path": str(self.root),
            "caption_ext": ".json",
            "resolution": 512,
            "buckets": True,
            "num_workers": 0,
            "moss_layout_bbox_transform": True,
        }
        config.update(overrides)
        return get_dataloader_from_datasets([config], batch_size=1, sd=self.model)

    def batch(self, **overrides):
        batch, = self.loader(**overrides)
        return batch

    def test_512_scales_odd_edges_outward_and_preserves_other_fields(self):
        original = layout([429, -111, 623, 117])
        raw = self.fixture((832, 1248), original)
        batch = self.batch()
        item, = batch.file_items
        self.assertEqual(tuple(batch.tensor.shape), (1, 3, 624, 416))
        self.assertEqual(item.raw_caption, raw)
        expected = layout([214, -56, 312, 59])
        self.assertEqual(json.loads(batch.get_caption_list()[0]), expected)

    def test_1024_native_bucket_preserves_original_bytes(self):
        raw = self.fixture((832, 1248), layout([429, -111, 623, 117]))
        batch = self.batch(resolution=1024)
        self.assertEqual(tuple(batch.tensor.shape), (1, 3, 1248, 832))
        self.assertEqual(batch.get_caption_list(), [raw])

    def test_multiple_resolutions_use_one_source_without_double_scaling(self):
        raw = self.fixture((1024, 1024), layout([-101, -203, 307, 409]))
        by_width = {batch.tensor.shape[-1]: batch for batch in self.loader(resolution=[512, 1024])}
        self.assertEqual(set(by_width), {512, 1024})
        self.assertEqual(by_width[1024].get_caption_list(), [raw])
        for _ in range(2):
            item, = by_width[512].file_items
            item.load_caption()
            self.assertEqual(json.loads(item.caption)["items"][0]["bbox"], [-51, -102, 154, 205])
            self.assertEqual(item.raw_caption, raw)

    def test_rounded_bucket_crop_applies_distinct_realized_axis_scales(self):
        self.fixture((832, 1248), layout([429, -110, 624, 118]))
        batch = self.batch(resolution=768)
        item, = batch.file_items
        self.assertEqual((item.scale_to_width, item.scale_to_height), (630, 944))
        self.assertEqual((item.crop_x, item.crop_y, item.crop_width, item.crop_height), (3, 0, 624, 944))
        self.assertEqual(json.loads(item.caption)["items"][0]["bbox"], [324, -84, 472, 90])

    def test_square_crop_clips_partial_items_and_drops_invisible_items(self):
        self.fixture((128, 64), layout([-24, -55, 16, -15], [-20, -60, 0, -41]))
        expected = layout([-24, -32, 16, -15])
        self.assertEqual(json.loads(self.batch(resolution=64, square_crop=True).get_caption_list()[0]), expected)

    def test_full_crop_can_fall_back_to_scene_caption(self):
        self.fixture((128, 64), layout([-20, -60, 0, -41]))
        self.assertEqual(self.batch(resolution=64, square_crop=True).get_caption_list(), ["A person beside a window."])

    def test_flips_follow_the_pixels_before_resize_and_crop(self):
        self.fixture((64, 64), layout([-20, -30, 12, 5]))
        actual = {}
        for batch in self.loader(resolution=64, flip_x=True, flip_y=True):
            item, = batch.file_items
            actual[item.flip_x, item.flip_y] = json.loads(item.caption)["items"][0]["bbox"]
        self.assertEqual(actual, {
            (False, False): [-20, -30, 12, 5],
            (True, False): [-20, -5, 12, 30],
            (False, True): [-12, -30, 20, 5],
            (True, True): [-12, -5, 20, 30],
        })

    def test_toggle_defaults_off(self):
        raw = self.fixture((832, 1248), layout([429, -111, 623, 117]))
        self.assertEqual(self.batch(moss_layout_bbox_transform=False).get_caption_list(), [raw])
        self.assertFalse(DatasetConfig().moss_layout_bbox_transform)

    def test_non_moss_architecture_leaves_caption_alone(self):
        raw = self.fixture((832, 1248), layout([429, -111, 623, 117]))
        self.model.arch = "ideogram4"
        self.assertEqual(self.batch().get_caption_list(), [raw])

    def test_ideogram_normalized_json_is_passed_through_even_when_enabled(self):
        raw = self.fixture((832, 1248), {"compositional_deconstruction": {
            "elements": [{"obj": {"bbox": [0.1, 0.2, 0.7, 0.8], "desc": "Chair"}}],
        }})
        self.assertEqual(self.batch().get_caption_list(), [raw])

    def test_normalized_moss_style_and_plain_captions_are_passed_through(self):
        normalized = layout([0.1, 0.2, 0.7, 0.8])
        normalized["bbox_coord_space"] = "normalized"
        for document in (normalized, "A seated person, soft daylight."):
            with self.subTest(document=document):
                raw = self.fixture((832, 1248), document)
                self.assertEqual(self.batch().get_caption_list(), [raw])

    def test_bucket_metadata_required_and_untracked_spatial_augmentations_rejected(self):
        self.fixture((832, 1248), layout([429, -111, 623, 117]))
        for options in ({"buckets": False}, {"augmentations": [{"method": "Rotate", "params": {"p": 1}}]}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "MOSS layout"):
                    self.loader(**options)

    def test_cached_fast_image_size_does_not_override_exif_oriented_geometry(self):
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (1248, 832)).save(self.root / "sample.jpg", exif=exif)
        document = layout([429, -111, 623, 117])
        (self.root / "sample.json").write_text(json.dumps(document), encoding="utf-8")
        self.loader(fast_image_size=True, moss_layout_bbox_transform=False)
        size_file = self.root / ".aitk_size.json"
        sizes = json.loads(size_file.read_text(encoding="utf-8"))
        key, = [key for key in sizes if key.endswith("sample.jpg")]
        sizes[key][:2] = [1248, 832]
        size_file.write_text(json.dumps(sizes), encoding="utf-8")
        batch = self.batch(fast_image_size=True)
        self.assertEqual(tuple(batch.tensor.shape), (1, 3, 624, 416))
        self.assertEqual(json.loads(batch.get_caption_list()[0]), layout([214, -56, 312, 59]))

    def test_random_crop_uses_the_sampled_offset(self):
        self.fixture((832, 1248), layout([429, -110, 624, 118]))
        with patch("toolkit.dataloader_mixins.random.randint", side_effect=lambda low, high: low):
            batch = self.batch(resolution=768, random_crop=True)
        item, = batch.file_items
        self.assertEqual((item.crop_x, item.crop_y), (0, 0))
        self.assertEqual(json.loads(item.caption)["items"][0]["bbox"], [324, -81, 472, 93])

    def test_shape_preserved_for_custom_caption_extension(self):
        raw = self.fixture((832, 1248), layout([429, -111, 623, 117]))
        (self.root / "sample.json").rename(self.root / "sample.txt")
        self.assertEqual(self.batch(caption_ext=".txt").get_caption_list(), [raw])

    def test_comma_based_prompt_transforms_rejected_for_opt_in_json(self):
        self.fixture((832, 1248), layout([429, -111, 623, 117]))
        for options in ({"shuffle_tokens": True}, {"token_dropout_rate": 0.1}, {"random_triggers": ["test"]}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "token shuffling, token dropout, and random triggers"):
                    self.loader(**options)

    def test_missing_trigger_stays_inside_layout_at_both_resolutions(self):
        raw = self.fixture((832, 1248), layout([429, -111, 623, 117]))
        for batch in self.loader(resolution=[512, 1024], trigger_word="nar1"):
            item, = batch.file_items
            self.assertEqual(item.raw_caption, raw)
            self.assertEqual(json.loads(item.caption)["summary"]["caption"], "nar1 A person beside a window.")
            self.assertEqual(self.model.inject_trigger_into_prompt(item.caption, "nar1", add_if_not_present=True), item.caption)

    def test_trigger_placeholders_apply_in_descriptions_but_preserve_literal_text(self):
        document = layout([429, -111, 623, 117])
        document["summary"]["caption"] = "A photo of [trigger]."
        document["items"][0].update(label="text", caption="Sign next to [name].", text="[name]")
        self.fixture((832, 1248), document)
        actual = json.loads(self.batch(trigger_word="nar1").get_caption_list()[0])
        self.assertEqual(actual["summary"]["caption"], "A photo of nar1.")
        self.assertEqual(actual["items"][0]["caption"], "Sign next to nar1.")
        self.assertEqual(actual["items"][0]["text"], "[name]")

    def test_training_config_rejects_prompt_saturation_or_split_triggers_for_moss_json(self):
        config = DatasetConfig(caption_ext="json", moss_layout_bbox_transform=True, trigger_word="person")
        with self.assertRaisesRegex(ValueError, "prompt_saturation_chance"):
            config.validate_moss_layout_training("moss_image", "person", 1.0)
        with self.assertRaisesRegex(ValueError, "trigger_word"):
            config.validate_moss_layout_training("moss_image", "another", 0)
        config.validate_moss_layout_training("moss_image", "person", 0)
        config.validate_moss_layout_training("ideogram4", "another", 1.0)

    def test_invalid_recognized_box_reports_sidecar(self):
        self.fixture((832, 1248), layout([10, 0, 5, 2]))
        with self.assertRaisesRegex(ValueError, r"sample\.json.*items\[0\]"):
            self.batch()

    def test_cache_encodes_and_keys_each_bucket_caption_before_loading_cached_batches(self):
        raw = self.fixture((832, 1248), layout([429, -111, 623, 117]))
        encoded = []

        def encode(prompts, **kwargs):
            prompts = [prompts] if isinstance(prompts, str) else prompts
            encoded.extend(prompts)
            return AdvancedPromptEmbeds(text_embeds=[torch.ones((2, 3)) for _ in prompts])

        with patch.object(self.model, "encode_prompt", side_effect=encode), patch.object(
            self.model, "set_device_state_preset"
        ):
            caches = self.loader(cache_text_embeddings=True, resolution=[512, 1024])
            by_width = {batch.tensor.shape[-1]: batch for batch in caches}
            key_512 = by_width[416].file_items[0].get_text_embedding_path()
            key_1024 = by_width[832].file_items[0].get_text_embedding_path()
            self.assertNotEqual(key_512, key_1024)
            self.assertTrue(os.path.isfile(key_512))
            self.assertTrue(os.path.isfile(key_1024))
            saved_captions = [by_width[w].get_caption_list()[0] for w in (416, 832)]
            self.assertCountEqual(encoded, saved_captions)
            self.assertEqual(saved_captions[1], raw)
            self.assertEqual(json.loads(saved_captions[0])["items"][0]["bbox"], [214, -56, 312, 59])
            second = list(self.loader(cache_text_embeddings=True, resolution=[512, 1024]))
            self.assertEqual(len(second), 2)
            self.assertEqual(len(encoded), 2)


if __name__ == "__main__":
    unittest.main()
