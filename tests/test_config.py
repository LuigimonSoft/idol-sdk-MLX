import json
import tempfile
import unittest
from pathlib import Path

from idol_mlx.config import ConfigError, load_config


def valid_config():
    model = lambda: {"name": "example", "args": {"width": 16}}
    return {"denoiser": model(), "decoder": model(), "decoder_gs": model(),
            "image_size": 1024, "max_voxel_level": 8,
            "image_encoders": {"dinov3": "example/dino", "flux2_vae": "example/vae"}}


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "config.json"

    def write(self, value):
        self.path.write_text(json.dumps(value), encoding="utf-8")

    def test_valid_config_preserves_extra_fields(self):
        value = valid_config()
        value["extra"] = {"a": 1}
        self.write(value)
        self.assertEqual(load_config(self.path), value)

    def test_missing_and_invalid_json_are_reported(self):
        with self.assertRaises(ConfigError):
            load_config(self.path)
        self.path.write_text("{", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(self.path)

    def test_required_structure_is_validated(self):
        for key, value in [("denoiser", None), ("decoder", {"name": "", "args": {}}),
                           ("decoder_gs", {"name": "ok", "args": []}), ("image_size", True),
                           ("max_voxel_level", 0), ("image_encoders", {"dinov3": "x"})]:
            config = valid_config()
            config[key] = value
            self.write(config)
            with self.subTest(key=key), self.assertRaises(ConfigError):
                load_config(self.path)


if __name__ == "__main__":
    unittest.main()
