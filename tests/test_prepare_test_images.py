import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts.prepare_test_images import checked_photo, prepare, synthetic_images


class FixtureTests(unittest.TestCase):
    def test_generated_images_have_distinct_alpha_contracts(self):
        images = synthetic_images()
        self.assertEqual(images["synthetic_opaque.png"].mode, "RGB")
        alpha = images["synthetic_alpha.png"].getchannel("A")
        self.assertEqual(alpha.getextrema(), (0, 255))
        self.assertEqual(images["synthetic_alpha.png"].size, (256, 256))

    def test_photo_rejects_digest_mismatch(self):
        with patch("scripts.prepare_test_images.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = b"unexpected"
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                checked_photo("photo.png", hashlib.sha256(b"expected").hexdigest())

    def test_dangling_fixture_symlink_is_rejected_before_downloading(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "images"
            output.mkdir()
            link = output / "synthetic_alpha.png"
            link.symlink_to(output / "missing.png")
            with patch("scripts.prepare_test_images.checked_photo") as fetch:
                with self.assertRaises(FileExistsError):
                    prepare(output)
                fetch.assert_not_called()
            self.assertTrue(link.is_symlink())
            self.assertEqual([item.name for item in output.iterdir()], [link.name])

    def test_prepare_writes_four_images_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "images"
            # Small PNG payloads simulate verified downloads; network is never used.
            fixture = synthetic_images()["synthetic_opaque.png"]
            from io import BytesIO
            buffer = BytesIO()
            fixture.save(buffer, format="PNG")
            with patch("scripts.prepare_test_images.checked_photo", return_value=buffer.getvalue()):
                prepare(output)
                self.assertEqual(len(list(output.iterdir())), 4)
                with Image.open(output / "synthetic_alpha.png") as image:
                    self.assertEqual(image.mode, "RGBA")
                with self.assertRaises(FileExistsError):
                    prepare(output)


if __name__ == "__main__":
    unittest.main()
