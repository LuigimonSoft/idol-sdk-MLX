"""Verify the package can be imported without loading the reference backend."""

import subprocess
import sys
import unittest


class MlxPackageTests(unittest.TestCase):
    def test_import_does_not_load_pytorch(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys, idol_mlx; assert 'torch' not in sys.modules"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
