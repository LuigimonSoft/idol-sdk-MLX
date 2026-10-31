import unittest
from unittest.mock import MagicMock, patch

import idol_mlx.device as device
from idol_mlx.device import MetalUnavailableError, require_metal


class RequireMetalTests(unittest.TestCase):
    def setUp(self):
        self.platform = patch("idol_mlx.device.sys.platform", "darwin")
        self.machine = patch("idol_mlx.device.platform.machine", return_value="arm64")
        self.loader = patch("idol_mlx.device.importlib.import_module")
        self.platform.start()
        self.machine_mock = self.machine.start()
        self.load = self.loader.start()
        self.addCleanup(self.platform.stop)
        self.addCleanup(self.machine.stop)
        self.addCleanup(self.loader.stop)
        self.mx = MagicMock()
        self.mx.metal.is_available.return_value = True
        self.load.return_value = self.mx

    def test_success_executes_and_evaluates_on_gpu(self):
        require_metal()
        self.mx.add.assert_called_once_with(self.mx.array.return_value, self.mx.array.return_value,
                                           stream=self.mx.gpu)
        self.mx.eval.assert_called_once_with(self.mx.add.return_value)

    def test_unsupported_operating_system(self):
        with patch.object(device.sys, "platform", "linux"):
            with self.assertRaisesRegex(MetalUnavailableError, "Apple Silicon macOS"):
                require_metal()
        self.load.assert_not_called()

    def test_intel_mac_is_rejected(self):
        self.machine_mock.return_value = "x86_64"
        with self.assertRaisesRegex(MetalUnavailableError, "Apple Silicon macOS"):
            require_metal()
        self.load.assert_not_called()

    def test_missing_mlx_has_installation_guidance(self):
        self.load.side_effect = ModuleNotFoundError("missing", name="mlx")
        with self.assertRaisesRegex(MetalUnavailableError, "MLX is not installed"):
            require_metal()

    def test_missing_dependency_is_distinguished(self):
        self.load.side_effect = ModuleNotFoundError("missing", name="other_dependency")
        with self.assertRaisesRegex(MetalUnavailableError, "dependency is missing"):
            require_metal()

    def test_initialization_failure_has_diagnostic(self):
        self.load.side_effect = ImportError("Metal unavailable")
        with self.assertRaisesRegex(MetalUnavailableError, "could not initialize"):
            require_metal()

    def test_unavailable_metal_is_reported(self):
        self.mx.metal.is_available.return_value = False
        with self.assertRaisesRegex(MetalUnavailableError, "Metal is unavailable"):
            require_metal()
        self.mx.add.assert_not_called()

    def test_failed_gpu_evaluation_is_reported(self):
        self.mx.eval.side_effect = RuntimeError("GPU failure")
        with self.assertRaisesRegex(MetalUnavailableError, "GPU operation failed"):
            require_metal()


if __name__ == "__main__":
    unittest.main()
