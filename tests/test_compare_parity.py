import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from scripts.compare_parity import compare_files, main


class CompareParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.reference = Path(self.temp.name) / "reference.safetensors"
        self.candidate = Path(self.temp.name) / "candidate.safetensors"
        save_file({"expected": np.array([1.0, 2.0], dtype=np.float32)}, str(self.reference))
        save_file({"actual": np.array([1.0, 2.01], dtype=np.float32)}, str(self.candidate))

    def run_cli(self, atol="0.02", rtol="0"):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main([str(self.reference), str(self.candidate), "--reference-key", "expected",
                         "--candidate-key", "actual", "--atol", atol, "--rtol", rtol])
        return code, output.getvalue(), errors.getvalue()

    def test_tolerance_controls_status_and_report(self):
        code, output, _ = self.run_cli()
        report = json.loads(output)
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["count"], 2)
        self.assertAlmostEqual(report["max_absolute_error"], 0.01, places=5)
        code, output, _ = self.run_cli(atol="0.001")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output)["failed_count"], 1)

    def test_nonfinite_and_invalid_inputs(self):
        save_file({"actual": np.array([np.nan, 2.0], dtype=np.float32)}, str(self.candidate))
        code, output, _ = self.run_cli()
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output)["finite"])
        code, _, errors = self.run_cli(atol="-1")
        self.assertEqual(code, 2)
        self.assertIn("non-negative", errors)
        with self.assertRaisesRegex(ValueError, "Reference tensor not found"):
            compare_files(self.reference, self.candidate, reference_key="missing", candidate_key="actual", atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "Candidate tensor not found"):
            compare_files(self.reference, self.candidate, reference_key="expected", candidate_key="missing", atol=0, rtol=0)

    def test_shape_mismatch_missing_file_and_invalid_format(self):
        save_file({"actual": np.array([1.0], dtype=np.float32)}, str(self.candidate))
        self.assertEqual(self.run_cli()[0], 2)
        self.candidate.unlink()
        self.assertEqual(self.run_cli()[0], 2)
        self.candidate.write_bytes(b"not safetensors")
        self.assertEqual(self.run_cli()[0], 2)


if __name__ == "__main__":
    unittest.main()
