"""Compare named tensors from trusted local safetensors files."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

from safetensors import SafetensorError
from safetensors.numpy import load_file

from scripts.parity_metrics import compare


def compare_files(reference_path, candidate_path, *, reference_key, candidate_key, atol, rtol):
    """Load paired tensors and return comparison metrics using explicit tolerances."""
    reference = load_file(str(reference_path))
    candidate = load_file(str(candidate_path))
    if reference_key not in reference:
        raise ValueError(f"Reference tensor not found: {reference_key}")
    if candidate_key not in candidate:
        raise ValueError(f"Candidate tensor not found: {candidate_key}")
    return compare(reference[reference_key], candidate[candidate_key], atol=atol, rtol=rtol)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="trusted reference safetensors file")
    parser.add_argument("candidate", type=Path, help="trusted candidate safetensors file")
    parser.add_argument("--reference-key", required=True)
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--atol", required=True, type=float)
    parser.add_argument("--rtol", required=True, type=float)
    args = parser.parse_args(argv)
    try:
        result = compare_files(args.reference, args.candidate, reference_key=args.reference_key,
                               candidate_key=args.candidate_key, atol=args.atol, rtol=args.rtol)
    except (OSError, ValueError, TypeError, SafetensorError) as error:
        print(f"Parity comparison failed: {error}", file=sys.stderr)
        return 2
    report = asdict(result)
    for key in ("max_absolute_error", "mean_absolute_error"):
        if report[key] is not None and not math.isfinite(report[key]):
            report[key] = "infinity"
    report.update(reference_key=args.reference_key, candidate_key=args.candidate_key,
                  atol=args.atol, rtol=args.rtol)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
