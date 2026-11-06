"""Explicit CUDA smoke check; no checkpoint or model allocation is needed.

Run manually after rebuilding Wall-X's CUDA extension:
    PYTHONPATH=. python tests/check_window_index_cuda.py --repeats 20

This is intentionally not an automatically collected GPU test.
"""
import argparse
import json
from pathlib import Path

import torch

from wall_x.fusions import ops


def reference(grids, merge=2, window=4):
    """Visit each frame's spatial windows in row-major order on the CPU."""
    indices, boundaries, offset = [], [0], 0
    for frames, height, width in grids:
        height, width = height // merge, width // merge
        for frame in range(frames):
            for row in range(0, height, window):
                for col in range(0, width, window):
                    values = [offset + frame * height * width + y * width + x
                              for y in range(row, min(row + window, height))
                              for x in range(col, min(col + window, width))]
                    indices.extend(values)
                    boundaries.append(boundaries[-1] + len(values) * merge * merge)
        offset += frames * height * width
    return torch.tensor(indices), torch.tensor(boundaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This explicit check requires a CUDA device")

    reports = []
    cases = [(f"pap_b{batch}", [[1, 14, 18]] * (2 * batch)) for batch in (1, 4, 32, 64)]
    # Unequal grids, exact window multiples and temporal frames also exercise
    # offsets across image boundaries, rather than only replicated PAP grids.
    cases.append(("mixed_128_images", [[1, 14, 18], [2, 16, 12], [1, 8, 8], [1, 20, 10]] * 32))
    for name, grids in cases:
        expected_index, expected_cu = reference(grids)
        host_grid = torch.tensor(grids, dtype=torch.int32, pin_memory=True)
        for use_nondefault in (False, True):
            stream = torch.cuda.Stream() if use_nondefault else torch.cuda.default_stream()
            for iteration in range(args.repeats):
                with torch.cuda.stream(stream):
                    grid = host_grid.to("cuda", non_blocking=True)
                    # Also queue a producer on the tested stream: the extension
                    # must not jump to CUDA's default stream behind its back.
                    grid = grid + torch.zeros_like(grid)
                    indices, cu = ops.get_window_index(
                        grid, window_size=112, spatial_merge_size=2,
                        patch_size=14, spatial_merge_unit=4,
                    )
                    actual_index, actual_cu = indices.cpu().long(), cu.cpu().long()
                if not torch.equal(actual_index, expected_index):
                    raise AssertionError(f"{name} stream={use_nondefault} repeat={iteration}: window ordering mismatch")
                if not torch.equal(actual_cu, expected_cu):
                    raise AssertionError(f"{name} stream={use_nondefault} repeat={iteration}: prefix sum mismatch")
            reports.append(dict(case=name, nondefault_stream=use_nondefault,
                                windows=len(expected_cu) - 1, merged_tokens=len(expected_index),
                                repeats=args.repeats, passed=True))
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, checks=reports, passed=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
