#!/usr/bin/env python3
"""Explicitly derive a checkpoint for audited source revision changes, offline.

Schema equality does not prove arbitrary source code is semantically identical.
Use trusted pickle checkpoints and cite the regression report for this change.
The ordinary trainer and checkpoint loaders retain their strict identity checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pressb.runtime.checkpoint_migration import migrate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True, type=Path, help="Trusted original SAC pickle checkpoint")
    parser.add_argument("--source-identity", required=True, type=Path, help="Original run's negotiated.json")
    parser.add_argument("--target-identity", required=True, type=Path, help="New nodes' actual negotiated identity JSON")
    parser.add_argument("--output", required=True, type=Path, help="New derived checkpoint path; must not exist")
    parser.add_argument("--regression-report", required=True, type=Path, help="Existing regression evidence, recorded by path and SHA256")
    parser.add_argument("--allow-source-revision-change", action="store_true", required=True,
                        help="Explicitly approve the audited source revisions; schema equality alone is insufficient")
    parser.add_argument("--source-frozen-checkpoint", type=Path, help="Original frozen DSRL checkpoint for residual migration")
    parser.add_argument("--target-frozen-checkpoint", type=Path, help="Previously migrated frozen DSRL checkpoint")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    report = migrate_checkpoint(
        args.source_checkpoint, args.source_identity, args.target_identity, args.output,
        allow_source_revision_change=args.allow_source_revision_change,
        regression_report=args.regression_report,
        source_frozen_checkpoint=args.source_frozen_checkpoint,
        target_frozen_checkpoint=args.target_frozen_checkpoint)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
