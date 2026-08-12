"""Evaluate a saved MLS-Bench inverse-folding checkpoint without training."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ldm_checkpoint_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=("CATH4.2", "CATH4.3", "TS50"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    module = load_module(args.candidate)
    dataset_name = "TS" if args.dataset == "TS50" else args.dataset
    test_dataset = module.load_dataset(dataset_name, args.data_root, "test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=module.collate_fn,
        pin_memory=True,
    )

    device = torch.device("cuda:0")
    model = module.InverseFoldingModel(
        hidden_dim=128,
        num_encoder_layers=3,
        k_neighbors=30,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    recovery, perplexity = module.evaluate(model, test_loader, device, "test")
    print(
        "TEMP_METRICS "
        + json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "dataset": args.dataset,
                "perplexity": perplexity,
                "recovery": recovery,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
