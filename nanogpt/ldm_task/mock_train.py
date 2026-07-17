#!/usr/bin/env python3
"""
Mock autoresearch training script for fast TTS search tests.

The search runner copies this file into each candidate state as `train.py`.
Code-edit agents can improve the mock score by changing the knobs below while
preserving the final diagnostics output shape used by the real training script.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Mock search knobs. These intentionally look like train.py hyperparameters.
# Lower val_bpb is better. The hidden-ish optimum is near:
# depth=10, width=704, matrix_lr=0.022, embedding_lr=0.55,
# weight_decay=0.16, value_gate_channels=64, warmdown_ratio=0.45,
# ngram_scale=0.18.
# ---------------------------------------------------------------------------

DEPTH = 8
WIDTH = 512
MATRIX_LR = 0.016
EMBEDDING_LR = 0.40
WEIGHT_DECAY = 0.10
VALUE_GATE_CHANNELS = 32
WARMDOWN_RATIO = 0.35
NGRAM_SCALE = 0.12


def penalty(value, target, scale, weight):
    return weight * ((float(value) - target) / scale) ** 2


def log_penalty(value, target, weight):
    value = max(float(value), 1e-12)
    target = max(float(target), 1e-12)
    return weight * math.log(value / target) ** 2


def file_jitter():
    """Tiny deterministic jitter so different code edits can break ties."""
    data = Path(__file__).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return (unit - 0.5) * 0.0004


def compute_mock_metrics():
    if DEPTH <= 0 or WIDTH <= 0 or MATRIX_LR <= 0 or EMBEDDING_LR <= 0:
        raise ValueError("Mock knobs must be positive.")
    if DEPTH * WIDTH > 10_000:
        raise RuntimeError("Mock OOM: DEPTH * WIDTH is too large.")

    score = 0.910
    score += penalty(DEPTH, 10, 2.0, 0.011)
    score += penalty(WIDTH, 704, 192.0, 0.010)
    score += log_penalty(MATRIX_LR, 0.022, 0.010)
    score += log_penalty(EMBEDDING_LR, 0.55, 0.007)
    score += penalty(WEIGHT_DECAY, 0.16, 0.08, 0.008)
    score += penalty(VALUE_GATE_CHANNELS, 64, 32.0, 0.006)
    score += penalty(WARMDOWN_RATIO, 0.45, 0.20, 0.006)
    score += penalty(NGRAM_SCALE, 0.18, 0.08, 0.006)

    if WIDTH % 64 != 0:
        score += 0.012
    if VALUE_GATE_CHANNELS % 16 != 0:
        score += 0.006

    score += file_jitter()

    num_params = DEPTH * WIDTH * WIDTH * 12
    steps_completed = max(100, int(780 - 0.018 * DEPTH * WIDTH))
    tokens_processed = steps_completed * 131_072
    peak_vram_mb = 900 + DEPTH * WIDTH * 0.75 + VALUE_GATE_CHANNELS * 3.0
    mfu_percent = max(5.0, min(85.0, 55.0 - abs(WIDTH - 704) / 40.0 - abs(DEPTH - 10)))
    final_train_bpb = score - 0.012 + penalty(WEIGHT_DECAY, 0.13, 0.10, 0.004)
    eval_train_gap = score - final_train_bpb

    return {
        "val_bpb": float(score),
        "training_seconds": 0.05,
        "total_seconds": 0.06,
        "startup_seconds": 0.01,
        "peak_vram_mb": float(peak_vram_mb),
        "mfu_percent": float(mfu_percent),
        "tokens_processed": int(tokens_processed),
        "steps_completed": int(steps_completed),
        "final_train_loss": float(1.7 + (score - 0.91) * 3.0),
        "final_train_bpb": float(final_train_bpb),
        "eval_train_gap": float(eval_train_gap),
        "early_loss_slope": -7.0 + penalty(MATRIX_LR, 0.022, 0.010, 1.0),
        "mid_training_loss_slope": -1.4 + penalty(EMBEDDING_LR, 0.55, 0.20, 0.5),
        "final_loss_slope": -0.2 + penalty(WARMDOWN_RATIO, 0.45, 0.20, 0.3),
        "num_params": int(num_params),
        "num_params_m": float(num_params / 1e6),
        "depth": int(DEPTH),
        "mock_knobs": {
            "DEPTH": DEPTH,
            "WIDTH": WIDTH,
            "MATRIX_LR": MATRIX_LR,
            "EMBEDDING_LR": EMBEDDING_LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "VALUE_GATE_CHANNELS": VALUE_GATE_CHANNELS,
            "WARMDOWN_RATIO": WARMDOWN_RATIO,
            "NGRAM_SCALE": NGRAM_SCALE,
        },
    }


def write_diagnostics(metrics):
    diagnostics_path = os.environ.get("AUTORESEARCH_DIAGNOSTICS_JSON", "mock_train_diagnostics.json")
    if not diagnostics_path:
        return
    path = Path(diagnostics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    start = time.time()
    if os.environ.get("TTS_MOCK_FAIL") == "1":
        print("FAIL")
        return 1

    time.sleep(float(os.environ.get("TTS_MOCK_SLEEP", "0.01")))
    metrics = compute_mock_metrics()
    metrics["total_seconds"] = float(time.time() - start)
    write_diagnostics(metrics)

    print("---")
    print(f"val_bpb:          {metrics['val_bpb']:.6f}")
    print(f"training_seconds: {metrics['training_seconds']:.1f}")
    print(f"total_seconds:    {metrics['total_seconds']:.1f}")
    print(f"peak_vram_mb:     {metrics['peak_vram_mb']:.1f}")
    print(f"mfu_percent:      {metrics['mfu_percent']:.2f}")
    print(f"total_tokens_M:   {metrics['tokens_processed'] / 1e6:.1f}")
    print(f"tokens_processed: {metrics['tokens_processed']}")
    print(f"num_steps:        {metrics['steps_completed']}")
    print(f"steps_completed:  {metrics['steps_completed']}")
    print(f"final_train_loss: {metrics['final_train_loss']:.6f}")
    print(f"final_train_bpb:  {metrics['final_train_bpb']:.6f}")
    print(f"eval_train_gap:   {metrics['eval_train_gap']:.6f}")
    print(f"early_loss_slope: {metrics['early_loss_slope']:.6f}")
    print(f"mid_training_loss_slope: {metrics['mid_training_loss_slope']:.6f}")
    print(f"final_loss_slope: {metrics['final_loss_slope']:.6f}")
    print(f"num_params_M:     {metrics['num_params_m']:.1f}")
    print(f"depth:            {metrics['depth']}")
    print("diagnostics_json_inline: " + json.dumps(metrics, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

