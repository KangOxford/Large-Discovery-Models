"""Isolated tensor-contract worker for untrusted quantizer source."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys


def _safe_builtins() -> dict[str, object]:
    return {
        "__import__": __import__,
        "__build_class__": __build_class__,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "RuntimeError": RuntimeError,
        "set": set,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "zip": zip,
    }


def main() -> int:
    import torch

    source = Path(sys.argv[1]).read_text(encoding="utf-8")
    requested_device = sys.argv[2]
    device = requested_device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA tensor contract requested but CUDA is unavailable")
    namespace = {
        "__builtins__": _safe_builtins(),
        "__name__": "__candidate__",
        "torch": torch,
        "math": math,
        "FP_BITS": 16.0,
    }
    exec(compile(source, "<candidate>", "exec"), namespace)
    quantizer = namespace["AdaptiveKVQuantizer"]()
    quantizer.reset_request(
        {"workload": "longbench_hotpotqa", "example_id": "contract"},
        {"budget_bits": 4},
    )
    position = str(quantizer.query_observation_position())
    if position not in {"pre_rope", "post_rope"}:
        raise ValueError("query_observation_position must be pre_rope or post_rope")

    if bool(quantizer.needs_prefill_qkv_observer()):
        observed = torch.randn(1, 8, 32, 128, device=device, dtype=torch.float32)
        quantizer.observe_prefill_qkv(0, observed, observed, observed, {})

    errors = []
    bits = []
    for seq_len in (1, 31, 64, 257):
        tensor = torch.randn(1, 8, seq_len, 128, device=device, dtype=torch.float32)
        for kind, operation in (
            ("key", quantizer.quantize_key),
            ("value", quantizer.quantize_value),
        ):
            result = operation(0, tensor, {"seq_len": seq_len, "kv_kind": kind})
            output, avg_bits = result if isinstance(result, tuple) else (result, 16.0)
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"quantize_{kind} must return a tensor or pair")
            if output.shape != tensor.shape:
                raise ValueError(f"quantize_{kind} changed the tensor shape")
            if output.dtype != tensor.dtype or output.device != tensor.device:
                raise ValueError(f"quantize_{kind} changed tensor dtype or device")
            if not torch.isfinite(output).all():
                raise ValueError(f"quantize_{kind} returned non-finite values")
            numeric_bits = float(avg_bits)
            if not math.isfinite(numeric_bits) or not 0 < numeric_bits <= 16:
                raise ValueError(f"quantize_{kind} returned invalid average bits")
            bits.append(numeric_bits)
            errors.append(float((output - tensor).abs().mean().item()))

    estimated = []
    for layer_id in range(36):
        for kind in ("key", "value"):
            value = float(quantizer.estimate_bits(layer_id, kind, 4096, 128, {}))
            if not math.isfinite(value) or not 0 < value <= 16:
                raise ValueError("estimate_bits must be finite and in (0, 16]")
            estimated.append(value)

    state_tensor_elements = sum(
        int(value.numel())
        for value in vars(quantizer).values()
        if isinstance(value, torch.Tensor)
    )
    print(json.dumps({
        "status": "ok",
        "device": device,
        "mean_absolute_error": sum(errors) / len(errors),
        "observed_effective_kv_bits": sum(bits) / len(bits),
        "effective_kv_bits": sum(estimated) / len(estimated),
        "kv_compression_ratio": 16.0 / max(sum(estimated) / len(estimated), 1.0e-9),
        "state_tensor_elements": state_tensor_elements,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
