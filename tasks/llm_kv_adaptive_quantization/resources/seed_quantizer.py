class AdaptiveKVQuantizer:
    """Editable KV-cache quantizer.

    The fixed harness supplies real key/value tensors from a Hugging Face
    DynamicCache and calls this class for the actual algorithm. Participants
    may rewrite the quantization math, residual policy, optional prefill
    observation, and memory accounting here without changing the benchmark
    datasets, model, or decode loop.
    """

    def __init__(self):
        self.bits = 4
        self.key_group_size = 32
        self.value_group_size = 32
        self.key_residual_length = 128
        self.value_residual_length = 128

    def reset_request(self, request_meta: dict, budget_state: dict):
        self.bits = min(4, int(budget_state.get("budget_bits", 4)))
        workload = str(request_meta.get("workload", ""))
        residual = 128 if workload.startswith("longbench_") else 32
        self.key_residual_length = residual
        self.value_residual_length = residual

    def needs_prefill_qkv_observer(self) -> bool:
        return False

    def observe_prefill_qkv(
        self,
        layer_id: int,
        query_states: torch.Tensor | None,
        key_states: torch.Tensor | None,
        value_states: torch.Tensor | None,
        attention_meta: dict,
    ) -> None:
        return None

    def query_observation_position(self) -> str:
        return "post_rope"

    def _residual_keep_length(self, seq_len: int, residual_length: int, residual_policy: str = "tail") -> int:
        residual_length = max(0, min(seq_len, int(residual_length)))
        if residual_length == 0 or residual_policy in {"none", ""}:
            return 0
        if residual_policy == "block_modulo":
            return seq_len % residual_length
        if residual_policy == "tail":
            return residual_length
        raise ValueError(f"Unsupported residual_policy={residual_policy}")

    def _minmax_quantize_last_dim(self, data: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
        if data.numel() == 0 or bits >= FP_BITS - 0.5:
            return data
        max_int = max(1, int(2**int(bits)) - 1)
        trailing = data.shape[-1]
        group_size = trailing if int(group_size) <= 0 else int(group_size)
        padded = math.ceil(trailing / group_size) * group_size
        work = data
        if padded != trailing:
            work = torch.nn.functional.pad(work, (0, padded - trailing))
        grouped = work.reshape(*work.shape[:-1], padded // group_size, group_size)
        gmin = grouped.amin(dim=-1, keepdim=True)
        gmax = grouped.amax(dim=-1, keepdim=True)
        scale = (gmax - gmin).clamp(min=1e-5) / max_int
        quant = torch.round((grouped - gmin) / scale).clamp(0, max_int)
        dequant = quant.mul(scale).add(gmin)
        return dequant.reshape(*work.shape[:-1], padded)[..., :trailing]

    def _quantize_grouped_minmax(
        self,
        layer_tensor: torch.Tensor,
        *,
        axis: str,
        bits: int,
        group_size: int,
        residual_length: int,
        residual_policy: str = "tail",
    ) -> tuple[torch.Tensor, float]:
        work = layer_tensor.float().clone()
        batch, heads, seq_len, head_dim = work.shape
        residual = self._residual_keep_length(seq_len, residual_length, residual_policy)
        quant_end = seq_len - residual
        if quant_end <= 0 or bits >= FP_BITS - 0.5:
            return work.to(layer_tensor.dtype), FP_BITS

        quant_slice = work[:, :, :quant_end, :]
        if axis == "channel":
            quant_len = quant_slice.shape[-2]
            group_size = quant_len if int(group_size) <= 0 else int(group_size)
            usable = quant_len - (quant_len % group_size)
            main = quant_slice[:, :, :usable, :]
            tail = quant_slice[:, :, usable:, :]
            if usable > 0:
                main = main.transpose(2, 3).reshape(batch, heads, head_dim, usable // group_size, group_size)
                main = self._minmax_quantize_last_dim(main, bits, group_size)
                work[:, :, :usable, :] = main.reshape(batch, heads, head_dim, usable).transpose(2, 3)
            if tail.numel() > 0:
                work[:, :, usable:quant_end, :] = tail
            fp_tokens = residual + (quant_len - usable)
            avg_bits = (usable * bits + fp_tokens * FP_BITS) / max(seq_len, 1)
        else:
            flat = quant_slice.transpose(1, 2).reshape(batch, quant_slice.shape[-2], heads * head_dim)
            flat = self._minmax_quantize_last_dim(flat, bits, group_size)
            work[:, :, :quant_end, :] = flat.reshape(batch, quant_slice.shape[-2], heads, head_dim).transpose(1, 2)
            avg_bits = (quant_end * bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(layer_tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._quantize_grouped_minmax(
            key_states,
            axis="channel",
            bits=self.bits,
            group_size=self.key_group_size,
            residual_length=self.key_residual_length,
            residual_policy="tail",
        )

    def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._quantize_grouped_minmax(
            value_states,
            axis="token",
            bits=self.bits,
            group_size=self.value_group_size,
            residual_length=self.value_residual_length,
            residual_policy="tail",
        )

    def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
        residual = self.key_residual_length if kv_kind == "key" else self.value_residual_length
        residual = self._residual_keep_length(seq_len, residual, "tail")
        quant_tokens = max(0, seq_len - residual)
        return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))
