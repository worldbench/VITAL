"""F0-only low-rank adapters for SimWAM's video K/V conditioning path.

The video transformer shares its projection weights across video frames.  A normal
LoRA would therefore change both the clean conditioning frame and noisy future
tokens during joint training.  ``F0GatedLoRALinear`` applies its residual only to
the first-frame token prefix.  Future-video and action queries can both consume the
adapted F0 K/V, while the released future-token path remains bitwise unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import torch
import torch.nn as nn


class F0GatedLoRALinear(nn.Module):
    """Frozen ``Linear`` plus a LoRA residual on the F0 token prefix only."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"F0 LoRA rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"F0 LoRA alpha must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"F0 LoRA dropout must be in [0,1), got {dropout}")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = (
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        )
        # This integer is deliberately not a tensor/buffer: it describes the
        # current token layout and must not become checkpoint state.  It remains
        # set through activation-checkpoint recomputation in backward.
        self.f0_prefix_tokens: int | None = None

    def set_f0_prefix_tokens(self, value: int) -> None:
        value = int(value)
        if value <= 0:
            raise ValueError(f"F0 prefix must be positive, got {value}")
        self.f0_prefix_tokens = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "F0-gated video projection expects [B,S,D], got "
                f"{tuple(x.shape)}"
            )
        if self.f0_prefix_tokens is None:
            raise RuntimeError(
                "F0 token prefix was not set before the video expert forward"
            )
        prefix = int(self.f0_prefix_tokens)
        if prefix > x.shape[1]:
            raise ValueError(
                f"F0 prefix {prefix} exceeds video sequence length {x.shape[1]}"
            )

        base_out = self.base(x)
        f0 = self.lora_dropout(x[:, :prefix])
        delta = (f0 @ self.lora_A.t()) @ self.lora_B.t()
        adapted_f0 = base_out[:, :prefix] + self.scaling * delta
        if prefix == x.shape[1]:
            return adapted_f0
        return torch.cat((adapted_f0, base_out[:, prefix:]), dim=1)


def _canonical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    target_modules = tuple(str(name) for name in config.get("target_modules", ("k", "v")))
    if target_modules != ("k", "v"):
        raise ValueError(
            "The causal F0 adapter currently supports target_modules=[k,v] only, "
            f"got {list(target_modules)}"
        )
    canonical = {
        "last_n_blocks": int(config.get("last_n_blocks", 8)),
        "rank": int(config.get("rank", 16)),
        "alpha": float(config.get("alpha", 16.0)),
        "dropout": float(config.get("dropout", 0.0)),
        "target_modules": list(target_modules),
    }
    if canonical["last_n_blocks"] <= 0:
        raise ValueError("f0_adapter.last_n_blocks must be positive")
    return canonical


def iter_f0_adapters(model: nn.Module) -> Iterator[tuple[str, F0GatedLoRALinear]]:
    for name, module in model.video_expert.named_modules():
        if isinstance(module, F0GatedLoRALinear):
            yield name, module


def configure_f0_kv_adapter(model: nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    """Attach F0-gated K/V LoRA to the last video blocks in place."""

    canonical = _canonical_config(config)
    existing = list(iter_f0_adapters(model))
    if existing:
        if getattr(model, "f0_adapter_config", None) != canonical:
            raise ValueError(
                "Model already has another F0 adapter configuration: "
                f"existing={getattr(model, 'f0_adapter_config', None)} "
                f"requested={canonical}"
            )
        return canonical

    blocks = model.video_expert.blocks
    if canonical["last_n_blocks"] > len(blocks):
        raise ValueError(
            "f0_adapter.last_n_blocks exceeds video depth: "
            f"{canonical['last_n_blocks']} > {len(blocks)}"
        )
    first_block = len(blocks) - canonical["last_n_blocks"]
    for block_index in range(first_block, len(blocks)):
        attention = blocks[block_index].self_attn
        for projection_name in canonical["target_modules"]:
            projection = getattr(attention, projection_name)
            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"video block {block_index} {projection_name} is not nn.Linear: "
                    f"{type(projection)}"
                )
            setattr(
                attention,
                projection_name,
                F0GatedLoRALinear(
                    projection,
                    rank=canonical["rank"],
                    alpha=canonical["alpha"],
                    dropout=canonical["dropout"],
                ),
            )

    model.f0_adapter_config = canonical
    model.f0_adapter_first_block = first_block
    return canonical


def set_f0_prefix_tokens(model: nn.Module, tokens_per_frame: int) -> None:
    adapters = list(iter_f0_adapters(model))
    if not adapters:
        return
    for _, adapter in adapters:
        adapter.set_f0_prefix_tokens(tokens_per_frame)


def f0_adapter_named_parameters(
    model: nn.Module,
) -> Iterator[tuple[str, nn.Parameter]]:
    for module_name, adapter in iter_f0_adapters(model):
        yield f"{module_name}.lora_A", adapter.lora_A
        yield f"{module_name}.lora_B", adapter.lora_B


def f0_adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in f0_adapter_named_parameters(model)
    }


def load_f0_adapter_state_dict(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    expected = dict(f0_adapter_named_parameters(model))
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(
            f"F0 adapter state mismatch: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            source = state[name]
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"F0 adapter tensor shape mismatch for {name}: "
                    f"{tuple(source.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))

