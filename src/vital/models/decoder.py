"""FP32 visual scaffold plus pose-residual decoder; no feature caches needed."""
import torch
from torch import nn
from .decoder_layers import SE2IncrementDecoder


class ResidualDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('residual_mean', torch.zeros(30))
        self.register_buffer('basis', torch.zeros(30, 12))
        self.register_buffer('coefficient_std', torch.ones(12))

    def forward(self, tokens):
        code = tokens[..., 64:].float()
        return ((code * self.coefficient_std) @ self.basis.T + self.residual_mean).reshape(-1, 40, 3)


class TrajectoryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_decoder = SE2IncrementDecoder(action_dim=64, parameterization='velocity')
        self.residual_decoder = ResidualDecoder()

    def forward(self, tokens):
        if tuple(tokens.shape[1:]) != (4, 76):
            raise ValueError('Expected [B,4,76] tokens')
        times = torch.arange(1, 41, device=tokens.device, dtype=torch.float32)[None] / 10
        base, _ = self.base_decoder(tokens[..., :64].float(), times.expand(len(tokens), -1))
        result = base.float() + self.residual_decoder(tokens)
        return torch.cat([result[..., :2], torch.atan2(torch.sin(result[..., 2:3]),
                                                      torch.cos(result[..., 2:3]))], -1)
