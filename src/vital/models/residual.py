"""Frozen six-coordinate post-policy used by the released one-step model."""
import torch
from torch import nn
from .residual_math import physical_project


class ResidualPolicy(nn.Module):
    def __init__(self, state, config):
        super().__init__()
        if config['kind'] != 'canonical6':
            raise ValueError('This release uses the fixed six-coordinate policy')
        self.config = config
        for key in ('feature_mean', 'feature_std', 'matrix', 'fixed_basis'):
            self.register_buffer(key, torch.empty_like(state[key]))
        self.net = nn.Sequential(nn.Linear(4528, 256), nn.SiLU(), nn.Linear(256, 256),
                                 nn.SiLU(), nn.Linear(256, 6))
        self.load_state_dict(state, strict=True)
        self.eval().requires_grad_(False)

    def correction(self, features):
        normalized = ((features.float() - self.feature_mean) /
                      self.feature_std.clamp_min(1e-3)).clamp(-10, 10)
        coordinates = self.net(normalized)
        bases = self.fixed_basis[None].expand(len(features), -1, -1)
        raw = torch.einsum('b...k,bkd->b...d', coordinates, bases)
        delta = physical_project(raw, self.matrix,
            **{key: self.config[key] for key in ('max_xy', 'max_yaw', 'max_rms_q')})
        return delta.reshape(-1, 4, 12)
