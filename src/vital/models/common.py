from torch import Tensor, nn

class MLP(nn.Module):
    """Two-layer projection with LayerNorm-friendly GELU activation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)

