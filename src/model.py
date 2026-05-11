import torch
import torch.nn as nn


class IsingAutoregressiveTransformer(nn.Module):
    def __init__(
        self,
        grid_size=16,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.0,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.seq_len = grid_size * grid_size

        # 0 = spin -1, 1 = spin +1, 2 = beginning-of-sequence token
        self.vocab_size = 3

        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.row_embedding = nn.Embedding(grid_size, d_model)
        self.col_embedding = nn.Embedding(grid_size, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, input_tokens):
        B, L = input_tokens.shape
        device = input_tokens.device

        positions = torch.arange(L, device=device)
        rows = positions // self.grid_size
        cols = positions % self.grid_size

        h = (
            self.token_embedding(input_tokens)
            + self.row_embedding(rows)[None, :, :]
            + self.col_embedding(cols)[None, :, :]
        )

        causal_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1,
        )

        h = self.transformer(h, mask=causal_mask)
        return self.head(h)


def spins_to_tokens(spins):
    """
    spins: torch tensor [B, N, N], values in {-1, +1}
    returns: torch tensor [B, N*N], values in {0, 1}
    """
    tokens = ((spins + 1) // 2).long()
    return tokens.reshape(tokens.shape[0], -1)