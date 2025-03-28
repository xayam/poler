
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=16 * 8 * 8,
                nhead=8,
                dim_feedforward=1536
            ),
            enable_nested_tensor=False,
            num_layers=3
        )
        self.decoder = nn.Sequential(
            nn.Linear(16 * 8 * 8, 1536),
            nn.ReLU(),
            nn.Linear(1536, 16 * 8 * 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, seq_len, 16, 8, 8)
        batch_size, seq_len = x.shape[:2]
        x = x.view(batch_size, seq_len, -1)  # flatten spatial dims
        x = self.encoder(x)  # (batch, seq, 16*8*8)
        x = self.decoder(x[:, -1, :])  # берем последний элемент последовательности
        return x.view(batch_size, 16, 8, 8)  # восстанавливаем форму
