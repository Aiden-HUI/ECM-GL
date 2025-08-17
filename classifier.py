# classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):

    def __init__(
            self,
            dim_2d: int = 256,
            dim_3d: int = 128,
            hidden_dim1: int = 192,
            hidden_dim2: int = 96,
            dropout: float = 0.1
    ):
        super().__init__()
        total_dim = dim_2d + dim_3d

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, z_2d, z_3d):
        x = torch.cat([z_2d, z_3d], dim=-1)

        return self.classifier(x).squeeze(-1)
