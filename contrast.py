# contrast.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph, dropout_node, dropout_edge, k_hop_subgraph
from torch.nn import Parameter
import torch.optim as optim
import random

class ContrastiveLearningModule(nn.Module):

    def __init__(self, dim_2d=256, dim_3d=128, proj_dim=64, temperature=0.7):
        super().__init__()
        self.proj_dim = proj_dim
        self.temperature = temperature

        self.proj_2d = nn.Sequential(
            nn.Linear(dim_2d, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
        self.proj_3d = nn.Sequential(
            nn.Linear(dim_3d, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in list(self.proj_2d) + list(self.proj_3d):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z_2d, z_3d):

        p_2d = F.normalize(self.proj_2d(z_2d), dim=-1)
        p_3d = F.normalize(self.proj_3d(z_3d), dim=-1)

        logits = torch.matmul(p_2d, p_3d.T) / self.temperature

        labels = torch.arange(logits.size(0), device=z_2d.device)

        loss_2d_to_3d = F.cross_entropy(logits, labels)
        loss_3d_to_2d = F.cross_entropy(logits.T, labels)
        loss = (loss_2d_to_3d + loss_3d_to_2d) / 2

        return loss, logits.detach()