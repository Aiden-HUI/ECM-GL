# model_2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool

class GINEncoder(nn.Module):

    def __init__(self, input_dim=133, hidden_dim=256, output_dim=256, num_layers=3):

        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.edge_proj = nn.Linear(9, hidden_dim)

        self.convs = nn.ModuleList()
        self.mlps  = nn.ModuleList()

        self.convs.append(GINConvLayer(hidden_dim, hidden_dim))
        self.mlps.append(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

        for _ in range(num_layers - 1):
            self.convs.append(GINConvLayer(hidden_dim, hidden_dim))
            self.mlps.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))

        self.pool_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch=None):

        edge_embed = self.edge_projection(edge_attr)

        h = self.input_proj(x)

        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_embed)
            h = self.mlps[i](h)
            if i > 0:
                h = h + res_conn
            h = F.leaky_relu(h, negative_slope=0.1)
            res_conn = h

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h_pool = self.pool_proj(h)

        max_pool = global_max_pool(h_pool, batch)
        mean_pool = global_mean_pool(h_pool, batch)
        graph_embedding = torch.cat([max_pool, mean_pool], dim=-1)

        return graph_embedding

    def edge_projection(self, edge_attr):
        if not hasattr(self, 'edge_proj'):
            self.edge_proj = nn.Linear(edge_attr.size(-1), self.hidden_dim)
        return self.edge_proj(edge_attr)


class GINConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.eps = nn.Parameter(torch.Tensor([0.0]))  # 可学习的 epsilon

    def forward(self, x, edge_index, edge_embed):
        row, col = edge_index

        msg = x[col] * edge_embed
        agg = torch.zeros_like(x)
        agg = agg.index_add_(0, row, msg)

        out = self.mlp((1 + self.eps) * x + agg)
        return out
