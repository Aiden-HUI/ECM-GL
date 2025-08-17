# model_3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_mean


class GeoDirNet(nn.Module):

    def __init__(self, node_dim=133, hidden_dim=256, output_dim=128, num_layers=4, k_neighbors=16):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.num_layers = num_layers

        self.embed = nn.Linear(node_dim, hidden_dim)

        self.conv_layers = nn.ModuleList([
            GeoDirNetConvLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, pos, x, batch=None):
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

        edge_index = self.build_knn_graph(pos, batch)

        h = self.embed(x)

        for conv in self.conv_layers:
            h = conv(h, pos, edge_index)

        h_mean = scatter_mean(h, batch, dim=0)
        graph_embed = self.proj_head(h_mean)

        return graph_embed

    def build_knn_graph(self, pos, batch=None):
        N = pos.size(0)
        k = min(self.k_neighbors + 1, N)
        dist = torch.cdist(pos, pos)
        idx = dist.topk(k=k, largest=False).indices
        knn_idx = idx[:, 1:]
        src = torch.arange(N, device=pos.device).unsqueeze(1).repeat(1, knn_idx.size(1))
        dst = knn_idx
        edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)  # [2, N*(k-1)]
        return edge_index


class GeoDirNetConvLayer(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='mean')

        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 3, out_dim),  # 输入特征 + 相对坐标
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(out_dim + in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.equiv_weight = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(),
            nn.Linear(16, out_dim)
        )

    def forward(self, h, pos, edge_index):
        edge_index, _ = add_self_loops(edge_index)
        out = self.propagate(edge_index, h=h, pos=pos)

        return out + h

    def message(self, h_i, h_j, pos_j, pos_i):
        rel_pos = pos_j - pos_i
        distance = torch.norm(rel_pos, dim=-1, keepdim=True)

        weight = self.equiv_weight(rel_pos)

        msg_input = torch.cat([h_i, h_j, rel_pos], dim=-1)
        msg = self.msg_mlp(msg_input)

        return (msg * weight) / (distance + 1e-6)

    def update(self, aggr_msg, h):
        return self.update_mlp(torch.cat([h, aggr_msg], dim=-1))
