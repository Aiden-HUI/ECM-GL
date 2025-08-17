# wrapper.py
import torch
import torch.nn as nn

class MultiTaskWrapper(nn.Module):

    def __init__(self, encoder_2d: nn.Module, encoder_3d: nn.Module,
                 contrast_module: nn.Module, cls_head: nn.Module):
        super().__init__()
        self.encoder_2d = encoder_2d
        self.encoder_3d = encoder_3d
        self.contrast_module = contrast_module
        self.cls_head = cls_head

    def forward(self, data):

        z_2d = self.encoder_2d(data.x, data.edge_index, data.edge_attr, data.batch)

        z_3d = self.encoder_3d(data.pos, data.x, data.batch)


        loss_contrast, sim_matrix = self.contrast_module(z_2d, z_3d)

        cls_logits = self.cls_head(z_2d, z_3d)

        return {
            'contrast_loss': loss_contrast,
            'cls_logits': cls_logits,
            'similarity_matrix': sim_matrix
        }
