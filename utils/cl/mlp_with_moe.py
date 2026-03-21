import torch
import torch.nn as nn
from utils.cl.moe import MoEAdapter

class MLPWithMoE(nn.Module):
    """Class for augmenting model MLP layers with MoE Adapters"""

    def __init__(self, mlp, d_model, num_experts=8, rank=16, top_k=2, dropout=0.0, 
                 existing_experts=None, existing_routers=None, mode="train", level_id=None):
        super().__init__()
        self.mlp = mlp
        self.moe = MoEAdapter(d_model, num_experts, rank, top_k, dropout,
                              existing_experts, existing_routers, mode, level_id)
        self.alpha = nn.Parameter(torch.ones(1) * 0.01) # Learnable scale so MLP and MoE are on the same scale
        self.alpha.requires_grad = True

    def forward(self, x):
        return self.mlp(x) + self.alpha * self.moe(x)
    

'''
Modifying the old model architecture:

num_experts = 8
rank = 16
top_k = 2
dropout = 0.0
existing_experts...
existing_routers...
mode = "train" 

for layer in model.model.layers:
    d_model = layer.mlp.down_proj.in_features
    layer.mlp = MLPWithMoE(layer.mlp, d_model, rank, top_k, dropout,
                            existing_experts, existing_routers, mode)
'''

'''
Freeze everything except MoEAdapter

# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Un-freeze MoEAdapter
for module in model.modules():
    if isinstance(module, MoEAdapter):
        for param in module.parameters():
            param.requires_grad = True

'''