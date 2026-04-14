import torch
import torch.nn as nn
from utils.cl.moe import MoEAdapter
 
class MLPWithMoE(nn.Module):
    """Class for augmenting model MLP layers with MoE Adapters"""
 
    def __init__(self, mlp, d_model, new_expert_count=2, rank=16, top_k=2, old_experts=None, 
                 old_routers=None, mode="train", level_id=None, old_alphas=None):
        super().__init__()

        self.mlp = mlp
        self.mode = mode
        self.level_id = level_id
        self.moe = MoEAdapter(d_model, new_expert_count, rank, top_k,
                              old_experts, old_routers, mode, level_id)
 
        if "eval" in mode:
            self.alphas = nn.ParameterList(
                [nn.Parameter(a.data.clone()) for a in old_alphas]
            )
            for alpha in self.alphas:
                alpha.requires_grad = False
 
        else:
            # At train time, carry over past alphas (frozen) and append a new trainable one.
            if old_alphas is not None:
                past_alphas = [nn.Parameter(a.data.clone()) for a in old_alphas]
            else:
                existing_task_count = len(old_routers) if old_routers else 0
                past_alphas = [nn.Parameter(torch.ones(1) * 0.01) for _ in range(existing_task_count)]

            new_alpha = nn.Parameter(torch.ones(1) * 0.01)
            self.alphas = nn.ParameterList(past_alphas + [new_alpha])
 
            # Freeze past alphas, leave current one trainable
            for alpha in self.alphas[:-1]:
                alpha.requires_grad = False
            self.alphas[-1].requires_grad = True
 

    def forward(self, x):
        if "train" in self.mode:
            alpha = self.alphas[-1]
        else:
            alpha = self.alphas[self.level_id]
        return self.mlp(x) + alpha * self.moe(x)