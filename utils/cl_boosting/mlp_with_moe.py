import torch
import torch.nn as nn
from utils.cl.moe import MoEAdapter
 
class MLPWithMoE(nn.Module):
    """Class for augmenting model MLP layers with MoE Adapters"""
 
    def __init__(self, mlp, d_model, num_experts=8, rank=16, top_k=2, dropout=0.0, 
                 existing_experts=None, existing_routers=None, mode="train", level_id=None,
                 existing_alphas=None):
        super().__init__()
        self.mlp = mlp
        self.mode = mode
        self.level_id = level_id
        self.moe = MoEAdapter(d_model, num_experts, rank, top_k, dropout,
                              existing_experts, existing_routers, mode, level_id)
 
        if mode == "eval":
            # At eval time, load exactly the saved alphas — no new alpha appended.
            # All are frozen since we're not training.
            assert existing_alphas is not None, "existing_alphas must be provided in eval mode"
            self.alphas = nn.ParameterList(
                [nn.Parameter(a.data.clone()) for a in existing_alphas]
            )
            for alpha in self.alphas:
                alpha.requires_grad = False
 
        else:
            # At train time, carry over past alphas (frozen) and append a new trainable one.
            if existing_alphas is not None:
                past_alphas = [nn.Parameter(a.data.clone()) for a in existing_alphas]
            else:
                num_existing_tasks = len(existing_routers) if existing_routers else 0
                past_alphas = [nn.Parameter(torch.ones(1) * 0.01) for _ in range(num_existing_tasks)]
            new_alpha = nn.Parameter(torch.ones(1) * 0.01)
            self.alphas = nn.ParameterList(past_alphas + [new_alpha])
 
            # Freeze past alphas, leave current one trainable
            for alpha in self.alphas[:-1]:
                alpha.requires_grad = False
            self.alphas[-1].requires_grad = True
 
    def forward(self, x):
        if self.mode == "train":
            alpha = self.alphas[-1]
        else:
            alpha = self.alphas[self.level_id]
        return self.mlp(x) + alpha * self.moe(x)