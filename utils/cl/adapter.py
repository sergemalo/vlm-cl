import torch.nn as nn
import math

"""
Note: LoRA-style adapter as proposed in paper https://arxiv.org/pdf/2403.11549 
    1) Low-rank structure adaptation
    2) Zero initialization
    
Importantly:
    - it does not update weight matrices (not true LoRA), 
    - it is a separate module (+ nonlinearity)
"""

class Adapter(nn.Module):
    """Class for one LoRA-style adapter"""
    
    def __init__(self, d_model=None, rank=None, dropout=0.0):
        super().__init__()

        # Dimensions & dropout
        self.d_model = d_model
        self.rank = rank
        self.dropout = dropout

        # LoRA parameters
        self.down_proj = nn.Linear(self.d_model, self.rank)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.rank, self.d_model)

        # Parameter initialization
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)


    def forward(self, x):

        z = self.down_proj(x)
        z = self.non_linear_func(z)
        z = self.up_proj(z)

        return z