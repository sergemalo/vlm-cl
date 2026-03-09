import torch
import torch.nn as nn
from adapter import Adapter

class MoEAdapter(nn.Module):
    """Class for Mixture-of-Expert"""

    def __init__(self, d_model, num_experts=8, rank=16, top_k=2, dropout=0.0, 
                 existing_experts=None, existing_routers=None, mode="train",
                 anneal_steps=5000, init_bonus=3.0):
        """
        existing_experts:           Experts initialized previously to be re-used 
                                    (i.e. we want to train for a more complex task and re-use experts
                                    defined previously)
        
        existing_routers:           Routers initialized for the previous-tasks
                                    (the Experts are in the same order for every task, but each Router
                                    can only select the Experts that were defined for this task, i.e.
                                    the self.num_experts is the number of indices the Router can output
                                    but it changes for every task)

        mode:                       Used to select the appropriate Router

        anneal_steps/init_bonus:    Used to guide routing towards new Experts early on
        """
        super().__init__()

        # Expert-related hyperparameters
        self.num_experts = num_experts
        self.top_k = top_k
        self.mode = mode

        # Cold start Routing-related hyperparameters
        self.num_old_experts = len(existing_experts or [])
        self.init_bonus = init_bonus
        self.anneal_steps = anneal_steps
        self.register_buffer("current_step", torch.tensor(0))

        # ----- Experts -----
        existing_experts = existing_experts or []
        assert len(existing_experts) <= num_experts, "More existing experts than num_experts"

        # Freeze old experts by default
        # If used, ExpertRegularizer (see expertregularizer.py) will un-freeze and apply Parameter Regularization
        for expert in existing_experts:
            for param in expert.parameters():
                param.requires_grad = False

        # Keep old experts, add the new Experts for this task
        new_experts = [
            Adapter(d_model=d_model, bottleneck=rank, dropout=dropout)
            for _ in range(num_experts - len(existing_experts))
        ]
        self.experts = nn.ModuleList(existing_experts + new_experts)

        # ----- Routers -----
        existing_routers = existing_routers or []

        # Freeze old Routers (technically unecessary as never used during training)
        for router in existing_routers:
            for param in router.parameters():
                param.requires_grad = False

        # Initialize new Router and add it to the list of Routers
        new_router = nn.Linear(d_model, self.num_experts)
        self.routers = nn.ModuleList(existing_routers + [new_router])

    
    def _get_boosted_logits(self, logits):
        """
        Boost new expert logits early in training, annealing to 0.
        Old experts are still selectable — we're leveling the field, not excluding them.
        """
        if not self.training or self.num_old_experts == 0:
            return logits

        progress = min(self.current_step.item() / self.anneal_steps, 1.0)
        bonus = self.init_bonus * (1.0 - progress)  # linear decay to 0

        boosted = logits.clone()
        boosted[..., self.num_old_experts:] = boosted[..., self.num_old_experts:] + bonus
        return boosted
    

    def forward(self, x):
        # x: [batch, seq, d_model] (# sequences, # tokens/sequence, dimension of each token)

        # Select the appropriate router
        # During training mode: Router is the new one (all inputs belong to new task)
        if self.mode == "train":
            router = self.routers[-1]
        
        # At inference, figure out the appropriate Router (batch of 1)
        else:
            router_idx = -1 # TODO: At inference, find the appropriate router (probably simple classifier)
            router = self.routers[router_idx]

        # Weight all of the Experts
        logits = router(x)
        logits = self._get_boosted_logits(logits) # to avoid Expert collapse
        gates = torch.softmax(logits, dim=-1)

        # Find the top-k Experts
        topk_vals, topk_idx = gates.topk(self.top_k, dim=-1)

        # Re-normalize top-k weights so they sum to 1
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        # Process input with top-k Experts
        output = torch.zeros_like(x)
        num_experts_for_task = router.out_features          # The number of Experts changes depending on the task
        for expert_id in range(num_experts_for_task):       # only iterate over visible experts
            expert = self.experts[expert_id]

            # Check if this Expert is in the top-k of any input token
            mask = (topk_idx == expert_id)  # [batch, seq, top_k]
            if not mask.any():
                continue

            expert_out = expert(x)  # [batch, seq, d_model]

            # Sum the gate weights for this expert across top_k slots
            weight = (topk_vals * mask).sum(dim=-1)  # [batch, seq]
            output += weight.unsqueeze(-1) * expert_out

        return output
    

"""
Example usage across training tasks:

# Task 1
moe_v1 = MoEAdapter(d_model, num_experts=4)

# Task 2
moe_v2 = MoEAdapter(d_model, num_experts=8,
                    existing_experts=list(moe_v1.experts),
                    existing_routers=list(moe_v1.routers))

# Task 3
moe_v3 = MoEAdapter(d_model, num_experts=12,
                    existing_experts=list(moe_v2.experts),
                    existing_routers=list(moe_v2.routers))
"""