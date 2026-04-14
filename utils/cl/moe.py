import torch
import torch.nn as nn
from utils.cl.adapter import Adapter

class MoEAdapter(nn.Module):
    """Class for Mixture-of-Expert"""

    def __init__(self, d_model, new_expert_count=2, rank=16, top_k=2, old_experts=None, old_routers=None, 
                 mode="train", level_id=None, anneal_steps=5000, init_bonus=3.0):
        super().__init__()

        self.new_expert_count = new_expert_count
        self.rank = rank
        self.top_k = top_k
        self.mode = mode
        self.level_id = level_id

        self._weight_tracker = {}

        # ----- Experts -----
        old_experts = old_experts or []

        # Freeze old Experts by default (not the case when fixed set)
        if mode == "train":
            for expert in old_experts:
                for param in expert.parameters():
                    param.requires_grad = False

        # Keep old Experts, add the new Experts for this task
        new_experts = [
            Adapter(d_model=d_model, rank=self.rank)
            for _ in range(new_expert_count)
        ]
        self.experts = nn.ModuleList(old_experts + new_experts)
        for expert in new_experts:
            for param in expert.parameters():
                param.requires_grad = True

        # ----- Routers -----
        old_routers = old_routers or []

        # Freeze old Routers (never used during training)
        for router in old_routers:
            for param in router.parameters():
                param.requires_grad = False

        # Initialize new Router and add it to the list of Routers
        if ("train" in mode):
            new_router = [nn.Linear(d_model, len(old_experts) + new_expert_count)] # has requires_grad by default
        else:
            new_router = []

        self.routers = nn.ModuleList(old_routers + new_router)

        # Cold start Routing-related hyperparameters
        self.old_expert_count = len(old_experts)
        self.init_bonus = init_bonus
        self.anneal_steps = anneal_steps
        self.register_buffer("current_step", torch.tensor(0))

    
    def _get_boosted_logits(self, logits):
        """
        Boost new Expert logits early in training, annealing to 0.
        Old experts are still selectable — we're leveling the field, not excluding them.
        """
        self.current_step += 1
        progress = min(self.current_step.item() / self.anneal_steps, 1.0)
        bonus = self.init_bonus * (1.0 - progress)  # linear decay to 0

        boosted = logits.clone()
        boosted[..., self.old_expert_count:] = boosted[..., self.old_expert_count:] + bonus
        return boosted
    

    def _init_tracker(self, level_id):
        """Used when monitoring the importance of old vs new Experts"""
        if level_id not in self._weight_tracker:
            self._weight_tracker[level_id] = {
                "old": {"sum": 0.0, "count": 0},
                "new": {"sum": 0.0, "count": 0},
            }

    def reset_routing_stats(self, level_id=None):
        """Clear accumulators for one level (or all levels if level_id is None)."""
        if level_id is None:
            self._weight_tracker.clear()
        elif level_id in self._weight_tracker:
            del self._weight_tracker[level_id]


    def get_routing_stats(self, level_id=None):
        """
        Returns {"old_avg": float, "new_avg": float} for the given level_id
        (defaults to self.level_id).  Returns None for a bucket with no data.
        """
        lid = level_id if level_id is not None else self.level_id
        if lid not in self._weight_tracker:
            return None
        t = self._weight_tracker[lid]
        old_avg = t["old"]["sum"] / t["old"]["count"] if t["old"]["count"] else 0.0
        new_avg = t["new"]["sum"] / t["new"]["count"] if t["new"]["count"] else 0.0
        return {"old_avg": old_avg, "new_avg": new_avg}
    

    def measure_routing_importance(self, topk_vals, topk_idx):
        """
        Measure importance of Old vs New Experts
        Importance is measure by sum of normalized top-k logits over each group
        """
        with torch.no_grad():
            self._init_tracker(self.level_id)
            t = self._weight_tracker[self.level_id]

            # For each token, sum the gate weights going to old vs new experts
            # topk_idx: [B, S, top_k], topk_vals: [B, S, top_k]
            is_new = (topk_idx > 2 + 2*self.level_id)          
            
            new_weight = (topk_vals * is_new).sum(dim=-1)       
            old_weight = (topk_vals * ~is_new).sum(dim=-1)      

            t["new"]["sum"]   += new_weight.sum().item()
            t["new"]["count"] += new_weight.numel()
            t["old"]["sum"]   += old_weight.sum().item()
            t["old"]["count"] += old_weight.numel()
    

    def forward(self, x):
        # Training mode: Oracle (the input belongs to the new task)
        if "train" in self.mode:
            router = self.routers[-1]
        
        # Inference: Classifier (the classifier outputs the corresponding level_id)
        else:
            router = self.routers[self.level_id]

        # Weight all of the Experts
        logits = router(x)
        if self.mode == "train":
            logits = self._get_boosted_logits(logits) # to avoid Expert collapse (only with growing set)
        gates = torch.softmax(logits, dim=-1)

        # Find the top-k Experts
        topk_vals, topk_idx = gates.topk(self.top_k, dim=-1)

        # Re-normalize top-k weights so they sum to 1
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
        
        # Process input with top-k Experts
        output = torch.zeros_like(x)
        expert_for_task_count = router.out_features          
        for expert_id in range(expert_for_task_count):       
            expert = self.experts[expert_id]

            # Check if this Expert is in the top-k of any input token
            mask = (topk_idx == expert_id)
            if not mask.any():
                continue

            expert_out = expert(x) 

            # Sum the gate weights for this expert across top_k slots
            weight = (topk_vals * mask).sum(dim=-1) 
            output += weight.unsqueeze(-1) * expert_out

        # Keep track of the Expert importance
        if self.mode == "eval_weight_tracker":
            self.measure_routing_importance(topk_vals, topk_idx)
        
        return output