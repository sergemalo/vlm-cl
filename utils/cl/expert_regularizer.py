import torch
from utils.cl.mlp_with_moe import MLPWithMoE

class ExpertRegularizer:
    """
    Collects snapshots and Fisher information only from old experts
    across all MLPWithMoE layers in the model.
    """
    def __init__(self, model, old_task_dataloader,
                 lambda_reg=100.0, mode="ewc", device="cpu"):
        self.lambda_reg = lambda_reg
        self.mode = mode
        self.device = device

        # Snapshot only old expert params, identified by requires_grad=False
        # (they were frozen before — now we unfreeze but remember who they are)
        self.anchors, self.param_refs = self._snapshot_old_experts(model)

        if mode == "ewc":
            self.fisher = self._compute_fisher(model, old_task_dataloader)


    def _snapshot_old_experts(self, model):
        """
        Walk all MLPWithMoE layers, snapshot old expert parameters.
        Old experts are identified by being frozen (requires_grad=False) at snapshot time,
        then unfrozen afterward for regularized finetuning.
        """
        anchors = {}        # name → frozen clone
        param_refs = {}     # name → live parameter (for penalty computation)

        for module_name, module in model.named_modules():
            if not isinstance(module, MLPWithMoE):
                continue

            moe = module.moe
            for expert_idx, expert in enumerate(moe.experts):
                # Old experts were frozen — use that as the identifier
                is_old = not next(expert.parameters()).requires_grad
                if not is_old:
                    continue

                for param_name, param in expert.named_parameters():
                    full_name = f"{module_name}.moe.experts.{expert_idx}.{param_name}"
                    anchors[full_name] = param.detach().clone()
                    param_refs[full_name] = param

                # Now unfreeze for regularized finetuning
                for param in expert.parameters():
                    param.requires_grad = True

        return anchors, param_refs


    def _compute_fisher(self, model, dataloader):
        fisher = {name: torch.zeros_like(anchor)
                for name, anchor in self.anchors.items()}

        model.eval()
        for inputs in dataloader:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            model.zero_grad()
            outputs = model(**inputs)  # loss computed internally from labels
            outputs.loss.backward()

            for name, param in self.param_refs.items():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        for name in fisher:
            fisher[name] /= len(dataloader)

        model.train()
        return fisher


    def penalty(self):
        """Call this in the training loop and add to task loss."""
        loss = torch.tensor(0.0, device=self.device)

        for name, param in self.param_refs.items():
            anchor = self.anchors[name]
            deviation = (param - anchor).pow(2)

            if self.mode == "ewc":
                loss += (self.fisher[name] * deviation).sum()
            else:                           # simple L2 anchor
                loss += deviation.sum()

        return self.lambda_reg * loss
    

"""
Example usage (Claude):

# --- After finishing Task N, before introducing new experts ---
regularizer = ExpertRegularizer(
    model=model,                            # full model with MLPWithMoE layers
    old_task_dataloader=task_n_dataloader,
    criterion=criterion,
    lambda_reg=100.0,
    mode="ewc",                             # or "l2" for simpler regularization
    device=device
)
# Old experts are now unfrozen inside _snapshot_old_experts

# --- Training on Task N+1 ---
for inputs, targets in task_n1_dataloader:
    optimizer.zero_grad()

    outputs = model(inputs)
    task_loss = criterion(outputs, targets)
    reg_loss = regularizer.penalty()        # only penalizes old expert drift

    loss = task_loss + reg_loss
    loss.backward()
    optimizer.step()
"""