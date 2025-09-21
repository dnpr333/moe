# corrected_router_moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp


class NoisyTopExpertsPerItemRouter(nn.Module):
    def __init__(self, input_dim, num_experts, num_selected_experts):
        super().__init__()
        self.num_experts = num_experts
        self.k = max(1, num_selected_experts)
        self.noise_std = 1 / max(num_experts, 1)
        self.gating_layer = nn.Linear(input_dim, num_experts, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gating_layer.weight)

    def _importance_auxiliary_loss(self, gates_softmax_per_item):
        importance_per_expert = gates_softmax_per_item.sum(dim=0)
        mean_imp = importance_per_expert.mean()
        std_imp = importance_per_expert.std(unbiased=False)  # stable with single element -> 0
        return (std_imp / (mean_imp + 1e-6)) ** 2

    def forward(self, x):
        """
        x: (num_tokens, dim)
        returns:
          combine_weights: (num_tokens, k)  # float
          top_k_indices:   (num_tokens, k)  # long
          metrics: dict with 'importance_loss'
        """
        logits = self.gating_layer(x)  # (num_tokens, num_experts)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        gates_softmax = F.softmax(logits, dim=-1)  # (num_tokens, num_experts)

        if self.num_experts == 1:
            # deterministic, all tokens go to expert 0
            top_k_indices = torch.zeros(x.size(0), 1, dtype=torch.long, device=x.device)
            combine_weights = torch.ones(x.size(0), 1, dtype=x.dtype, device=x.device)
        else:
            topk_vals, topk_idx = torch.topk(gates_softmax, self.k, dim=-1)  # (num_tokens, k), (num_tokens, k)
            combine_weights = topk_vals
            top_k_indices = topk_idx

        metrics = {'importance_loss': self._importance_auxiliary_loss(gates_softmax)}
        return combine_weights, top_k_indices, metrics


class SparseMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, k=2, expert_dropout=0.0, router_config=None):
        super().__init__()
        self.num_experts = max(1, num_experts)
        self.k = min(k, self.num_experts)
        self.input_dim = input_dim

        # experts: timm.Mlp(in_features=input_dim, hidden_features=hidden_dim)
        self.experts = nn.ModuleList([
            Mlp(in_features=input_dim, hidden_features=hidden_dim, act_layer=nn.GELU, drop=expert_dropout)
            for _ in range(self.num_experts)
        ])

        r_args = (router_config or {}).copy()
        r_args.pop('num_selected_experts', None)
        self.router = NoisyTopExpertsPerItemRouter(input_dim=input_dim,
                                                   num_experts=self.num_experts,
                                                   num_selected_experts=self.k,
                                                   **r_args)

    def forward(self, x, capacity_ratio=1.05):
        # x: (batch, seq_len, dim)
        bsz, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)  # (num_tokens, dim)
        num_tokens = x_flat.size(0)

        # 1) routing
        combine_weights, top_k_indices, router_metrics = self.router(x_flat)
        
        # combine_weights: (num_tokens, k), top_k_indices: (num_tokens, k)

        # 2) buffer capacity
        buffer_cap = round((self.k * num_tokens * capacity_ratio) / max(self.num_experts, 1))

        # trivial fast path for single expert
        if self.num_experts == 1:
            expert_out = self.experts[0](x_flat)            # (num_tokens, dim)
            out = expert_out.view(bsz, seq_len, dim)
            # metrics: router importance_loss is safe (should be zero)
            return out, {'auxiliary_loss': router_metrics.get('importance_loss', torch.tensor(0., device=x.device))}

        # 3) dispatch mask / capacity enforcement
        dispatch_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).to(x.dtype)  # (num_tokens, k, num_experts)
        device = x.device
        remaining_capacity = torch.full((self.num_experts,), buffer_cap, device=device, dtype=torch.long)
        within_capacity_mask = torch.zeros(num_tokens, self.k, device=device, dtype=torch.bool)

        # iterate by slot then expert (num_experts and k small)
        for slot in range(self.k):
            slot_experts = top_k_indices[:, slot]  # (num_tokens,)
            for e in range(self.num_experts):
                if remaining_capacity[e] <= 0:
                    continue
                idxs = (slot_experts == e).nonzero(as_tuple=True)[0]
                if idxs.numel() == 0:
                    continue
                take_n = int(min(idxs.numel(), int(remaining_capacity[e].item())))
                if take_n == 0:
                    continue
                take = idxs[:take_n]
                within_capacity_mask[take, slot] = True
                remaining_capacity[e] -= take_n

        # apply masks
        dispatch_mask = dispatch_mask * within_capacity_mask.unsqueeze(-1).to(dispatch_mask.dtype)
        combine_weights = combine_weights * within_capacity_mask.to(combine_weights.dtype) 
        active_sums = combine_weights.sum(dim=1, keepdim=True)
        combine_weights = torch.where(active_sums > 0, combine_weights / active_sums, combine_weights)

        # tokens per expert (for load loss)
        tokens_per_expert = dispatch_mask.sum(dim=[0, 1]).float()  # (num_experts,)

        # 4) dispatch to experts
        active_mask = within_capacity_mask  # (num_tokens, k)
        token_indices = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, self.k)[active_mask]  # (num_active,)
        slot_indices = torch.arange(self.k, device=device).unsqueeze(0).expand(num_tokens, -1)[active_mask]     # (num_active,)
        expert_indices = top_k_indices[active_mask]  # (num_active,)

        expert_inputs = x_flat[token_indices]  # (num_active, dim)

        # process each expert
        expert_outputs_flat = torch.zeros_like(expert_inputs)
        for e in range(self.num_experts):
            mask_e = (expert_indices == e)
            if mask_e.sum() > 0:
                inputs_e = expert_inputs[mask_e]
                outs_e = self.experts[e](inputs_e)
                expert_outputs_flat[mask_e] = outs_e

        # 5) scatter back
        expert_outputs = torch.zeros(num_tokens, self.k, dim, device=device)
        flat_indices = token_indices * self.k + slot_indices  # indices in flattened (num_tokens*k)
        expert_outputs.view(-1, dim)[flat_indices] = expert_outputs_flat

        # weighted sum: combine_weights (num_tokens,k), expert_outputs (num_tokens,k,dim) -> (num_tokens, dim)
        weighted_expert_outputs = torch.einsum('tk,tkd->td', combine_weights, expert_outputs)

        # 6) auxiliary losses
        l_load = (torch.std(tokens_per_expert, unbiased=False) / (tokens_per_expert.mean() + 1e-6)) ** 2
        l_aux = 0.5 * (router_metrics.get('importance_loss', torch.tensor(0., device=device)) + l_load)

        out = weighted_expert_outputs.view(bsz, seq_len, dim)
        return out, {'auxiliary_loss': l_aux, 'load_balance': l_load}
