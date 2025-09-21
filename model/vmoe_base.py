# vision_transformer_moe_refactored.py
# Adapted to mirror the structure and features of the V-MoE JAX repository.
# FIX: Corrected TypeError and implemented proper ensemble routing.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed

    
class NoisyTopExpertsPerItemRouter(nn.Module):
    def __init__(self, input_dim, num_experts, num_selected_experts):
        super().__init__()
        self.num_experts = num_experts
        self.k = num_selected_experts
        self.noise_std = 1 / max(num_experts, 1)
        self.gating_layer = nn.Linear(input_dim, num_experts, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gating_layer.weight)

    def _importance_auxiliary_loss(self, gates_softmax_per_item):
        importance_per_expert = gates_softmax_per_item.sum(dim=0)
        mean_imp = importance_per_expert.mean()
        std_imp  = importance_per_expert.std(unbiased=False)
        return (std_imp / (mean_imp + 1e-6)) ** 2

    def forward(self, x):
        logits = self.gating_layer(x)
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std

        gates_softmax = F.softmax(logits, dim=-1)

        # Single-expert special case: no top-k needed, all weight to expert 0
        if self.num_experts == 1:
            top_k_indices = torch.zeros(x.size(0), 1, dtype=torch.long, device=x.device)
            top_k_gates   = torch.ones_like(top_k_indices, dtype=x.dtype)
        else:
            top_k_gates, top_k_indices = torch.topk(gates_softmax, self.k, dim=-1)

        metrics = {'importance_loss': self._importance_auxiliary_loss(gates_softmax)}
        return top_k_gates, top_k_indices, metrics
    
    
class SparseMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, k=2,expert_dropout = 0.0, router_config=None):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.input_dim = input_dim

        self.experts = nn.ModuleList([
            Mlp(in_features=input_dim, hidden_features=hidden_dim, act_layer=nn.GELU, drop=expert_dropout) 
            for _ in range(self.num_experts)
        ])
        
        router_args = (router_config or {}).copy()
        # Tham số k đã được truyền trực tiếp, nên có thể loại bỏ khỏi config
        router_args.pop('num_selected_experts', None) 
        
        self.router = NoisyTopExpertsPerItemRouter(
            input_dim=input_dim, 
            num_experts=self.num_experts, 
            num_selected_experts=k, 
            **router_args
        )
    def forward(self, x, capacity_ratio=1.05):
        bsz, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)

        # 1. Routing
        combine_w, top_k_idx, router_metrics = self.router(x_flat)

        # 2. Capacity & dispatch
        num_tokens = x_flat.size(0)
        buffer_cap = round((self.k * num_tokens * capacity_ratio) /
                           max(self.num_experts, 1))
        dispatch_mask = F.one_hot(top_k_idx,
                                  num_classes=self.num_experts).to(x.dtype)

        # trivial path if only one expert
        if self.num_experts == 1:
            expert_outputs = self.experts[0](x_flat)
            out = expert_outputs.view(bsz, seq_len, dim)
            # no balancing loss needed
            return out, {'auxiliary_loss': router_metrics['importance_loss']}
        
        # Priority-based capacity enforcement (vectorized, device-safe)
        device = x.device
        remaining_capacity = torch.full((self.num_experts,), buffer_cap, device=device, dtype=torch.long)
        within_capacity_mask = torch.zeros(num_tokens, self.k, device=device, dtype=torch.bool)

        # Loop over slots then experts (num_experts and k small)
        for slot in range(self.k):
            slot_experts = top_k_idx[:, slot]  # (num_tokens,) on device
            # For each expert, take up to remaining_capacity[e] tokens in the order they appear
            for e in range(self.num_experts):
                if remaining_capacity[e] <= 0:
                    continue
                # indices of tokens that requested expert e at this slot
                idxs = (slot_experts == e).nonzero(as_tuple=True)[0]  # stays on device
                if idxs.numel() == 0:
                    continue
                # choose first N tokens up to remaining_capacity
                take = idxs[:remaining_capacity[e].item()] if isinstance(remaining_capacity[e], torch.Tensor) else idxs[:remaining_capacity[e]]
                if take.numel() == 0:
                    continue
                within_capacity_mask[take, slot] = True
                # decrement remaining capacity
                remaining_capacity[e] -= take.numel()

        
        # Apply masks
        dispatch_mask *= within_capacity_mask.unsqueeze(-1).to(dispatch_mask.dtype)
        combine_weights *= within_capacity_mask.to(combine_weights.dtype)
        
        # Compute tokens_per_expert after masking for losses
        tokens_per_expert = dispatch_mask.sum(dim=[0, 1])  # (num_experts,)
        
        # 4. Dispatch inputs to experts
        # Find active dispatches
        active_mask = within_capacity_mask  # (num_tokens, k)
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, self.k)[active_mask]  # (num_active,)
        slot_indices = torch.arange(self.k, device=x.device).unsqueeze(0).expand(num_tokens, -1)[active_mask]  # (num_active,)
        expert_indices = top_k_idx[active_mask]  # (num_active,)
        
        # Gather inputs
        expert_inputs = x_flat[token_indices]  # (num_active, dim)
        
        # Process per expert
        expert_outputs_flat = torch.zeros_like(expert_inputs)  # (num_active, dim)
        for i in range(self.num_experts):
            expert_mask = (expert_indices == i)
            if expert_mask.sum() > 0:
                inputs_i = expert_inputs[expert_mask]
                outputs_i = self.experts[i](inputs_i)
                expert_outputs_flat[expert_mask] = outputs_i
        
        # 5. Combine results
        # Scatter back to (num_tokens, k, dim)
        expert_outputs = torch.zeros(num_tokens, self.k, dim, device=x.device)
        # Compute flat indices for scatter
        flat_indices = token_indices * self.k + slot_indices
        expert_outputs.view(-1, dim)[flat_indices] = expert_outputs_flat
        
        # Weighted sum
        weighted_expert_outputs = torch.einsum('tk,tkd->td', combine_weights, expert_outputs)
        tokens_per_expert = dispatch_mask.sum(dim=[0, 1]).float()
        l_load = (torch.std(tokens_per_expert, unbiased=False) /
                  (tokens_per_expert.mean() + 1e-6)) ** 2
        l_aux = 0.5 * (router_metrics['importance_loss'] + l_load)

        out = weighted_expert_outputs.view(bsz, seq_len, dim)
        return out, {'auxiliary_loss': l_aux, 'load_balance': l_load}
