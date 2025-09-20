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
        self.noise_std = 1 / num_experts 
        self.gating_layer = nn.Linear(input_dim, num_experts, bias=False)

    def _importance_auxiliary_loss(self, gates_softmax_per_item):
        importance_per_expert = gates_softmax_per_item.sum(dim=0)
        mean_importance = importance_per_expert.mean()
        std_importance = importance_per_expert.std()
        return (std_importance / (mean_importance + 1e-6)) ** 2

    def forward(self, x):
        logits = self.gating_layer(x)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits += noise
        
        gates_softmax = F.softmax(logits, dim=-1)
        importance_loss = self._importance_auxiliary_loss(gates_softmax)
        top_k_gates, top_k_indices = torch.topk(gates_softmax, self.k, dim=-1)
        combine_weights = top_k_gates 
        metrics = {'auxiliary_loss': importance_loss}
        return combine_weights, top_k_indices, metrics

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
        batch_size, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim) 
        num_tokens = x_flat.size(0)
        assert x_flat.size(-1) == self.router.gating_layer.in_features, (
            f"Router in_features ({self.router.gating_layer.in_features}) != token dim ({x_flat.size(-1)})"
        )

        # 1. Routing
        combine_weights, top_k_indices, l_auxi = self.router(x_flat)
        
        # 2. Buffer Capacity
        buffer_capacity = round((self.k * num_tokens * capacity_ratio) / self.num_experts)

        # 3. Dispatch mask and capacity enforcement with priority
        dispatch_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).to(x.dtype)  # (num_tokens, k, num_experts)
        
        # Priority-based capacity enforcement (vanilla routing approximation)
        remaining_capacity = torch.full((self.num_experts,), buffer_capacity, device=x.device, dtype=torch.long)
        within_capacity_mask = torch.zeros(num_tokens, self.k, device=x.device, dtype=torch.bool)
        
        for slot in range(self.k):  # Process higher priority slots first
            slot_experts = top_k_indices[:, slot]  # (num_tokens,)
            # For each slot, assign in token order (approximates paper; for exact, sort by gate value)
            position_in_expert = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)
            for t in range(num_tokens):
                e = slot_experts[t].item()
                if position_in_expert[e] < buffer_capacity and remaining_capacity[e] > 0:
                    within_capacity_mask[t, slot] = True
                    position_in_expert[e] += 1
                    remaining_capacity[e] -= 1
        
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
        expert_indices = top_k_indices[active_mask]  # (num_active,)
        
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
        
        # 6. Auxiliary losses
        metrics = {}
        if self.training:  # Use self.training for consistency
            load_per_expert = tokens_per_expert.float()
            l_load = (torch.std(load_per_expert) / (torch.mean(load_per_expert) + 1e-6)) ** 2
            l_imp = l_auxi['auxiliary_loss']
            l_aux = (l_imp + l_load) * 0.5  # Average, but paper multiplies by lambda outside
            metrics['auxiliary_loss'] = l_aux
            metrics['load_balance'] = l_load

        out = weighted_expert_outputs.reshape(batch_size, seq_len, dim)
        return out, metrics
