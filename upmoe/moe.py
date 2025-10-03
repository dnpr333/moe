import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Optional, Union
from dataclasses import dataclass

from router import TopKRouter, ExpertChoiceRouting, NoisyTopItemsPerExpertRouter

@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[nn.Module, type] = None
    shared_experts: Union[nn.Module, type] = None

class BaseMoELayer(nn.Module, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config: TransformerConfig or a config object with necessary MoE params.
    """

    def __init__(
        self,
        config=None,
        layer_number: int = None,
    ):
        super(BaseMoELayer, self).__init__()
        self.config = config
        self.layer_number = layer_number

        self.num_local_experts = self.config.num_moe_experts
        self.use_shared_expert = (
            getattr(self.config, "moe_shared_expert_intermediate_size", None) is not None
        )
        self.shared_expert_overlap = getattr(self.config, "moe_shared_expert_overlap", False)

        self.local_expert_indices = list(range(self.num_local_experts))
        self.router = None
        self.experts = None
        self.shared_experts = None

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer and inform the router."""
        self.layer_number = layer_number
        if self.router is not None:
            self.router.set_layer_number(layer_number)

class MoELayer(BaseMoELayer):
    """
    MoE layer (single GPU, upcycling ViT) with expert-per-token routing and implicit aux_loss.
    """

    def __init__(self, config=None, submodules: Optional[MoESubmodules] = None, layer_number: Optional[int] = None):
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.submodules = submodules

        # Router: expert-per-token with aux_loss
        self.router = NoisyTopItemsPerExpertRouter(config=self.config)

        self.experts = submodules.experts  # nn.ModuleList([...])
        self.num_experts = len(self.experts)
        self.shared_experts = submodules.shared_experts if submodules else None

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states

        # routing returns probs, routing_map, aux_loss
        probs, routing_map, aux_loss = self.router(hidden_states)

        probs = probs.view(batch_size, seq_len, self.num_experts)
        routing_map = routing_map.view(batch_size, seq_len, self.num_experts)

        return hidden_states, probs, routing_map, residual, aux_loss

    def experts_compute(self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor, residual: torch.Tensor):
        """
        Expert-per-token routing: each token is processed by top-C experts.
        Weighted sum over assigned experts.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        outputs = torch.zeros_like(hidden_states)

        # iterate over experts for weighted sum
        for idx, expert in enumerate(self.experts):
            mask = routing_map[..., idx].bool()  # which tokens this expert is assigned
            if mask.sum() == 0:
                continue
            expert_in = hidden_states[mask]       # tokens assigned to this expert
            expert_out = expert(expert_in)        # output
            outputs[mask] += expert_out * probs[..., idx][mask].unsqueeze(-1)

        shared_expert_output = None
        if self.shared_experts is not None:
            shared_expert_output = self.shared_experts(residual)

        return outputs, shared_expert_output

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def forward(self, hidden_states: torch.Tensor):
        hidden_states, probs, routing_map, residual, aux_loss = self.router_and_preprocess(hidden_states)
        output, shared_expert_output = self.experts_compute(hidden_states, probs, routing_map, residual)
        output = self.combine(output, shared_expert_output)
        return output, aux_loss

# class MoELayer(BaseMoELayer):
#     """
#     Mixture of Experts layer (simplified for single GPU, up
#     cycling ViT).
#     - Keeps router + expert compute pipeline.
#     - Returns aux_loss separately for final loss computation.
#     """

#     def __init__(self, config=None, submodules: Optional[MoESubmodules] = None, layer_number: Optional[int] = None):
#         super(MoELayer, self).__init__(config=config, layer_number=layer_number)
#         self.submodules = submodules

#         # Router
#         self.router = ExpertChoiceRouting(config=self.config)

#         self.experts = submodules.experts  # nn.ModuleList([...])
#         self.num_experts = len(self.experts)

#         # Shared expert (optional)
#         self.shared_experts = submodules.shared_experts if submodules else None

#     def router_and_preprocess(self, hidden_states: torch.Tensor):
#         #print(f"inside router preprocess hidden states: {hidden_states.shape}")
#         batch_size, seq_len, _ = hidden_states.shape
#         residual = hidden_states

#         probs, routing_map, aux_loss = self.router(hidden_states)
#         probs = probs.view(batch_size, seq_len, self.num_experts)
#         routing_map = routing_map.view(batch_size, seq_len, self.num_experts)

#         return hidden_states, probs, routing_map, residual, aux_loss

#     def experts_compute(self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor, residual: torch.Tensor):
#         batch_size, seq_len, hidden_dim = hidden_states.shape
#         outputs = torch.zeros_like(hidden_states)

#         # Iterate over experts
#         for idx, expert in enumerate(self.experts):
#             mask = routing_map[..., idx].bool()  # [batch, seq]
#             if mask.sum() == 0:
#                 continue
#             expert_in = hidden_states[mask]  # [num_tokens, hidden_dim]
#             expert_out = expert(expert_in)   # [num_tokens, hidden_dim]
#             outputs[mask] += expert_out * probs[..., idx][mask].unsqueeze(-1)

#         shared_expert_output = None
#         if self.shared_experts is not None:
#             shared_expert_output = self.shared_experts(residual)

#         return outputs, shared_expert_output

#     def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
#         if shared_expert_output is not None:
#             output = output + shared_expert_output
#         return output

#     def forward(self, hidden_states: torch.Tensor):
#         hidden_states, probs, routing_map, residual, aux_loss = self.router_and_preprocess(hidden_states)
#         output, shared_expert_output = self.experts_compute(hidden_states, probs, routing_map, residual)
#         output = self.combine(output, shared_expert_output)
#         return output, aux_loss  # Return output for next layer and aux_loss separately
