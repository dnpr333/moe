import torch
from typing import List

def permute(x, local_map, num_out_tokens=None, fused=False):
    """
    x: [num_tokens, hidden_dim]
    local_map: [num_tokens, num_local_experts] (binary mask)
    """
    # Convert binary mask -> expert indices
    expert_idx = local_map.argmax(dim=-1)  # [num_tokens]

    # Sort tokens by expert
    sorted_expert_idx, idx = torch.sort(expert_idx)
    permuted = x.index_select(0, idx)

    if num_out_tokens is None:
        num_out_tokens = permuted.size(0)

    reverse_idx = torch.argsort(idx)
    return permuted, None, reverse_idx

def unpermute(x, reverse_idx, restore_shape=None, routing_map=None, fused=False):
    out = x.index_select(0, reverse_idx)
    if restore_shape is not None:
        out = out.view(restore_shape)
    return out

class MoETokenDispatcher:
    """
    All tokens and experts live on the same device, so no communication needed.
    """

    def __init__(self, 
                 config=None):
        self.config = config
        self.shared_experts = None

    def dispatch_preprocess(self, tokens, routing_map, probs):
        # Just return inputs as-is
        return tokens, probs

    def token_dispatch(self, hidden_states, probs):
        # No dispatch needed, everything stays local
        return hidden_states, probs

    def dispatch_postprocess(self, hidden_states, probs):
        # Return tokens and probs directly
        return hidden_states, None, probs

    def combine_preprocess(self, hidden_states):
        # No preprocessing needed
        return hidden_states

    def token_combine(self, hidden_states):
        # Nothing to combine across devices, return as-is
        return hidden_states

    def combine_postprocess(self, hidden_states):
        # No reordering needed
        return hidden_states

    def set_shared_experts(self, shared_experts):
        self.shared_experts = shared_experts

class MoEAllGatherTokenDispatcher:
    """
    Simplified Token Dispatcher for single-device MoE.
    Just reshapes, permutes tokens for local experts, and restores them.
    """

    def __init__(self, num_local_experts: int=0, local_expert_indices: List[int]=None, config=None):
        self.num_local_experts = num_local_experts
        assert self.num_local_experts > 0, "Expected at least one expert"

        self.local_expert_indices = local_expert_indices
        assert len(self.local_expert_indices) > 0, "Expected at least one local expert index"

        self.config = config
        self.router_topk = config.moe_router_topk
        self.add_bias = config.add_bias_linear

        # Cached tensors from dispatch/combine phases
        self.hidden_shape = None
        self.local_map = None
        self.local_probs = None
        self.reversed_local_input_permutation_mapping = None

    def dispatch_preprocess(self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor):
        """Flatten hidden states and cache routing map."""
        self.hidden_shape = hidden_states.shape
        # [S, B, H] -> [S*B, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self.routing_map = routing_map
        return hidden_states, probs

    def token_dispatch(self, hidden_states, probs):
        return hidden_states, probs

    def dispatch_postprocess(self, hidden_states, probs):
        """Select and permute tokens for local experts."""
        self.hidden_shape_before_permute = hidden_states.shape

        # Mask for local experts
        self.local_map = self.routing_map[:, self.local_expert_indices[0]:self.local_expert_indices[-1] + 1].contiguous()
        self.local_probs = probs[:, self.local_expert_indices[0]:self.local_expert_indices[-1] + 1].contiguous()

        tokens_per_expert = self.local_map.sum(dim=0).long().cpu()

        # print(f"x: {hidden_states.shape}")
        permuted_local_hidden_states, _, self.reversed_local_input_permutation_mapping = permute(
            hidden_states,
            self.local_map,
            num_out_tokens=tokens_per_expert.sum(),
            fused=self.config.moe_permute_fusion,
        )

        # Select probs for permuted tokens
        self.local_probs = self.local_probs.T.contiguous().masked_select(self.local_map.T.contiguous())
        self.routing_map = None

        # hidden_states: [num_tokens, hidden_dim]
        assignments = self.local_map.nonzero(as_tuple=False)  # (token_idx, expert_idx)

        # Expand tokens: duplicate if top-k > 1
        expanded_hidden_states = hidden_states[assignments[:, 0]]

        # Count tokens per expert
        tokens_per_expert = torch.bincount(assignments[:, 1], minlength=len(self.local_expert_indices))

        return expanded_hidden_states, tokens_per_expert, assignments[:, 1]

        # return permuted_local_hidden_states, tokens_per_expert, self.local_probs

    def combine_preprocess(self, hidden_states):
        """Undo permutation before combining expert outputs."""
        return unpermute(
            hidden_states,
            self.reversed_local_input_permutation_mapping,
            # restore_shape=self.hidden_shape_before_permute,
            routing_map=self.local_map,
            fused=self.config.moe_permute_fusion,
        )

    def token_combine(self, hidden_states):
        """No-op for single device (no reduce-scatter)."""
        return hidden_states

    def combine_postprocess(self, hidden_states):
        """
        Aggregate expert outputs back to original sequence.
        hidden_states: [num_assignments, hidden_dim]
        self.local_map: [num_tokens, num_experts] (0/1 routing map)
        self.hidden_shape: original shape [B, S, H]
        """
        B, S, H = self.hidden_shape
        num_tokens = B * S

        # Flatten original space
        combined = torch.zeros(num_tokens, H, device=hidden_states.device)

        # token assignments
        idx = self.local_map.nonzero(as_tuple=False)[:, 0]  # [num_assignments]

        # scatter-add expert outputs into token slots
        combined.index_add_(0, idx, hidden_states)

        # reshape back
        return combined.view(B, S, H)


