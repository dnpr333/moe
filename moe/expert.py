import torch
import torch.nn as nn

class Modified_MLP(nn.Module):
    """
    Simple MLP used inside MoE experts.
    h: hidden size
    ffn_hidden_size: intermediate hidden size (usually 4*h)
    """

    def __init__(self, config, ffn_hidden_size=None, is_expert=False):
        super().__init__()
        self.config = config

        input_size = config.hidden_size
        if ffn_hidden_size is None:
            if is_expert:
                raise ValueError("Expert MLP requires ffn_hidden_size explicitly.")
            ffn_hidden_size = config.ffn_hidden_size

        # Double hidden size if using Gated Linear Unit
        if config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.fc1 = nn.Linear(input_size, ffn_hidden_size, bias=config.add_bias_linear)
        self.activation = config.activation_func
        self.fc2 = nn.Linear(ffn_hidden_size, config.hidden_size, bias=config.add_bias_linear)

    def forward(self, hidden_states, per_token_scale=None):
        x = self.fc1(hidden_states)

        # Apply activation (and GLU if enabled)
        if self.config.gated_linear_unit:
            x_glu, x_linear = torch.chunk(x, 2, dim=-1)
            x = self.activation(x_glu) * (x_linear + self.config.glu_linear_offset)
        else:
            x = self.activation(x)

        # Optional scaling per token
        if per_token_scale is not None:
            x = x * per_token_scale.unsqueeze(-1)

        output = self.fc2(x)
        return output, None  # bias handled inside Linear

class SequentialMLP(nn.Module):
    """MoE Expert layer for single-GPU setting."""

    def __init__(self, num_local_experts, config):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.local_experts = nn.ModuleList([
            Modified_MLP(config, ffn_hidden_size=config.moe_ffn_hidden_size, is_expert=True)
            for _ in range(num_local_experts)
        ])
        self.apply_probs_on_input = config.moe_apply_probs_on_input
        self.topk = config.moe_router_topk

    def forward(self, hidden_states, tokens_per_expert, probs):
        if self.apply_probs_on_input:
            assert self.topk == 1
            hidden_states = probs.unsqueeze(-1) * hidden_states
            probs = torch.ones_like(probs)

        if self.num_local_experts == 1:
            return self.local_experts[0](hidden_states, probs)

        tokens_list = torch.split(hidden_states, tokens_per_expert.tolist())
        probs_list = torch.split(probs, tokens_per_expert.tolist())

        outputs = []
        for expert, tokens, p in zip(self.local_experts, tokens_list, probs_list):
            out, bias = expert(tokens, p)
            outputs.append(out)

        return torch.cat(outputs, dim=0), None
