import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Optional, Union
from dataclasses import dataclass

from router import TopKRouter
from dispatcher import MoETokenDispatcher, MoEAllGatherTokenDispatcher
from utils import build_module
from expert import SequentialMLP

@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[nn.Module, type] = None
    shared_experts: Union[nn.Module, type] = None

class BaseMoELayer(nn.Module, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config=None,
        # config: TransformerConfig,
        layer_number: Optional[int] = None,
        # pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super(BaseMoELayer, self).__init__()   
        self.config = config
        self.layer_number = layer_number
        # self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        # self.attn_tp_group = pg_collection.tp
        # ep_size = utils.get_pg_size(self.ep_group)
        # ep_rank = utils.get_pg_rank(self.ep_group)
        # assert ep_size > 0, "Expected non-negative expert parallel size"

        # assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        # self.local_expert_indices = [
        #     local_expert_indices_offset + i for i in range(self.num_local_experts)
        # ]

        self.local_expert_indices = list(range(self.config.num_moe_experts))

        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None

        self.experts = SequentialMLP(
            num_local_experts=self.num_local_experts,
            config=self.config,
            # submodules=self.submodules
        )

        self.shared_experts = None
        # self.token_dispatcher: Optional[MoETokenDispatcher] = None
        
        self.token_dispatcher = MoEAllGatherTokenDispatcher(config=self.config,
                                            num_local_experts=self.num_local_experts,
                                            local_expert_indices=self.local_expert_indices)
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)

class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config=None,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        # pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.

        super(MoELayer, self).__init__(
            config=config, layer_number=layer_number
        )
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )

        # Initialize router
        self.router = TopKRouter(config=self.config)

        # Initialize token dispatcher

        # Initialize experts
        self.experts = SequentialMLP(
            num_local_experts=self.num_local_experts,
            config=self.config,
            # submodules=self.submodules
        )
        # self.experts = build_module(
        #     self.submodules.experts,
        #     self.num_local_experts,
        #     self.config,
        #     # pg_collection=pg_collection,
        # )


        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """Compute and preprocess token routing for dispatch.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping. It then preprocesses the
        hidden states and probabilities for the token dispatcher. The original
        hidden states are returned as a residual connection.
        """
        residual = hidden_states
        probs, routing_map = self.router(hidden_states)
        probs, _ = self.router(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs, residual

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.
        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        # return hidden_states, probs
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def experts_compute(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, residual: torch.Tensor
    ):
        """Computes the output of the experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        If a shared expert is configured and not overlapped with communication,
        it is also applied. The output from the experts is preprocessed for the
        combine step.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            shared_expert_output = self.shared_experts(residual)

        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        # dispatched_input, tokens_per_expert, permuted_probs = hidden_states, None, probs

        # print(type(self.experts))
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        # assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)
        output = expert_output

        return output, shared_expert_output, mlp_bias

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication). It then adds the output
        from the shared expert if it exists.
        """
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states):
            hidden_states, probs, residual = self.router_and_preprocess(hidden_states)
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            output, shared_expert_output, mlp_bias = self.experts_compute(
                dispatched_input, probs, residual
            )
            output = self.combine(output, shared_expert_output)
            return output, mlp_bias

        output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()