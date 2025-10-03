import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from utils import router_gating_linear, sinkhorn, MoEAuxLossAutoScaler, apply_router_token_dropping, apply_random_logits,compute_routing_scores_for_aux_loss
from loss import switch_load_balancing_loss_func, z_loss_func, topk_routing_with_score_function
from typing import Tuple
import torch.nn.functional as F

class Router(ABC, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.layer_number = None

        self.weight = nn.Parameter(
            torch.empty((self.num_experts, config.hidden_size), dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.config.perform_initialization:
            self.config.init_method(self.weight)

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        logits = router_gating_linear(input, self.weight, input.dtype)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def forward(self, input: torch.Tensor):
        raise NotImplementedError

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number

class NoisyTopItemsPerExpertRouter(nn.Module):
    """
    Expert-choice router (items-per-expert) for single GPU with auxiliary loss.
    Each expert selects top-C tokens to process.
    Returns:
        probs: softmax probabilities [batch, seq, num_experts]
        routing_map: binary assignments [batch, seq, num_experts]
        aux_loss: sum of importance + load loss
    """

    def __init__(self, config, C: int = 1, noise_std: float = 1.0, deterministic: bool = False, normalize_weights: bool = True):
        super().__init__()
        self.num_experts = config.num_moe_experts
        self.hidden_size = config.hidden_size
        self.C = C
        self.noise_std = noise_std
        self.deterministic = deterministic
        self.normalize_weights = normalize_weights

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

    def _importance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        # variance of expert probabilities across tokens
        importance = probs.sum(dim=(0, 1))  # sum over batch and seq
        return (importance.std() / (importance.mean() + 1e-6)) ** 2

    def _load_loss(self, routing_map: torch.Tensor) -> torch.Tensor:
        # variance of token assignments per expert
        load = routing_map.sum(dim=(0, 1))  # sum over batch and seq
        return (load.std() / (load.mean() + 1e-6)) ** 2

    def routing(self, logits: torch.Tensor):
        """
        logits: [batch*seq, num_experts]
        returns: probs, routing_map, aux_loss
        """
        if not self.deterministic and self.noise_std > 0.0:
            logits = logits + self.noise_std * torch.randn_like(logits)

        # Softmax probabilities per expert
        probs = F.softmax(logits, dim=-1)

        # Expert-choice: each expert selects top-C tokens
        batch_seq, num_experts = logits.shape
        routing_map = torch.zeros_like(logits)

        # Transpose: [num_experts, batch*seq]
        logits_t = logits.T
        topC_values, topC_indices = torch.topk(logits_t, k=self.C, dim=-1)  # top tokens per expert

        for expert_idx in range(num_experts):
            token_indices = topC_indices[expert_idx]  # selected token indices for this expert
            routing_map[token_indices, expert_idx] = 1.0

        # Normalize routing weights if needed
        if self.normalize_weights:
            probs = probs * routing_map  # zero out non-selected tokens per expert
            probs_sum = probs.sum(dim=-1, keepdim=True) + 1e-6
            probs = probs / probs_sum

        aux_loss = self._importance_loss(probs) + self._load_loss(routing_map)
        return probs, routing_map, aux_loss

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [batch, seq, hidden_dim]
        returns: probs [batch, seq, num_experts], routing_map [batch, seq, num_experts], aux_loss
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_input = hidden_states.view(-1, hidden_dim)  # [batch*seq, hidden_dim]
        logits = self.gate(flat_input)                   # [batch*seq, num_experts]

        probs, routing_map, aux_loss = self.routing(logits)

        # reshape back to [batch, seq, num_experts]
        probs = probs.view(batch_size, seq_len, self.num_experts)
        routing_map = routing_map.view(batch_size, seq_len, self.num_experts)

        return probs, routing_map, aux_loss

class ExpertChoiceRouting(Router):
    """Single-device Expert Choice Routing with auxiliary losses."""

    def __init__(self, config, C: int = 2, noise_std: float = 1.0, deterministic: bool = False):
        super().__init__(config)
        self.C = C  # top-C items per expert
        self.noise_std = noise_std
        self.deterministic = deterministic
        self.gate = nn.Linear(config.hidden_size, config.num_moe_experts, bias=False)

    def _importance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        importance = probs.sum(dim=0)
        return (importance.std() / (importance.mean() + 1e-6)) ** 2

    def _load_loss(self, logits: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
        load = assignments.sum(dim=0)
        return (load.std() / (load.mean() + 1e-6)) ** 2

    def routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assign tokens to experts and compute auxiliary losses.

        Returns:
            assignments: [batch_size, num_experts] binary mask
            probs: softmax probabilities
            aux_loss: combined auxiliary loss (importance + load)
        """
        if not self.deterministic and self.noise_std > 0.0:
            noise = self.noise_std * torch.randn_like(logits)
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)

        _, topC_indices = torch.topk(logits, k=self.C, dim=-1)  # top-C experts per token
        assignments = torch.zeros_like(logits)
        batch_seq_range = torch.arange(logits.size(0)).unsqueeze(1).expand(-1, self.C)
        assignments[batch_seq_range, topC_indices] = 1.0

        # _, topC_indices = torch.topk(logits, k=self.C, dim=0)
        # assignments = torch.zeros_like(logits)
        # assignments[topC_indices, torch.arange(logits.size(1))] = 1.0

        importance_loss = self._importance_loss(probs)
        load_loss = self._load_loss(logits, assignments)
        aux_loss = importance_loss + load_loss

        return probs, assignments, aux_loss

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = input.shape
        flat_input = input.view(-1, hidden_dim)  # [batch*seq, hidden]
        logits = self.gate(flat_input)           # [batch*seq, num_experts]

        probs, routing_map, aux_loss = self.routing(logits)
        return probs, routing_map, aux_loss  # shapes: [batch*seq, num_experts]

class TopKRouter(Router):
    """Route each token to the top-k experts.

    The workflow of TopKRouter is as follows:
    (1) Calculate the logits by the router gating network.
    (2) Calculate the routing probabilities and map for top-k selection with score function.
    (3) [Optional] Apply token dropping to top-k expert selection.
    (4) [Optional] Apply the auxiliary load balancing loss for the given scores and routing map.

    Naming convention:
        logits: The output logits by the router gating network.
        scores: The scores after score function used to select the experts and calculate aux loss.
        probs: The topk weights used to combined the experts' outputs.
        routing_map: The masked routing map between tokens and experts.
    """

    def __init__(
        self, 
        config=None, 
        # pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

        # Initialize global tokens per expert for global aux loss
        if self.get_aux_loss_coeff("global_aux_loss") > 0:
            self.register_buffer(
                'global_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'ga_steps',
                torch.tensor(0, dtype=torch.float32, device=torch.cuda.current_device()),
                persistent=False,
            )
        else:
            self.global_tokens_per_expert = None
            self.ga_steps = None

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def get_aux_loss_coeff(self, aux_loss_type: str) -> float:
        """Return the aux loss coeff for the given auxiliary loss type.
        If the auxiliary loss type is not found, return 0.0.
        """
        if isinstance(self.routing_type, str):
            if self.routing_type == aux_loss_type:
                return self.config.moe_aux_loss_coeff
        if isinstance(self.routing_type, list):
            try:
                idx = self.routing_type.index(aux_loss_type)
                return self.config.moe_aux_loss_coeff[idx]
            except ValueError:
                return 0.0
        return 0.0

    def is_aux_loss_enabled(self) -> bool:
        """Check if the auxiliary loss is enabled."""
        for aux_loss_type in ["aux_loss", "seq_aux_loss", "global_aux_loss"]:
            if self.get_aux_loss_coeff(aux_loss_type) > 0:
                return True
        return False

    def _apply_aux_loss(
        self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor, routing_map: torch.Tensor
    ):
        """Apply the auxiliary loss for the given scores and routing map."""
        aux_loss_coeff = self.get_aux_loss_coeff("aux_loss")
        if aux_loss_coeff == 0:
            return probs
        tokens_per_expert = routing_map.sum(dim=0)

        num_tokens = routing_map.shape[0]
        total_num_tokens = num_tokens # self.tp_cp_group.size() if using pg collections

        aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=aux_loss_coeff,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs, aux_loss_coeff, aux_loss, "load_balancing_loss"
        )
        return probs

    def _apply_seq_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        seq_length: int,
        bsz: int,
    ):
        """Apply the sequence-level auxiliary loss for the given scores and routing map.

        To calculate the sequence-level aux loss, we reshape the batch_size dimension to
        experts dimension. The resulted loss by switch_load_balancing_loss_func is equal
        to the sum of aux loss for each sequence in the batch. And then we divide the aux
        loss by the batch size to get averaged aux loss.
        """
        seq_aux_loss_coeff = self.get_aux_loss_coeff("seq_aux_loss")
        if seq_aux_loss_coeff == 0:
            return probs

        scores_for_aux_loss = scores_for_aux_loss.reshape(seq_length, -1)
        tokens_per_expert = routing_map.reshape(seq_length, -1).sum(dim=0)
        # tokens_per_expert = reduce_from_tensor_model_parallel_region(
        #     tokens_per_expert, self.tp_cp_group
        # )

        total_num_tokens = seq_length

        aux_loss = (
            switch_load_balancing_loss_func(
                probs=scores_for_aux_loss,
                tokens_per_expert=tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.topk,
                num_experts=self.config.num_moe_experts,
                moe_aux_loss_coeff=seq_aux_loss_coeff,
            )
            / bsz
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs, seq_aux_loss_coeff, aux_loss, "seq_load_balancing_loss"
        )
        return probs

    def _apply_global_aux_loss(
        self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor, routing_map: torch.Tensor
    ):
        """Apply the global auxiliary loss for the given scores and routing map."""
        global_aux_loss_coeff = self.get_aux_loss_coeff("global_aux_loss")
        if global_aux_loss_coeff == 0:
            return probs

        tokens_per_expert = routing_map.sum(dim=0)

        self.global_tokens_per_expert += tokens_per_expert
        self.ga_steps += 1
        averated_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps

        num_tokens = scores_for_aux_loss.shape[0]
        total_num_tokens = num_tokens

        global_aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=averated_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=global_aux_loss_coeff,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs,
            global_aux_loss_coeff,
            global_aux_loss,
            "global_load_balancing_loss",
            # self.tp_dp_cp_group,
        )
        return probs

    def attach_and_log_load_balancing_loss(
        self,
        activation: torch.Tensor,
        aux_loss_coeff: float,
        aux_loss: torch.Tensor,
        aux_loss_name: str,
        # reduce_group: torch.distributed.ProcessGroup,
    ):
        """Attach aux loss function to activation and add to logging."""
        # TODO (zijiey): fix the per_layer_logging for MTP, currently it will incorrectly
        # add the aux loss logging value to other layer's since it is difficult to get the
        # correct layer_number for MTP. It does not affect the correctness of the calculation
        # results and the reduced load_balancing_loss logging value.
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        # save_to_aux_losses_tracker(
        #     aux_loss_name,
        #     aux_loss / aux_loss_coeff,
        #     self.layer_number,
        #     num_layers,
        #     # reduce_group=reduce_group,
        # )
        if self.calculate_per_token_loss:
            # Scale the aux_loss by the number of tokens.
            # The expected final scaling for aux_loss gradients is 1/(num_micro_batches * dp_size).
            # After commit 02648000, Megatron started using the number of total tokens to scale
            # gradients under the argument of calculate_per_token_loss,
            # which scales both the main_loss gradient and aux_loss gradient by
            # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads function.
            # To correct this scaling, we need to scale the aux_loss by num_local_tokens here.
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss * activation.shape[0])
        else:
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training and torch.is_grad_enabled():
            # Skip Z loss calculations when using torch.no_grad() or checkpointing.
            moe_z_loss_coeff = self.config.moe_z_loss_coeff / self.tp_cp_group.size()
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            scale_up = 1.0
            if self.calculate_per_token_loss:
                # The expected final scaling for z_loss gradients is
                # 1/(num_micro_batches * dp_size).
                # After commit 02648000, Megatron started using the number of total tokens
                # to scale gradients under the argument of calculate_per_token_loss,
                # which scales both the main_loss gradient and z_loss gradient by
                # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads().
                # To correct this scaling, we need to scale the z_loss by num_local_tokens here.
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss * logits.shape[0])
            else:
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss)

            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers
            # save_to_aux_losses_tracker(
            #     "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, num_layers
            # )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): Probabilities of token→expert assignment.
            routing_map (torch.Tensor): Token→expert assignment mask [num_tokens, num_experts].
            aux_loss (torch.Tensor or None): Scalar auxiliary loss for load balancing.
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        # Calculate probs and routing_map for token dispatching
        if self.routing_type == "sinkhorn":
            probs, routing_map = self.sinkhorn_load_balancing(logits)
        else:
            probs, routing_map = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
            )

        # Apply token dropping
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        aux_loss = None
        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            # Compute scores for aux loss
            routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
                logits, self.topk, self.score_function
            )

            # Collect all aux losses
            aux_losses = []

            # Standard load balancing
            before = probs
            probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
            if not torch.equal(before, probs):
                aux_losses.append(getattr(self, "last_aux_loss", None))

            # Sequence-level
            before = probs
            probs = self._apply_seq_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
            )
            if not torch.equal(before, probs):
                aux_losses.append(getattr(self, "last_seq_aux_loss", None))

            # Global
            before = probs
            probs = self._apply_global_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss
            )
            if not torch.equal(before, probs):
                aux_losses.append(getattr(self, "last_global_aux_loss", None))

            # Reduce to a single scalar if possible
            aux_losses = [a for a in aux_losses if a is not None]
            if len(aux_losses) > 0:
                aux_loss = torch.stack(aux_losses).sum()

        # Update stats
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)

        return probs, routing_map, aux_loss

    def reset_global_aux_loss_tracker(self):
        """Reset the global aux loss tracker."""
        if self.global_tokens_per_expert is not None:
            self.global_tokens_per_expert.zero_()
            self.ga_steps.zero_()

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            probs (torch.Tensor), routing_map (torch.Tensor), aux_loss (torch.Tensor or None)
        """
        self._maintain_float32_expert_bias()

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        if self.config.moe_router_force_load_balancing:
            logits = apply_random_logits(logits)

        probs, routing_map, aux_loss = self.routing(logits)
        return probs, routing_map, aux_loss

    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def _save_to_state_dict(self, *args, **kwargs):
        """Save the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before saving
        return super()._save_to_state_dict(*args, **kwargs)



