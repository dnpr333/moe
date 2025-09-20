from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from model.vmoe_base import SparseMoE


class MoeFFN(nn.Module):
    """Feed-forward layer replaced by a SparseMoE."""
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_experts: int,
                 k: int = 1,
                 expert_dropout: float = 0.0,
                 capacity_ratio: float = 1.05):
        super().__init__()
        self.capacity_ratio = capacity_ratio
        self.moe = SparseMoE(
            input_dim=hidden_size,
            hidden_dim=intermediate_size,
            num_experts=num_experts,
            k=k,
            expert_dropout=expert_dropout  # Pass to SparseMoE (0.0 pretrain, 0.1 finetune)
        )
        self.last_auxiliary_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, metrics = self.moe(x, capacity_ratio=self.capacity_ratio)
        aux = metrics.get("auxiliary_loss", 0.0)
        if not torch.is_tensor(aux):
            aux = torch.tensor(aux, device=out.device, dtype=out.dtype)
        else:
            aux = aux.to(out.device)

        # Store for later aggregation by ViTMOE
        self.last_auxiliary_loss = aux.detach()
        return out


class ViTMOE(nn.Module):
    """
    Vision Transformer with optional Mixture-of-Experts layers.

    Args:
        config: dict with keys:
            - model_name: HuggingFace model id (default "google/vit-base-patch16-224-in21k")
            - num_classes: number of output classes
            - moe_layers: list of encoder layer indices to replace with MoE
            - num_experts: number of experts per MoE layer
            - top_k: how many experts to route to (k in top-k gating)
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        model_name   = config.get("model_name", "google/vit-base-patch16-224-in21k")
        num_classes  = config.get("num_classes", 100)
        moe_layers   = config.get("moe_layers", [])    
        num_experts  = config.get("num_experts", 4)
        top_k        = config.get("top_k", 1)
        is_training  = config.get("is_training",False)
        # Load base ViT model
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # Replace selected encoder blocks with MoE feed-forward
        # for i, block in enumerate(self.model.vit.encoder.layer):
        #     if i in moe_layers:
        #         hidden = block.intermediate.dense.in_features
        #         inter  = block.intermediate.dense.out_features
        #         block.intermediate = MoeFFN(hidden, inter,
        #                                     num_experts=num_experts,
        #                                     k=top_k)
        #         # the output dense after the MLP is redundant when using MoE
        #         block.output.dense = nn.Identity()

class ViTMOE(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        model_name      = config.get("model_name", "google/vit-base-patch16-224-in21k")
        num_classes     = config.get("num_classes", 100)
        moe_layers      = config.get("moe_layers", [])    
        num_experts     = config.get("num_experts", 4)
        top_k           = config.get("top_k", 1)
        expert_dropout  = config.get("expert_dropout", 0.0)
        capacity_ratio  = config.get("capacity_ratio", 1.05)

        # Load base ViT model
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # Replace selected encoder blocks with MoE feed-forward
        # for i, block in enumerate(self.model.vit.encoder.layer):
        #     if i in moe_layers:
        #         hidden = block.intermediate.dense.in_features
        #         inter  = block.intermediate.dense.out_features
        #         block.intermediate = MoeFFN(hidden, inter,
        #                                     num_experts=num_experts,
        #                                     k=top_k,
        #                                     expert_dropout=expert_dropout,
        #                                     capacity_ratio=capacity_ratio)
        #         # The output dense after the MLP is redundant when using MoE
        #         block.output.dense = nn.Identity()

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        for block in self.model.vit.encoder.layer:
            if isinstance(block.intermediate, MoeFFN):
                block.intermediate.last_auxiliary_loss = None

        # Call the HF ViT forward
        outputs = self.model(pixel_values, labels=labels)

        # Aggregate saved aux losses from all MoeFFN modules (if any)
        aux_list = []
        for block in self.model.vit.encoder.layer:
            if isinstance(block.intermediate, MoeFFN) and block.intermediate.last_auxiliary_loss is not None:
                aux_list.append(block.intermediate.last_auxiliary_loss)

        if len(aux_list) > 0:
            # Average across MoE layers (paper effectively averages via lambda scaling)
            aux_tensor = torch.stack(aux_list).mean()
            # Attach to HuggingFace output object
            setattr(outputs, "auxiliary_loss", aux_tensor)
        return outputs