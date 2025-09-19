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
                 num_experts: int = 4,
                 k: int = 1):
        super().__init__()
        self.moe = SparseMoE(
            input_dim=hidden_size,
            hidden_dim=intermediate_size,
            num_experts=num_experts,
            k=k
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _metrics = self.moe(x, is_training=self.training)
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

        # Load base ViT model
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # Replace selected encoder blocks with MoE feed-forward
        for i, block in enumerate(self.model.vit.encoder.layer):
            if i in moe_layers:
                hidden = block.intermediate.dense.in_features
                inter  = block.intermediate.dense.out_features
                block.intermediate = MoeFFN(hidden, inter,
                                            num_experts=num_experts,
                                            k=top_k)
                # the output dense after the MLP is redundant when using MoE
                block.output.dense = nn.Identity()

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        return self.model(pixel_values, labels=labels)