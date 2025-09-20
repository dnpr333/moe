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
                 k: int = 1):
        super().__init__()
        self.moe = SparseMoE(
            input_dim=hidden_size,
            hidden_dim=intermediate_size,
            num_experts=num_experts,
            k=k
        )
        self.last_auxiliary_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, metrics = self.moe(x, is_training=True)
        aux = 0.0
        if isinstance(metrics, dict) and "auxiliary_loss" in metrics:
            aux = metrics["auxiliary_loss"]
        if not torch.is_tensor(aux):
            aux = torch.tensor(aux, device=out.device, dtype=out.dtype)
        else:
            aux = aux.to(out.device)

        # store for later aggregation by ViTMOE
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

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        
        device = pixel_values.device if pixel_values is not None else next(self.parameters()).device

        # Clear previous stored aux losses (avoid stale values)
        for block in self.model.vit.encoder.layer:
            if isinstance(block.intermediate, MoeFFN):
                block.intermediate.last_auxiliary_loss = torch.tensor(0.0, device=device)

        # Call the HF ViT forward
        outputs = self.model(pixel_values, labels=labels)

        # Aggregate saved aux losses from all MoeFFN modules (if any)
        aux_list = []
        for block in self.model.vit.encoder.layer:
            if isinstance(block.intermediate, MoeFFN) and block.intermediate.last_auxiliary_loss is not None:
                aux_list.append(block.intermediate.last_auxiliary_loss)

        if len(aux_list) > 0:
            # average across MoE layers (you can sum instead if you prefer)
            aux_tensor = torch.stack(aux_list).mean()
            # attach to HuggingFace output object for backward compatibility
            try:
                outputs.auxiliary_loss = aux_tensor
            except Exception:
                # if it's not a dataclass just in case, attach attribute anyway
                setattr(outputs, "auxiliary_loss", aux_tensor)
        return outputs