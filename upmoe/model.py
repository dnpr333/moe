import torch 
import torch.nn as nn
from transformers import ViTConfig, AutoModelForImageClassification
from moe import MoELayer, MoESubmodules
from copy import deepcopy
from typing import Optional
import math

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional, List

class VisionTransformerWithMoE(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_experts: int,
        moe_config: ViTConfig,
        moe_layer_indices: list[int] = None, 
    ):
        super().__init__()

        self.vit = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name,
            num_labels=100
        )
        total_blocks = len(self.vit.vit.encoder.layer)
        self.moe_layer_indices = moe_layer_indices if moe_layer_indices is not None else list(range(total_blocks))

        self.moe_layers = nn.ModuleDict()
        for i, block in enumerate(self.vit.vit.encoder.layer):
            if i not in self.moe_layer_indices:
                continue

            orig_intermediate = block.intermediate.dense
            orig_output_dense = block.output.dense
            orig_output_dropout = block.output.dropout

            # Create experts
            experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(orig_intermediate.in_features, orig_intermediate.out_features),
                    nn.GELU(),
                    nn.Linear(orig_output_dense.in_features, orig_output_dense.out_features),
                    deepcopy(orig_output_dropout)
                )
                for _ in range(num_experts)
            ])

            # Copy pretrained weights to experts
            for expert in experts:
                expert[0].weight.data.copy_(orig_intermediate.weight.data)
                expert[0].bias.data.copy_(orig_intermediate.bias.data)
                expert[2].weight.data.copy_(orig_output_dense.weight.data)
                expert[2].bias.data.copy_(orig_output_dense.bias.data)

            # Store MoE layer
            self.moe_layers[str(i)] = MoELayer(
                config=moe_config,
                submodules=MoESubmodules(experts=experts, shared_experts=None),
                layer_number=i
            )

    def forward(self, pixel_values: torch.Tensor):
        """
        Forward pass:
        - Propagate x through the ViT encoder with MoE layers.
        - Collect aux_loss from all MoE blocks.
        """
        aux_loss_total = 0.0
        x = pixel_values

        hidden_states = self.vit.vit.embeddings(x)

        for i, block in enumerate(self.vit.vit.encoder.layer):
            if str(i) in self.moe_layers:
                hidden_states, aux_loss = self.moe_layers[str(i)](hidden_states)
                aux_loss_total += aux_loss
            else:
                hidden_states = block(hidden_states)[0]  

        hidden_states = self.vit.vit.layernorm(hidden_states)

        logits = self.vit.classifier(hidden_states[:, 0, :])

        return logits, aux_loss_total

# class VisionTransformerWithMoE(nn.Module):
#     def __init__(
#         self,
#         pretrained_model_name: str,
#         num_experts: int,
#         moe_config: ViTConfig,
#         moe_layer_indices: list[int] = None, 
#     ):
#         super().__init__()
#         # Load pretrained ViT
#         self.vit = AutoModelForImageClassification.from_pretrained(
#             pretrained_model_name,
#             num_labels=100
#         )
#         self.moe_layer_indices = moe_layer_indices if moe_layer_indices is not None else list(range(len(self.vit.vit.encoder.layer)))

#         # Replace selected FFN layers with MoELayer
#         for i, block in enumerate(self.vit.vit.encoder.layer):
#             if i not in self.moe_layer_indices:
#                 continue  

#             orig_intermediate = block.intermediate.dense
#             orig_output_dense = block.output.dense
#             orig_output_dropout = block.output.dropout

#             experts = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Linear(orig_intermediate.in_features, orig_intermediate.out_features),
#                     nn.GELU(),
#                     nn.Linear(orig_output_dense.in_features, orig_output_dense.out_features),
#                     deepcopy(orig_output_dropout)
#                 )
#                 for _ in range(num_experts)
#             ])

#             # Copy pretrained weights
#             for expert in experts:
#                 expert[0].weight.data.copy_(orig_intermediate.weight.data)
#                 expert[0].bias.data.copy_(orig_intermediate.bias.data)
#                 expert[2].weight.data.copy_(orig_output_dense.weight.data)
#                 expert[2].bias.data.copy_(orig_output_dense.bias.data)

#             block.intermediate = None
#             block.output = None
#             block.moe = MoELayer(
#                 config=moe_config,
#                 submodules=MoESubmodules(experts=experts, shared_experts=None),
#                 layer_number=i
#             )

#     def forward(self, pixel_values, labels=None):
#         """
#         Forward pass:
#         - Propagate x through the ViT encoder with MoE layers.
#         - Collect aux_loss from all MoELayer blocks.
#         """
#         aux_loss_total = 0.0
#         x = pixel_values
#         hidden_states = self.vit.vit.embeddings(x)  # patch embeddings + CLS token
#         if isinstance(hidden_states, tuple):
#             hidden_states = torch.stack(hidden_states, dim=0)
#         #print(hidden_states.shape)
	
#         for i, block in enumerate(self.vit.vit.encoder.layer):
#             #print(f"layer {i} block {block}")
#             #print(f"hidden shape {hidden_states.shape}")
#             if hasattr(block, "moe") and block.moe is not None:
#                 hidden_states, aux_loss = block.moe(hidden_states)
#                 if aux_loss is not None:
#                     aux_loss_total += aux_loss
#             else:
#                 hidden_states = block(hidden_states)[0]
#                 #print(hidden_states.shape)

#         # print(f"Done forward through all layers in vit blocks")
#         if hasattr(self.vit.vit, "layernorm"):
#             hidden_states = self.vit.vit.layernorm(hidden_states)

#         # cls head
#         logits = self.vit.classifier(hidden_states[:, 0, :])  

#         return logits, aux_loss_total