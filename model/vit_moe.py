# vit_moe.py
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import timm  # [CHANGE] Import timm instead of transformers

# The vmoe_base.SparseMoE is assumed to be in model/vmoe_base.py
# If it's not, you might need to adjust the import path.
# For this example to be self-contained, I will add a placeholder for SparseMoE.
# --- Placeholder for SparseMoE ---
class DummyExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class SparseMoE(nn.Module):
    """A placeholder implementation of SparseMoE for demonstration."""
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int, k: int, expert_dropout: float):
        super().__init__()
        self.k = k
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            DummyExpert(input_dim, hidden_dim) for _ in range(num_experts)
        ])
        # In a real implementation, you'd have more sophisticated gating and dispatching logic.

    def forward(self, x: torch.Tensor, capacity_ratio: float = 2.0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # This is a highly simplified forward pass for demonstration.
        # A real implementation would involve token dispatching, load balancing, etc.
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        
        gate_logits = self.gate(x_flat)
        top_k_weights, top_k_indices = torch.topk(gate_logits, self.k, dim=-1)
        top_k_weights = torch.softmax(top_k_weights, dim=-1)

        # Simple weighted sum of the top-k experts
        output_flat = torch.zeros_like(x_flat)
        for i in range(self.k):
            indices = top_k_indices[:, i]
            weights = top_k_weights[:, i].unsqueeze(-1)
            
            # This is inefficient but illustrates the concept.
            # A real implementation uses gather/scatter for efficiency.
            for exp_idx in range(len(self.experts)):
                mask = (indices == exp_idx)
                if mask.any():
                    output_flat[mask] += weights[mask] * self.experts[exp_idx](x_flat[mask])
        
        output = output_flat.view(batch_size, seq_len, dim)
        
        # Simulate auxiliary loss (load balancing loss)
        aux_loss = (gate_logits.std() / gate_logits.mean())**2 # A simple proxy for load balancing
        metrics = {"auxiliary_loss": aux_loss}
        
        return output, metrics
# --- End of Placeholder ---


def _get_expert_fc_layers(expert: nn.Module) -> Tuple[nn.Linear, nn.Linear]:
    """
    Returns the (fc1, fc2) pair in the expert.
    Prioritizes attribute names fc1, fc2 (timm.Mlp). If not found, takes the first and last linear layers.
    (This function is already compatible with timm's Mlp, no changes needed)
    """
    if hasattr(expert, "fc1") and hasattr(expert, "fc2"):
        return expert.fc1, expert.fc2
    # fallback: get all nn.Linear in module traversal order, use first & last
    linears = [m for m in expert.modules() if isinstance(m, nn.Linear)]
    if len(linears) >= 2:
        return linears[0], linears[-1]
    raise RuntimeError("Could not find 2 nn.Linear layers in the expert; update the layer retrieval logic for your expert.")


def copy_mlp_to_experts(moe_ffn: nn.Module, dense_intermediate: nn.Linear, dense_output: nn.Linear, verbose: bool = False):
    """
    Copies weights/bias from dense_intermediate (hidden -> intermediate)
    and dense_output (intermediate -> hidden) to all experts in moe_ffn.moe.experts.
    Checks shapes before copying to avoid errors.
    (No changes needed here)
    """
    # Clone source tensors (safe if the source is later replaced)
    w1 = dense_intermediate.weight.data.clone()
    b1 = dense_intermediate.bias.data.clone() if dense_intermediate.bias is not None else None
    w2 = dense_output.weight.data.clone()
    b2 = dense_output.bias.data.clone() if dense_output.bias is not None else None

    # Copy to each expert
    for ei, expert in enumerate(moe_ffn.moe.experts):
        fc1, fc2 = _get_expert_fc_layers(expert)

        # Check shapes: fc1.weight.shape == w1.shape, fc2.weight.shape == w2.shape
        if fc1.weight.shape != w1.shape:
            raise RuntimeError(f"Shape mismatch expert[{ei}].fc1 {tuple(fc1.weight.shape)} != dense_intermediate {tuple(w1.shape)}")
        if fc2.weight.shape != w2.shape:
            raise RuntimeError(f"Shape mismatch expert[{ei}].fc2 {tuple(fc2.weight.shape)} != dense_output {tuple(w2.shape)}")

        # Copy with device/dtype conversion
        fc1.weight.data.copy_(w1.to(fc1.weight.device, fc1.weight.dtype))
        if b1 is not None:
            if fc1.bias is None:
                raise RuntimeError(f"Expert[{ei}].fc1 has no bias but source dense_intermediate has bias.")
            fc1.bias.data.copy_(b1.to(fc1.bias.device, fc1.bias.dtype))

        fc2.weight.data.copy_(w2.to(fc2.weight.device, fc2.weight.dtype))
        if b2 is not None:
            if fc2.bias is None:
                raise RuntimeError(f"Expert[{ei}].fc2 has no bias but source dense_output has bias.")
            fc2.bias.data.copy_(b2.to(fc2.bias.device, fc2.bias.dtype))

        if verbose:
            print(f"[copy] expert[{ei}] <- dense (fc1 {tuple(fc1.weight.shape)}, fc2 {tuple(fc2.weight.shape)})")


class MoeFFN(nn.Module):
    """
    Replaces the entire MLP (hidden -> intermediate -> hidden) with a MoE layer.
    Each expert is timm.Mlp-like (fc1: hidden->intermediate, fc2: intermediate->hidden).
    (No changes needed here)
    """
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_experts: int,
                 k: int = 1,
                 expert_dropout: float = 0.0,
                 capacity_ratio: float = 2):
        super().__init__()
        self.capacity_ratio = capacity_ratio
        # **Note**: input_dim = hidden_size, hidden_dim = intermediate_size
        # to match timm.Mlp(in_features=hidden_size, hidden_features=intermediate_size)
        self.moe = SparseMoE(
            input_dim=hidden_size,
            hidden_dim=intermediate_size,
            num_experts=num_experts,
            k=k,
            expert_dropout=expert_dropout
        )
        self.last_auxiliary_loss: Optional[torch.Tensor] = None

    def init_from_dense(self, dense_intermediate: nn.Linear, dense_output: nn.Linear, verbose: bool = False):
        """
        Copies the two dense layers (dense_intermediate, dense_output) into all experts.
        Call this after creating MoeFFN and before installing it in the model.
        """
        copy_mlp_to_experts(self, dense_intermediate, dense_output, verbose=verbose)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, metrics = self.moe(x, capacity_ratio=self.capacity_ratio)
        aux = metrics.get("auxiliary_loss", 0.0)
        if not torch.is_tensor(aux):
            aux = torch.tensor(aux, device=out.device, dtype=out.dtype)
        else:
            aux = aux.to(out.device)
        self.last_auxiliary_loss = aux.detach()
        return out


class ViTMOE(nn.Module):
    """
    [MAJOR CHANGE]
    Wrapper: loads a timm ViT, replaces selected transformer blocks' MLP by MoeFFN,
    and copies pretrained MLP weights into every expert (upcycling).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # [CHANGE] Use a timm model name as default
        model_name     = config.get("model_name", "vit_base_patch16_224")
        num_classes    = config.get("num_classes", 100)
        moe_layers     = config.get("moe_layers", [])   
        num_experts    = config.get("num_experts", 4)
        top_k          = config.get("top_k", 1)
        expert_dropout = config.get("expert_dropout", 0.0)
        capacity_ratio = config.get("capacity_ratio", 2)
        verbose_copy   = config.get("verbose_copy", False)

        # [CHANGE] Load pretrained ViT using timm.create_model
        # `pretrained=True` loads weights from the default source (e.g., ImageNet-21k or ImageNet-1k)
        # `num_classes` automatically replaces the classification head.
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )

        # [CHANGE] Replace designated layers by iterating through `self.model.blocks`
        for i, block in enumerate(self.model.blocks):
            if i in moe_layers:
                # [CHANGE] Access the MLP block and its linear layers (fc1, fc2)
                # In timm, the entire MLP is conveniently packaged in `block.mlp`
                original_mlp = block.mlp
                dense_intermediate = original_mlp.fc1
                dense_output = original_mlp.fc2

                hidden_size = dense_intermediate.in_features
                intermediate_size = dense_intermediate.out_features
                moe_ffn = MoeFFN(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    k=top_k,
                    expert_dropout=expert_dropout,
                    capacity_ratio=capacity_ratio
                )

                # Copy pretrained MLP weights into all experts
                moe_ffn.init_from_dense(dense_intermediate, dense_output, verbose=verbose_copy)

                # [CHANGE] Replace the entire mlp block. This is cleaner than the HF version.
                block.mlp = moe_ffn

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # [CHANGE] The forward pass is modified to match timm's output and to
        # manually construct a dictionary similar to Hugging Face's output.

        # Reset any stored aux losses
        for block in self.model.blocks:
            if isinstance(block.mlp, MoeFFN):
                block.mlp.last_auxiliary_loss = None

        # timm models return logits directly
        logits = self.model(pixel_values)

        # Collect auxiliary losses from the MoE layers
        aux_losses = []
        for block in self.model.blocks:
            if isinstance(block.mlp, MoeFFN) and block.mlp.last_auxiliary_loss is not None:
                aux_losses.append(block.mlp.last_auxiliary_loss)

        # Create an output dictionary to hold results
        outputs = {"logits": logits}
        
        if aux_losses:
            outputs["auxiliary_loss"] = torch.stack(aux_losses).mean()

        # If labels are provided, calculate the main loss (mimicking HF behavior)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            main_loss = loss_fct(logits.view(-1, self.model.num_classes), labels.view(-1))
            outputs["loss"] = main_loss
            
        return outputs