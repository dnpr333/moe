# vit_moe.py
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from model.vmoe_base import SparseMoE


def _get_expert_fc_layers(expert: nn.Module) -> Tuple[nn.Linear, nn.Linear]:
    """
    Trả về cặp (fc1, fc2) trong expert.
    Ưu tiên attribute names fc1, fc2 (timm.Mlp). Nếu không có, lấy linear đầu và linear cuối.
    """
    if hasattr(expert, "fc1") and hasattr(expert, "fc2"):
        return expert.fc1, expert.fc2
    # fallback: lấy tất cả nn.Linear theo thứ tự duyệt module, dùng first & last
    linears = [m for m in expert.modules() if isinstance(m, nn.Linear)]
    if len(linears) >= 2:
        return linears[0], linears[-1]
    raise RuntimeError("Không tìm thấy 2 nn.Linear trong expert; cập nhật logic lấy layer cho expert của bạn.")


def copy_mlp_to_experts(moe_ffn: nn.Module, dense_intermediate: nn.Linear, dense_output: nn.Linear, verbose: bool = False):
    """
    Copy weights/bias từ dense_intermediate (hidden -> intermediate)
    và dense_output (intermediate -> hidden) sang tất cả experts trong moe_ffn.moe.experts.
    Kiểm tra shape trước khi copy để tránh lỗi.
    """
    # Clone source tensors (an toàn nếu source sau đó bị replace)
    w1 = dense_intermediate.weight.data.clone()
    b1 = dense_intermediate.bias.data.clone() if dense_intermediate.bias is not None else None
    w2 = dense_output.weight.data.clone()
    b2 = dense_output.bias.data.clone() if dense_output.bias is not None else None

    # Copy vào từng expert
    for ei, expert in enumerate(moe_ffn.moe.experts):
        fc1, fc2 = _get_expert_fc_layers(expert)

        # Kiểm tra shape: fc1.weight shape == w1.shape, fc2.weight shape == w2.shape
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
    Thay thế toàn bộ MLP (hidden -> intermediate -> hidden) bằng MoE.
    Mỗi expert là timm.Mlp-like (fc1: hidden->intermediate, fc2: intermediate->hidden).
    """
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_experts: int,
                 k: int = 1,
                 expert_dropout: float = 0.0,
                 capacity_ratio: float = 1.05):
        super().__init__()
        self.capacity_ratio = capacity_ratio
        # **Chú ý**: input_dim = hidden_size, hidden_dim = intermediate_size
        # để timm.Mlp(in_features=hidden_size, hidden_features=intermediate_size)
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
        Copy 2 dense layers (dense_intermediate, dense_output) vào tất cả experts.
        Gọi sau khi tạo MoeFFN và trước khi gắn vào model.
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
    Wrapper: load HF ViT, replace selected transformer blocks' MLP by MoeFFN,
    and copy pretrained MLP weights into every expert (upcycling).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_name     = config.get("model_name", "google/vit-base-patch16-224-in21k")
        num_classes    = config.get("num_classes", 100)
        moe_layers     = config.get("moe_layers", [])   # list of indices to replace
        num_experts    = config.get("num_experts", 4)
        top_k          = config.get("top_k", 1)
        expert_dropout = config.get("expert_dropout", 0.0)
        capacity_ratio = config.get("capacity_ratio", 1.05)
        verbose_copy   = config.get("verbose_copy", False)

        # Load pretrained ViT
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # Replace designated layers
        # for i, block in enumerate(self.model.vit.encoder.layer):
        #     if i in moe_layers:
        #         # Source dense layers (original MLP)
        #         dense_intermediate = block.intermediate.dense 
        #         dense_output = block.output.dense             # Linear(intermediate -> hidden

        #         hidden_size = dense_intermediate.in_features
        #         intermediate_size = dense_intermediate.out_features
        #         moe_ffn = MoeFFN(
        #             hidden_size=hidden_size,
        #             intermediate_size=intermediate_size,
        #             num_experts=num_experts,
        #             k=top_k,
        #             expert_dropout=expert_dropout,
        #             capacity_ratio=capacity_ratio
        #         )

        #         # Copy pretrained MLP weights into all experts
        #         moe_ffn.init_from_dense(dense_intermediate, dense_output, verbose=verbose_copy)

        #         # Replace full MLP: assign MoeFFN to block.intermediate and identity to block.output.dense
        #         block.intermediate = moe_ffn
        #         block.output.dense = nn.Identity()

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # Reset any stored aux losses
        for block in self.model.vit.encoder.layer:
            if isinstance(block.intermediate, MoeFFN):
                block.intermediate.last_auxiliary_loss = None

        outputs = self.model(pixel_values, labels=labels)

        aux_losses = []
        for block in self.model.vit.encoder.layer:
            if isinstance(block.intermediate, MoeFFN) and block.intermediate.last_auxiliary_loss is not None:
                aux_losses.append(block.intermediate.last_auxiliary_loss)

        if aux_losses:
            setattr(outputs, "auxiliary_loss", torch.stack(aux_losses).mean())

        return outputs
