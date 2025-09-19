# vision_transformer_moe_refactored.py
# Adapted to mirror the structure and features of the V-MoE JAX repository.
# FIX: Corrected TypeError and implemented proper ensemble routing.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed

# ==================================================================
# 1. Core MoE Components (mirroring vmoe/nn/routing.py)
# ==================================================================
class EncoderMoe(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_hidden_dim, moe_config=None):
        super().__init__()
        moe_config = moe_config or {}
        moe_layers = moe_config.get('layers', ())
        
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i in moe_layers:
                mlp_block = SparseMoE(
                    input_dim=dim,
                    hidden_dim=moe_config.get('mlp_dim', mlp_hidden_dim),
                    num_experts=moe_config.get('num_experts', 4),
                    k=moe_config.get('router', {}).get('num_selected_experts', 1),
                    router_config=moe_config.get('router', {})
                )
            else:
                class MlpWrapper(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
                    def forward(self, x):
                        return self.mlp(x), {}
                mlp_block = MlpWrapper()

            self.blocks.append(TransformerBlock(dim, num_heads, mlp_block))

    def forward(self, x, is_training):
        all_metrics = []
        for blk in self.blocks:
            
            x, metrics = blk(x,is_training)
            all_metrics.append(metrics)
        
        if any(all_metrics):
            aux_loss = torch.stack([m.get('auxiliary_loss', torch.tensor(0.0, device=x.device)) for m in all_metrics if m]).sum()
            final_metrics = {'auxiliary_loss': aux_loss}
        else:
            final_metrics = {'auxiliary_loss': torch.tensor(0.0, device=x.device)}
        return x, final_metrics
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_block):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = mlp_block

    def forward(self, x, is_training):
        x = x + self.attn(*[self.norm1(x)]*3)[0]
        if isinstance(self.mlp,SparseMoE):
            ffn_out, metrics = self.mlp(self.norm2(x),is_training)
        else:
            ffn_out, metrics = self.mlp(self.norm2(x))
        x = x + ffn_out
        return x, metrics
class NoisyTopExpertsPerItemRouter(nn.Module):
    def __init__(self, input_dim, num_experts, num_selected_experts, noise_std=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.k = num_selected_experts
        self.noise_std = noise_std
        self.gating_layer = nn.Linear(input_dim, num_experts, bias=False)

    def _importance_auxiliary_loss(self, gates_softmax_per_item):
        importance_per_expert = gates_softmax_per_item.sum(dim=0)
        mean_importance = importance_per_expert.mean()
        std_importance = importance_per_expert.std()
        return (std_importance / (mean_importance + 1e-6)) ** 2

    def forward(self, x):
        logits = self.gating_layer(x)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std / self.num_experts
            logits += noise
        
        gates_softmax = F.softmax(logits, dim=-1)
        importance_loss = self._importance_auxiliary_loss(gates_softmax)
        top_k_gates, top_k_indices = torch.topk(gates_softmax, self.k, dim=-1)
        combine_weights = F.softmax(top_k_gates, dim=-1)
        metrics = {'auxiliary_loss': importance_loss}
        return combine_weights, top_k_indices, metrics

class SparseMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, k=2, router_config=None):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.input_dim = input_dim

        self.experts = nn.ModuleList([
            Mlp(in_features=input_dim, hidden_features=hidden_dim, act_layer=nn.GELU) 
            for _ in range(self.num_experts)
        ])
        
        router_args = (router_config or {}).copy()
        # Tham số k đã được truyền trực tiếp, nên có thể loại bỏ khỏi config
        router_args.pop('num_selected_experts', None) 
        
        self.router = NoisyTopExpertsPerItemRouter(
            input_dim=input_dim, 
            num_experts=self.num_experts, 
            num_selected_experts=k, 
            **router_args
        )

    def forward(self, x, is_training, capacity_ratio=1.05):
        batch_size, seq_len, dim = x.shape
        # (N*P, D) trong đó N là batch_size, P là seq_len
        x_flat = x.reshape(-1, dim) 
        num_tokens = x_flat.size(0)

        # 1. Định tuyến (Routing)
        # combine_weights: trọng số để kết hợp đầu ra của expert
        # top_k_indices: chỉ số của các expert được chọn cho mỗi token
        # router_logits: logits thô từ router, dùng để tính loss phụ
        combine_weights, top_k_indices, l_auxi = self.router(x_flat)
        
        # 2. Tính toán dung lượng bộ đệm (Buffer Capacity)
        # Công thức từ paper [2]
        # P trong paper là số token/ảnh, ở đây ta gộp N*P = num_tokens
        buffer_capacity = round((self.k * num_tokens * capacity_ratio) / self.num_experts)

        # 3. Phân phối token vào các expert với giới hạn dung lượng
        expert_outputs_flat = torch.zeros_like(x_flat)
        
        # Tạo one-hot mask cho các expert được chọn
        # shape: (num_tokens, k, num_experts)
        dispatch_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).to(x.dtype)

        # Đếm số lượng token được gán cho mỗi expert
        # shape: (num_experts)
        tokens_per_expert = torch.sum(dispatch_mask.sum(dim=1), dim=0)

        # Xác định các token nào được giữ lại (không vượt quá capacity)
        # Đây là phần logic phức tạp để mô phỏng việc "bỏ" các token dư thừa
        # shape: (num_tokens, k)
        position_in_expert_buffer = torch.cumsum(dispatch_mask, dim=0) * dispatch_mask
        
        # Giữ lại token nếu vị trí của nó trong buffer <= buffer_capacity
        within_capacity_mask = (position_in_expert_buffer <= buffer_capacity).all(dim=-1)
        
        # Áp dụng mask
        dispatch_mask *= within_capacity_mask.unsqueeze(-1)
        combine_weights = combine_weights*within_capacity_mask
        
        # 4. Xử lý qua các expert (vector hóa thay vì dùng vòng lặp)
        # Gom các token theo expert đã chọn
        # shape: (num_tokens * k, dim)
        expert_inputs = torch.einsum('tki,td->tkd', dispatch_mask, x_flat)
        expert_inputs = expert_inputs.reshape(-1, dim)
        
        # Chạy qua tất cả các expert cùng lúc
        # expert_inputs được sắp xếp sao cho num_tokens đầu tiên dành cho expert 0,
        # num_tokens tiếp theo cho expert 1, v.v.
        expert_outputs = torch.empty_like(expert_inputs)
        for i in range(self.num_experts):
            start = i * num_tokens
            end = (i + 1) * num_tokens
            # Chỉ xử lý các token có dispatch_mask > 0
            idx = torch.where(expert_inputs[start:end].sum(dim=1) != 0)
            if len(idx) > 0:
                expert_outputs[start:end][idx] = self.experts[i](expert_inputs[start:end][idx])

        expert_outputs = expert_outputs.reshape(num_tokens, self.k, dim)
        
        # 5. Kết hợp kết quả
        # Trọng số hóa và cộng dồn kết quả từ các expert
        # shape: (num_tokens, dim)
        weighted_expert_outputs = torch.einsum('tk,tkd->td', combine_weights, expert_outputs)
        
        # Cộng vào tensor đầu ra
        expert_outputs_flat += weighted_expert_outputs

        # 6. Tính toán các hàm loss phụ để cân bằng tải
        metrics = {}
        if is_training:
            load_per_expert = tokens_per_expert
            l_load = (torch.std(load_per_expert) / torch.mean(load_per_expert))**2
            l_imp = l_auxi['auxiliary_loss']
            l_aux = (l_imp + l_load) / 2.0
            metrics['l_aux'] = l_aux
            metrics['load_balance'] = l_load # Để theo dõi

        out = expert_outputs_flat.reshape(batch_size, seq_len, dim)
        return out, metrics
# ==================================================================
# 2. Modular Transformer Components (mirroring vmoe/nn/vit_moe.py)
# ==================================================================

# class SparseMoE(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_experts=4, k=1, router_config=None):
#         super().__init__()
#         self.num_experts = num_experts
#         self.k = k
#         self.experts = nn.ModuleList([
#             Mlp(in_features=input_dim, hidden_features=hidden_dim, act_layer=nn.GELU) 
#             for _ in range(num_experts)
#         ])
        
#         router_args = (router_config or {}).copy()
#         router_args.pop('num_selected_experts', None)
        
#         self.router = NoisyTopExpertsPerItemRouter(
#             input_dim=input_dim, 
#             num_experts=num_experts, 
#             num_selected_experts=k, 
#             **router_args
#         )

#     def forward(self, x):
#         batch_size, seq_len, dim = x.shape
#         x_flat = x.reshape(-1, dim)
        
#         combine_weights, top_k_indices, metrics = self.router(x_flat)
        
#         expert_outputs_flat = torch.zeros_like(x_flat)
#         dispatch_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).permute(2, 0, 1)

#         for i in range(self.num_experts):
#             token_indices, choice_indices = torch.where(dispatch_mask[i])
#             if token_indices.shape[0] > 0:
#                 expert_input = x_flat[token_indices]
#                 expert_output = self.experts[i](expert_input)
#                 weights = combine_weights[token_indices, choice_indices].unsqueeze(1)
#                 expert_outputs_flat.index_add_(0, token_indices, expert_output * weights)
        
#         out = expert_outputs_flat.reshape(batch_size, seq_len, dim)
#         return out, metrics

# ==================================================================
# 3. Main Model Architectures (mirroring vmoe/nn/models.py)
# ==================================================================

class VisionTransformerMoe(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=128,
                 depth=4, num_heads=4, mlp_hidden_dim=256, encoder_config=None):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=3, embed_dim=dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.encoder = EncoderMoe(depth, dim, num_heads, mlp_hidden_dim, moe_config=encoder_config.get('moe', {}))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, is_training):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x, metrics = self.encoder(x,is_training)
        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits, metrics

# ==================================================================
# 4. ENSEMBLE VARIANT (mirroring vmoe/nn/vit_moe_ensemble.py)
# ==================================================================

class NoisyTopExpertsPerItemEnsembleRouter(NoisyTopExpertsPerItemRouter):
    """Ensemble router. It partitions experts for different ensemble members."""
    def __init__(self, input_dim, num_experts, num_selected_experts, ensemble_size, **kwargs):
        super().__init__(input_dim, num_experts, num_selected_experts, **kwargs)
        self.ensemble_size = ensemble_size
        assert num_experts % ensemble_size == 0
        self.experts_per_member = num_experts // ensemble_size

    def forward(self, x):
        num_tokens_total, _ = x.shape
        num_tokens_orig = num_tokens_total // self.ensemble_size
        
        logits = self.gating_layer(x)
        logits_reshaped = logits.view(num_tokens_orig, self.ensemble_size, self.num_experts)
        
        mask = torch.full_like(logits_reshaped, -1e9)
        for i in range(self.ensemble_size):
            start, end = i * self.experts_per_member, (i + 1) * self.experts_per_member
            mask[:, i, start:end] = 0
        
        logits_masked = (logits_reshaped + mask).view(num_tokens_total, self.num_experts)
        
        # Now, continue with the standard noisy top-k logic on the masked logits
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits_masked) * self.noise_std / self.num_experts
            logits_masked += noise
            
        gates_softmax = F.softmax(logits_masked, dim=-1)
        importance_loss = self._importance_auxiliary_loss(gates_softmax)
        top_k_gates, top_k_indices = torch.topk(gates_softmax, self.k, dim=-1)
        combine_weights = F.softmax(top_k_gates, dim=-1)
        
        metrics = {'auxiliary_loss': importance_loss}
        return combine_weights, top_k_indices, metrics

class SparseMoEEnsemble(SparseMoE):
    """SparseMoE layer for the ensemble model that uses the ensemble router."""
    def __init__(self, input_dim, hidden_dim, num_experts=4, k=1, router_config=None, ensemble_size=1):
        # We can't call super().__init__ because it creates the wrong router.
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            Mlp(in_features=input_dim, hidden_features=hidden_dim, act_layer=nn.GELU)
            for _ in range(num_experts)
        ])
        router_args = (router_config or {}).copy()
        router_args.pop('num_selected_experts', None)
        self.router = NoisyTopExpertsPerItemEnsembleRouter(
            input_dim=input_dim, num_experts=num_experts, num_selected_experts=k,
            ensemble_size=ensemble_size, **router_args
        )

class EncoderMoeEnsemble(EncoderMoe):
    """Encoder that repeats the input and uses the specialized ensemble MoE blocks."""
    def __init__(self, depth, dim, num_heads, mlp_hidden_dim, moe_config=None):
        nn.Module.__init__(self)
        moe_config = moe_config or {}
        moe_layers = set(moe_config.get('layers', ()))
        self.ensemble_size = moe_config.get('ensemble_size', 1)
        
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i in moe_layers:
                mlp_block = SparseMoEEnsemble(
                    input_dim=dim, hidden_dim=moe_config.get('mlp_dim', mlp_hidden_dim),
                    num_experts=moe_config.get('num_experts', 4),
                    k=moe_config.get('router', {}).get('num_selected_experts', 1),
                    router_config=moe_config.get('router', {}),
                    ensemble_size=self.ensemble_size
                )
            else:
                class MlpWrapper(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
                    def forward(self, x): return self.mlp(x), {}
                mlp_block = MlpWrapper()
            self.blocks.append(TransformerBlock(dim, num_heads, mlp_block))

    def forward(self, x):
        all_metrics = []
        is_first_moe_layer = True
        for i, blk in enumerate(self.blocks):
            # The check `isinstance(blk.mlp, (SparseMoEEnsemble))` ensures this logic
            # only triggers in the ensemble model.
            if isinstance(blk.mlp, (SparseMoEEnsemble)) and is_first_moe_layer:
                x = x.repeat_interleave(self.ensemble_size, dim=0)
                is_first_moe_layer = False
            x, metrics = blk(x)
            all_metrics.append(metrics)

        if any(all_metrics):
            aux_loss = torch.stack([m.get('auxiliary_loss', torch.tensor(0.0, device=x.device)) for m in all_metrics if m]).sum()
            final_metrics = {'auxiliary_loss': aux_loss}
        else:
            final_metrics = {'auxiliary_loss': torch.tensor(0.0, device=x.device)}
        return x, final_metrics

class VisionTransformerMoeEnsemble(VisionTransformerMoe):
    """Ensemble version of the Vision Transformer with MoE."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        encoder_config = kwargs.get('encoder_config', {})
        self.encoder = EncoderMoeEnsemble(
            depth=kwargs['depth'], dim=kwargs['dim'], num_heads=kwargs['num_heads'],
            mlp_hidden_dim=kwargs['mlp_hidden_dim'], moe_config=encoder_config.get('moe', {})
        )

# ==================================================================
# 5. Ensemble Evaluation Logic (mirroring vmoe/evaluate/ensemble.py)
# ==================================================================
def ensemble_softmax_xent_eval(logits, labels, ensemble_size):
    """Evaluation loss for the ensemble model."""
    logits = logits.view(-1, ensemble_size, logits.shape[-1])
    log_p = F.log_softmax(logits, dim=-1)
    log_p_ensemble = torch.logsumexp(log_p, dim=1) - torch.log(torch.tensor(ensemble_size))
    # Note: Assumes labels are one-hot encoded
    loss = -(labels * log_p_ensemble).sum(dim=-1).mean()
    return loss