# vision_transformer_moe_cifar10.py
# Based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Modified to use Vision Transformer with Mixture-of-Experts (MoE) FFN.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# MoE Implementation
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        return self.ffn(x)

class SparseMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, k=1, noisy_gating=True):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        batch_size, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)  # (B*L, D)

        # Router
        logits = self.router(x_flat)  # (B*L, num_experts)
        if self.noisy_gating:
            logits = logits + torch.randn_like(logits) * 1e-2

        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
        gates = F.softmax(topk_vals, dim=-1)

        out = torch.zeros_like(x_flat)
        for i in range(self.k):
            expert_idx = topk_idx[:, i]
            for j, expert in enumerate(self.experts):
                mask = (expert_idx == j).float().unsqueeze(-1)
                if mask.sum() > 0:
                    out += gates[:, i].unsqueeze(-1) * expert(x_flat * mask)

        return out.reshape(batch_size, seq_len, dim)

# Transformer Block with MoE
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, num_experts=4, k=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = SparseMoE(input_dim=dim, hidden_dim=mlp_hidden_dim,
                             num_experts=num_experts, k=k)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        moe_out = self.moe(x)
        x = self.norm2(x + moe_out)
        return x

# Vision Transformer with MoE
class VisionTransformerMoE(nn.Module):
    def __init__(self, image_size=32, patch_size=4, dim=128,
                 depth=4, num_heads=4, mlp_hidden_dim=256,
                 num_classes=10, num_experts=4, k=1):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_hidden_dim,
                             num_experts=num_experts, k=k)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        # Create patches
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, -1, p*p)
        patches = patches.permute(0, 2, 1, 3).reshape(B, -1, C*p*p)

        tokens = self.patch_embed(patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.fc(x[:, 0])


