import torch
import torch.nn as nn
import torch.nn.functional as F
from router import NoisyTopExpertsPerItemRouter

class IdentityLayer(nn.Module):
    def forward(self, x):
        return x

def gelu(x):
    return F.gelu(x)

class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, dropout_rate=0.0, deterministic=False):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.GELU()
        self.deterministic = deterministic

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MlpMoeBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, num_experts, group_size,
                 dropout_rate=0.0, deterministic=False, router=None, router_kwargs=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_experts = num_experts
        self.group_size = group_size
        self.dropout_rate = dropout_rate
        self.deterministic = deterministic

        # Experts: one MLP per expert
        self.experts = nn.ModuleList([
            MlpBlock(hidden_dim, mlp_dim, dropout_rate, deterministic) for _ in range(num_experts)
        ])

        # Router
        # router_cls = router or NoisyTopExpertsPerItemRouter
        # print(router_kwargs)
        self.router = NoisyTopExpertsPerItemRouter(**router_kwargs)

    def forward(self, x):
        b, s, h = x.shape
        assert h == self.hidden_dim, f"Expected hidden_dim={self.hidden_dim}, got {h}"

        num_groups = (b * s) // self.group_size
        x = x.reshape(num_groups, self.group_size, h)

        gates_softmax, metrics = self.router(x)

        outputs = torch.zeros_like(x)
        # print(self.experts)
        for i, expert in enumerate(self.experts):
            gate_weights = gates_softmax[..., i].unsqueeze(-1)  # (groups, group_size, 1)
            expert_out = expert(x)                              # (groups, group_size, hidden)
            outputs += gate_weights * expert_out

        outputs = outputs.reshape(b, s, h)
        return outputs, metrics

class MapHead(nn.Module):
    def __init__(self, mlp_dim, num_heads, qk_norm=False, hidden_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.qk_norm = qk_norm
        self.hidden_size = hidden_size

        self.probe = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.xavier_uniform_(self.probe)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size)
        )

    def forward(self, x):
        B, N, C = x.shape
        probe = self.probe.expand(B, -1, -1)
        out, _ = self.attn(query=probe, key=x, value=x, need_weights=False)
        y = self.norm(out)
        y = self.mlp(y)
        return (out + y)[:, 0, :]

class EncoderBlock(nn.Module):
    def __init__(self, mlp_block, num_heads, hidden_size,
                 dropout_rate=0.0, attention_dropout_rate=0.0,
                 attention_qk_norm=False, deterministic=False):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.attention_qk_norm = attention_qk_norm
        self.deterministic = deterministic

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout_rate,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp_block = mlp_block(hidden_size=hidden_size)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.dropout(attn_out)
        x = x + h

        y = self.norm2(x)
        y_out = self.mlp_block(y)

        if isinstance(y_out, tuple):
            y_out, metrics = y_out
            return x + y_out, metrics
        else:
            return x + y_out
        
class AddPositionEmbs(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, 0, hidden_size))  # shape set at runtime
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        if self.pos_embedding.shape[1] != N:
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, N, C, device=x.device, dtype=x.dtype)
            )
            nn.init.normal_(self.pos_embedding, std=0.02)
        return x + self.pos_embedding


class EncoderMoe(nn.Module):
    DEFAULT_SINCOS2D_TEMPERATURE = 10_000.

    def __init__(self, num_layers, mlp_dim, num_heads,
                 hidden_size,
                 dropout_rate=0.0,
                 attention_dropout_rate=0.0,
                 attention_qk_norm=False,
                 moe=None,
                 deterministic=False,
                 position_emb=None,
                 mlp_cls=None,
                 moe_mlp_cls=None,
                 encoder_block_cls=None):
        super().__init__()
        # print(hidden_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.attention_qk_norm = attention_qk_norm
        self.moe = moe or {}
        self.deterministic = deterministic
        self.position_emb_cfg = position_emb or {}

        dense_mlp_params = dict(hidden_dim=hidden_size, mlp_dim=mlp_dim, dropout_rate=dropout_rate)
        # print(dense_mlp_params)
        moe_mlp_params = {**dense_mlp_params, **self.moe}
        # print(moe_mlp_params)
        self.moe_layers = set(moe_mlp_params.pop("layers", ()))
        self.dense_mlp_cls = mlp_cls or (lambda **kwargs: MlpBlock(**dense_mlp_params))
        self.moe_mlp_cls = moe_mlp_cls or (lambda **kwargs: MlpMoeBlock(**moe_mlp_params,
                                                                router_kwargs=moe_mlp_params.get("router", {})))
        self.encoder_block_cls = encoder_block_cls or (lambda **kwargs: EncoderBlock(
            mlp_block=None,
            num_heads=num_heads,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            attention_qk_norm=attention_qk_norm,
            deterministic=deterministic
        ))

        self.blocks = nn.ModuleList()
        for block_idx in range(num_layers):
            if block_idx in self.moe_layers:
                block = EncoderBlock(
                    mlp_block=lambda hidden_size=hidden_size: self.moe_mlp_cls(hidden_size=hidden_size),
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    attention_qk_norm=attention_qk_norm,
                    deterministic=deterministic
                )
            else:
                block = EncoderBlock(
                    mlp_block=lambda hidden_size=hidden_size: self.dense_mlp_cls(hidden_size=hidden_size),
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    attention_qk_norm=attention_qk_norm,
                    deterministic=deterministic
                )
            self.blocks.append(block)

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.add_position_emb(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        metrics = {}
        for i, block in enumerate(self.blocks):
            if i in self.moe_layers:
                x, block_metrics = block(x)
                metrics[f"encoderblock_{i}"] = block_metrics
            else:
                x = block(x)

        encoded = self.norm(x)

        if metrics:
            metrics["auxiliary_loss"] = sum(m["auxiliary_loss"] for m in metrics.values())
        else:
            metrics["auxiliary_loss"] = 0.0

        return encoded, metrics

    def add_position_emb(self, x):
        pos_cfg = self.position_emb_cfg
        name = pos_cfg.get("name", "learned")

        if name == "none":
            return x
        elif name == "learned":
            return AddPositionEmbs(self.hidden_size)(x)
        elif name == "sincos2d":
            B, N, C = x.shape
            h, w = pos_cfg["h"], pos_cfg["w"]
            if C % 4 != 0 or C < 8:
                raise ValueError(f"hidden_size={C} must be multiple of 4 and >= 8")

            temperature = pos_cfg.get("temperature", self.DEFAULT_SINCOS2D_TEMPERATURE)
            y, x_grid = torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=x.device),
                torch.arange(w, dtype=torch.float32, device=x.device),
                indexing="ij"
            )
            y, x_grid = y.flatten(), x_grid.flatten()
            omega = torch.arange(C // 4, dtype=torch.float32, device=x.device) / (C // 4 - 1)
            omega = 1.0 / (temperature ** omega)

            x_emb = torch.einsum("n,d->nd", x_grid, omega)
            y_emb = torch.einsum("n,d->nd", y, omega)
            pos_emb = torch.cat(
                [torch.sin(x_emb), torch.cos(x_emb), torch.sin(y_emb), torch.cos(y_emb)], dim=1
            )[None, ...]

            if N == h * w:
                return x + pos_emb
            elif N == h * w + 1:
                a = x[:, :1, :]
                b = x[:, 1:, :] + pos_emb
                return torch.cat([a, b], dim=1)
            else:
                raise ValueError(f"Unsupported sequence length {N} for given {h}x{w}")
        else:
            raise ValueError(f"Unsupported position embedding: {name}")

class VisionTransformerMoe(nn.Module):
    def __init__(self, 
                 num_classes: int = None,
                 patch_size: tuple = (16, 16),
                 hidden_size: int = 768,
                 encoder: dict = None,
                 classifier: str = "token",
                 representation_size: int = None,
                 deterministic: bool = False,
                 head_bias_init: float = 0.0,
                 head_kernel_zero_init: bool = True,
                 encoder_cls: nn.Module = None):
        super().__init__()
        # print(num_classes)

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.encoder_cfg = encoder or {}
        self.classifier = classifier
        self.representation_size = representation_size
        self.deterministic = deterministic
        self.head_bias_init = head_bias_init
        self.head_kernel_zero_init = head_kernel_zero_init
        self.encoder_cls = encoder_cls or EncoderMoe
        
        self.embedding = nn.Conv2d(
            in_channels=3, out_channels=hidden_size,
            kernel_size=patch_size, stride=patch_size
        )

        if classifier == "token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.cls_token = None

        if representation_size is not None:
            self.pre_logits = nn.Sequential(
                nn.Linear(hidden_size, representation_size),
                nn.Tanh()
            )
        else:
            self.pre_logits = IdentityLayer()

        if num_classes is not None:
            self.head = nn.Linear(
                hidden_size if representation_size is None else representation_size,
                num_classes
            )
            if head_kernel_zero_init:
                nn.init.zeros_(self.head.weight)
            if head_bias_init != 0.0:
                nn.init.constant_(self.head.bias, head_bias_init)
        else:
            self.head = None

    def forward(self, x):
        x = self.embedding(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        encoder = self.encoder_cls(**self.encoder_cfg).to(x.device) 
        x, metrics = encoder(x)

        if self.classifier in ["token", "0"]:
            x = x[:, 0]
        elif self.classifier == "gap":
            x = x.mean(dim=1)
        elif self.classifier == "map":
            x = MapHead(
                num_heads=self.encoder['num_heads'], mlp_dim=self.encoder['mlp_dim'],
                qk_norm=self.encoder.get('attention_qk_norm', False),
                name='MapHead')(x)

        x = self.pre_logits(x)

        if self.head is not None:
            logits = self.head(x)
            return logits, metrics
        else:
            return x, metrics