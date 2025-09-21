import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    return F.gelu(x)


def get_sincos2d_posemb(h: int, w: int, dim: int, device, dtype=torch.float32, temperature: float = 10000.0):
    if dim % 4 != 0 or dim < 8:
        raise ValueError("hidden_size (dim) must be multiple of 4 and >= 8 for sincos2d")
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.flatten()
    yy = yy.flatten()
    n = xx.shape[0]
    d = dim // 4
    omega = torch.arange(d, device=device, dtype=dtype) / (d - 1)
    omega = 1.0 / (temperature ** omega)
    x_proj = xx.unsqueeze(1) * omega.unsqueeze(0)  # (n, d)
    y_proj = yy.unsqueeze(1) * omega.unsqueeze(0)
    posemb = torch.cat([torch.sin(x_proj), torch.cos(x_proj), torch.sin(y_proj), torch.cos(y_proj)], dim=-1)  # (n, dim)
    return posemb.unsqueeze(0)  # (1, n, dim)


# -------------------------
# MLP block (matches ViT MlpBlock semantics)
# -------------------------
class MlpBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int, dropout_rate: float = 0.0, deterministic: Optional[bool] = None):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self._deterministic = deterministic

    def deterministic(self) -> bool:
        if self._deterministic is not None:
            return self._deterministic
        return not self.training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = gelu(x)
        if not self.deterministic():
            x = self.dropout(x)
        x = self.fc2(x)
        if not self.deterministic():
            x = self.dropout(x)
        return x


# -------------------------
# Router: Noisy Top-K per token (per-group if inputs grouped)
# -------------------------
class NoisyTopExpertsPerItemRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, num_selected_experts: int = 1,
                 noise_std: float = 1.0, deterministic: Optional[bool] = None):
        super().__init__()
        self.num_experts = num_experts
        self.k = num_selected_experts
        self.noise_std = noise_std
        self.logit_proj = nn.Linear(hidden_dim, num_experts)
        self._deterministic = deterministic

    def deterministic(self):
        if self._deterministic is not None:
            return self._deterministic
        return not self.training

    def forward(self, tokens: torch.Tensor):
        """
        tokens: (G, S, H) where G = groups (batch*groups), S = group_size, H = hidden_dim
        returns dispatcher dict and metrics dict.
        dispatcher: {'topk_idx': (G,S,k), 'topk_w': (G,S,k)}
        metrics: {'auxiliary_loss': scalar, 'gates': (G,S,E)}  # gates optional (can be memory heavy)
        """
        G, S, H = tokens.shape
        flat = tokens.reshape(G * S, H)  # (N, H)
        logits = self.logit_proj(flat)   # (N, E)

        if self.noise_std is not None and (not self.deterministic()):
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)  # (N, E)
        if self.k >= self.num_experts:
            topk_vals, topk_idx = probs.topk(self.num_experts, dim=-1)
        else:
            topk_vals, topk_idx = probs.topk(self.k, dim=-1)  # (N, k)

        topk_w = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)  # normalize (N, k)

        # reshape to (G, S, k)
        topk_idx = topk_idx.view(G, S, -1)
        topk_w = topk_w.view(G, S, -1)

        # Optional diagnostics: compute simple load-balancing aux loss
        importance = probs.sum(dim=0)  # (E,)
        load = (probs > 0).float().sum(dim=0)  # (E,) coarse
        aux = ((importance * load).float().var()).unsqueeze(0)  # scalar variance as proxy
        auxiliary_loss = aux * 1e-2  # scale factor (tunable)

        dispatcher = {'topk_idx': topk_idx, 'topk_w': topk_w}
        metrics = {'auxiliary_loss': auxiliary_loss.squeeze(0), 'gates': probs.view(G, S, self.num_experts)}
        return dispatcher, metrics

# -------------------------
# MLP-MoE Block (single-device)
# -------------------------
class MlpMoeBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int, num_experts: int, group_size: int,
                 dropout_rate: float = 0.0, deterministic: bool = False,
                 router: Optional[Dict[str, Any]] = None):
        """
        hidden_dim: input/output dim of tokens
        mlp_dim: intermediate dim of per-expert MLP
        num_experts: number of experts
        group_size: group size for routing
        router: dict, e.g. {'num_selected_experts': 2, 'noise_std': 1/8}
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_experts = num_experts
        self.group_size = group_size
        self.dropout_rate = dropout_rate
        self.deterministic_flag = deterministic

        router_kwargs = router or {}
        k = int(router_kwargs.get('num_selected_experts', router_kwargs.get('k', 1)))
        noise_std = float(router_kwargs.get('noise_std', router_kwargs.get('noise', 1.0)))

        self.experts = nn.ModuleList([
            MlpBlock(hidden_dim, self.mlp_dim,
                     dropout_rate=self.dropout_rate,
                     deterministic=self.deterministic_flag)
            for _ in range(self.num_experts)
        ])

        self.router = NoisyTopExpertsPerItemRouter(
            hidden_dim, self.num_experts,
            num_selected_experts=k,
            noise_std=noise_std,
            deterministic=self.deterministic_flag
        )

    def forward(self, inputs: torch.Tensor):
        """
        inputs: (num_seqs, seq_length, hidden_size)
        returns: outputs: (num_seqs, seq_length, hidden_size), metrics: dict
        """
        assert inputs.dim() == 3, f'Expected ndim=3, got {inputs.shape}'
        ns, seq_len, hidden = inputs.shape
        self._ensure_experts_and_router(hidden)

        # reshape to groups: (-1, group_size, hidden)
        pad = 0
        if (seq_len % self.group_size) != 0:
            pad = self.group_size - (seq_len % self.group_size)
            pad_tensor = inputs.new_zeros((ns, pad, hidden))
            inputs_padded = torch.cat([inputs, pad_tensor], dim=1)
        else:
            inputs_padded = inputs

        new_seq_len = inputs_padded.shape[1]
        groups_per_seq = new_seq_len // self.group_size
        G = ns * groups_per_seq
        S = self.group_size

        x = inputs_padded.view(ns * groups_per_seq, S, hidden)  # (G, S, H)

        # Router (dispatcher)
        dispatcher, metrics = self.router(x)  # dispatcher: topk_idx (G,S,k), topk_w (G,S,k)
        topk_idx = dispatcher['topk_idx']    # LongTensor (G,S,k)
        topk_w = dispatcher['topk_w']        # FloatTensor (G,S,k)
        k = topk_idx.shape[-1]

        # Prepare output buffer (flattened)
        x_flat = x.reshape(G * S, hidden)  # (N, H)
        out_flat = x_flat.new_zeros(x_flat.shape)

        token_idx = torch.arange(G * S, device=x.device).view(G, S)  # (G, S)

        # For each expert, gather all tokens assigned to it across top-k, run expert once on that sub-batch
        # Build per-expert list of flat indices and weight sums
        weight_acc = torch.zeros(G * S, device=x.device, dtype=x_flat.dtype)
        out_acc = torch.zeros_like(out_flat)

        for expert_id in range(self.num_experts):
            # Boolean mask where expert appears in any topk position
            # Collect indices where topk_idx == expert_id (G,S,k)
            mask = (topk_idx == expert_id)  # bool (G,S,k)
            if not mask.any():
                continue
            # compute weight per token as sum over k positions where this expert selected
            # topk_w shape (G,S,k)
            w_for_expert = (topk_w * mask.float()).sum(dim=-1)  # (G,S)
            sel_mask = w_for_expert > 0  # (G,S)
            if not sel_mask.any():
                continue
            sel_flat_idx = token_idx[sel_mask]  # (N_sel,)
            selected_vectors = x_flat[sel_flat_idx]  # (N_sel, H)
            # run expert
            expert_out = self.experts[expert_id](selected_vectors)  # (N_sel, H)
            w_sel = w_for_expert[sel_mask].unsqueeze(-1)  # (N_sel,1)
            weighted = expert_out * w_sel  # weighted outputs
            # scatter add to out_acc and weight_acc
            out_acc[sel_flat_idx] += weighted
            weight_acc[sel_flat_idx] += w_for_expert[sel_mask]

        # normalize tokens that got multiple experts
        eps = 1e-9
        out_flat = out_acc / (weight_acc.unsqueeze(-1) + eps)

        # reshape output back to (G, S, H)
        outputs = out_flat.view(G, S, hidden)
        # reshape to (ns, new_seq_len, hidden)
        outputs = outputs.view(ns, groups_per_seq * S, hidden)
        if pad:
            outputs = outputs[:, :seq_len, :]

        return outputs, metrics

# -------------------------
# MapHead: Multihead attention pooling with learned probe
# -------------------------
class MapHead(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int, num_heads: int, qk_norm: bool = False):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.probe)
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        # deterministic MlpBlock
        self.mlp = MlpBlock(hidden_dim, mlp_dim, dropout_rate=0.0, deterministic=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H)
        B, L, H = x.shape
        probe = self.probe.expand(B, -1, -1)  # (B,1,H)
        attn_out, _ = self.attn(query=probe, key=x, value=x, need_weights=False)  # (B,1,H)
        y = self.ln(attn_out)
        y = self.mlp(y)  # (B,1,H)
        out = (attn_out + y)[:, 0, :]  # (B, H)
        return out

# -------------------------
# EncoderBlock: attention + MLP (which might be MoE)
# -------------------------
class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_block_factory, num_heads: int,
                 dropout_rate: float = 0.0, attention_dropout_rate: float = 0.0,
                 attention_qk_norm: bool = False, deterministic: bool = False):
        """
        mlp_block_factory: callable that returns an instance of MlpBlock or MlpMoeBlock
        """
        super().__init__()
        self.mlp_block_factory = mlp_block_factory
        self.num_heads = num_heads
        self.attention_qk_norm = attention_qk_norm
        self.dropout_rate = dropout_rate
        self.deterministic = deterministic

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.res_dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor):
        # inputs: (B, L, H)
        x = self.ln1(inputs)
        attn_out, _ = self.attn(query=x, key=x, value=x, need_weights=False)
        if not self.deterministic:
            attn_out = self.attn_dropout(attn_out)
        x = x + attn_out
        y = self.ln2(x)

        mlp_module = self.mlp_block_factory()
        y_out = mlp_module(y)

        if isinstance(y_out, tuple):  # MoE returns (out, metrics)
            y, metrics = y_out
            if not self.deterministic:
                y = self.res_dropout(y)
            return x + y, metrics
        else:
            if not self.deterministic:
                y_out = self.res_dropout(y_out)
            return x + y_out
        
class EncoderMoe(nn.Module):
    DEFAULT_SINCOS2D_TEMPERATURE = 10000.0

    def __init__(self, hidden_size: int, num_layers: int, mlp_dim: int, num_heads: int,
                 dropout_rate: float = 0.0, attention_dropout_rate: float = 0.0,
                 attention_qk_norm: bool = False, moe: Optional[Dict[str, Any]] = None,
                 deterministic: bool = False, position_emb: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.attention_qk_norm = attention_qk_norm
        self.moe = moe or {}
        self.deterministic = deterministic
        self.position_emb = position_emb or {}

        dense_mlp_params = dict(mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate)
        moe_mlp_params = {**dense_mlp_params, **(self.moe or {})}
        moe_mlp_layers = tuple(moe_mlp_params.get('layers', ()))

        # Factories: capture hidden_size
        def dense_mlp_factory():
            return MlpBlock(self.hidden_size, self.mlp_dim,
                            dropout_rate=self.dropout_rate,
                            deterministic=self.deterministic)

        def moe_mlp_factory():
            router_kwargs = moe_mlp_params.get('router', {})
            return MlpMoeBlock(hidden_dim=self.hidden_size,
                               mlp_dim=self.mlp_dim,
                               num_experts=moe_mlp_params.get('num_experts', 1),
                               group_size=moe_mlp_params.get('group_size', 1),
                               dropout_rate=self.dropout_rate,
                               deterministic=self.deterministic,
                               router=router_kwargs)

        # Build all blocks upfront
        self.layers = nn.ModuleList()
        for block in range(self.num_layers):
            if block in moe_mlp_layers:
                self.layers.append(
                    EncoderBlock(hidden_dim=self.hidden_size,
                                 mlp_block_factory=moe_mlp_factory,
                                 num_heads=self.num_heads,
                                 dropout_rate=self.dropout_rate,
                                 attention_dropout_rate=self.attention_dropout_rate,
                                 attention_qk_norm=self.attention_qk_norm,
                                 deterministic=self.deterministic)
                )
            else:
                self.layers.append(
                    EncoderBlock(hidden_dim=self.hidden_size,
                                 mlp_block_factory=dense_mlp_factory,
                                 num_heads=self.num_heads,
                                 dropout_rate=self.dropout_rate,
                                 attention_dropout_rate=self.attention_dropout_rate,
                                 attention_qk_norm=self.attention_qk_norm,
                                 deterministic=self.deterministic)
                )

        self.final_ln = nn.LayerNorm(self.hidden_size)

    def forward(self, inputs: torch.Tensor):
        x = inputs
        if self.dropout_rate > 0 and not self.deterministic:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        metrics: Dict[str, Any] = {}
        for block_idx, layer in enumerate(self.layers):
            out = layer(x)
            if isinstance(out, tuple):  # MoE block
                x, block_metrics = out
                metrics[f'encoderblock_{block_idx}'] = block_metrics
            else:
                x = out

        encoded = self.final_ln(x)

        # Sum auxiliary losses
        aux = 0.0
        for v in metrics.values():
            if isinstance(v, dict) and 'auxiliary_loss' in v:
                aux += float(v['auxiliary_loss'].item()
                             if isinstance(v['auxiliary_loss'], torch.Tensor)
                             else v['auxiliary_loss'])
        metrics['auxiliary_loss'] = torch.tensor(aux, device=inputs.device, dtype=inputs.dtype)
        return encoded, metrics

# -------------------------
# VisionTransformerMoe: top-level model
# -------------------------
class VisionTransformerMoe(nn.Module):
    def __init__(self,
                 num_classes: Optional[int],
                 patch_size: Tuple[int, int],
                 hidden_size: int,
                 encoder: Dict[str, Any],
                 classifier: str = 'token',
                 representation_size: Optional[int] = None,
                 deterministic: bool = False,
                 head_bias_init: float = 0.0,
                 head_kernel_zero_init: bool = True,
                 encoder_cls=EncoderMoe):
        """
        encoder: dict containing encoder settings. Example:
        {
          'num_layers': 12,
          'mlp_dim': 1536,
          'num_heads': 12,
          'dropout_rate': 0.0,
          'attention_dropout_rate': 0.0,
          'moe': {
             'layers': [6,7],
             'num_experts': 8,
             'group_size': 1,
             'router': {'num_selected_experts':2, 'noise_std':0.125}
          },
          'position_emb': {'name':'learned'} or {'name':'sincos2d', 'h':h,'w':w}
        }
        """
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size  # tuple (ph, pw)
        self.hidden_size = hidden_size
        self.encoder_cfg = dict(encoder)
        self.classifier = classifier
        self.representation_size = representation_size
        self.deterministic = deterministic
        self.head_bias_init = head_bias_init
        self.head_kernel_zero_init = head_kernel_zero_init
        self.encoder_cls = encoder_cls

        # patch embedding conv (features = hidden_size)
        ks = tuple(self.patch_size) if isinstance(self.patch_size, (list, tuple)) else (self.patch_size, self.patch_size)
        self.embedding = nn.Conv2d(in_channels=3, out_channels=self.hidden_size, kernel_size=ks, stride=ks, padding=0)

        # positional embedding: learned by default
        self.position_emb_cfg = self.encoder_cfg.get('position_emb', {'name': 'learned'})
        self.posemb = None
        self.posemb_sincos = (self.position_emb_cfg.get('name') == 'sincos2d')
        # If learned and we know number of patches, create posemb in forward when size known.

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        nn.init.zeros_(self.cls_token)

        # pre_logits (representation)
        if self.representation_size is not None:
            self.pre_logits = nn.Linear(self.hidden_size, self.representation_size)
        else:
            self.pre_logits = nn.Identity()

        # head (will be created once)
        if self.num_classes is not None:
            out_dim = self.representation_size if self.representation_size is not None else self.hidden_size
            self.head = nn.Linear(out_dim, self.num_classes)
            if self.head_kernel_zero_init:
                nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, self.head_bias_init)
        else:
            self.head = None

    def _maybe_create_posemb(self, x):
        # x: conv output (B, Hgrid, Wgrid, C) or after reshape (B, Npatch, C)
        # We'll create learned posemb param if needed
        if self.position_emb_cfg.get('name', 'learned') == 'learned':
            # determine n patches
            # embedding conv hasn't been applied yet here: call after conv in forward
            pass

    def forward(self, inputs: torch.Tensor):
        """
        inputs: (B, C=3, H_img, W_img)
        returns: (logits OR representation, metrics)
        """
        B = inputs.shape[0]
        print(inputs.shape)
        x = self.embedding(inputs)  # (B, C_out=hidden, Hf, Wf)
        print(x.shape)
        _, C_out, Hf, Wf = x.shape

        # prepare encoder kwargs, handle sincos2d
        encoder_kwargs = dict(self.encoder_cfg)
        if encoder_kwargs.get('position_emb', {}).get('name') == 'sincos2d':
            encoder_kwargs['position_emb'] = dict(encoder_kwargs['position_emb'])
            encoder_kwargs['position_emb']['h'] = Hf
            encoder_kwargs['position_emb']['w'] = Wf

        # reshape conv output to (B, N, hidden)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C_out)  # (B, N, hidden)

        # class token
        if self.classifier == 'token':
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)  # (B, N+1, hidden)

        # add positional embedding
        pos_cfg = encoder_kwargs.get('position_emb', {}) or {}
        name = pos_cfg.get('name', 'learned')
        n = x.shape[1]
        device = x.device
        dtype = x.dtype

        if name == 'none':
            x_pe = x
        elif name == 'learned':
            # create learned posemb if not exists or mismatch size
            if (self.posemb is None) or (self.posemb.shape[1] != n):
                self.posemb = nn.Parameter(torch.zeros(1, n, self.hidden_size, device=device, dtype=dtype))
                nn.init.normal_(self.posemb, std=0.02)
            x_pe = x + self.posemb
        elif name == 'sincos2d':
            posemb = get_sincos2d_posemb(pos_cfg['h'], pos_cfg['w'], self.hidden_size, device=device, dtype=dtype)
            if n == posemb.shape[1]:
                x_pe = x + posemb
            elif n == posemb.shape[1] + 1:
                a = x[:, :1, :]
                b = x[:, 1:, :] + posemb
                x_pe = torch.cat([a, b], dim=1)
            else:
                raise ValueError(f"Unsupported sequence length {n=} for the given {pos_cfg['h']=} and {pos_cfg['w']=}")
        else:
            raise ValueError(f"Unsupported position embedding: {name}")

        x = F.dropout(x_pe, p=self.encoder_cfg.get('dropout_rate', 0.0), training=self.training)

        # call encoder
        encoder_args = dict(num_layers=self.encoder_cfg['num_layers'],
                            mlp_dim=self.encoder_cfg['mlp_dim'],
                            num_heads=self.encoder_cfg['num_heads'],
                            dropout_rate=self.encoder_cfg.get('dropout_rate', 0.0),
                            attention_dropout_rate=self.encoder_cfg.get('attention_dropout_rate', 0.0),
                            attention_qk_norm=self.encoder_cfg.get('attention_qk_norm', False),
                            moe=self.encoder_cfg.get('moe', None),
                            deterministic=self.deterministic)
        encoder = self.encoder_cls(**encoder_args)
        x, metrics = encoder(x)

        # pooling/classifier
        if self.classifier == 'token' or self.classifier == '0':
            rep = x[:, 0]
        elif self.classifier == 'gap':
            rep = x.mean(dim=1)
        elif self.classifier == 'map':
            maphead = MapHead(hidden_dim=self.encoder_cfg['mlp_dim'], mlp_dim=self.encoder_cfg['mlp_dim'], num_heads=self.encoder_cfg['num_heads'])
            rep = maphead(x)
        else:
            raise ValueError(f'Unknown classifier: {self.classifier!r}')

        if self.representation_size is not None:
            pre = self.pre_logits(rep)
            pre = torch.tanh(pre)
        else:
            pre = self.pre_logits(rep)

        if self.num_classes is not None:
            logits = self.head(pre)
            return logits, metrics
        else:
            return pre, metrics
