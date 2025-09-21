import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_sum(*weighted_tensors):
    total_weight = sum(w for w, _ in weighted_tensors)
    return sum(w * t for w, t in weighted_tensors) / total_weight

class NoisyTopExpertsPerItemRouter(nn.Module):
    def __init__(self, num_experts, num_selected_experts=1, noise_std=1.0,
                 gshard_loss_weight=0.0, importance_loss_weight=1.0,
                 load_loss_weight=1.0, dispatcher=None, deterministic=False,
                 dtype=None):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        self.noise_std = noise_std
        self.gshard_loss_weight = gshard_loss_weight
        self.importance_loss_weight = importance_loss_weight
        self.load_loss_weight = load_loss_weight
        self.dispatcher = dispatcher or {}
        self.deterministic = deterministic
        self.dtype = dtype

        self.gate_layer = nn.LazyLinear(out_features=num_experts, bias=False)

    def forward(self, x):
        b, s, h = x.shape
        if self.gate_layer.in_features is None:
            self.gate_layer = nn.Linear(h, self.num_experts, bias=False).to(x.device)

        gates_logits = self.gate_layer(x)

        gates_softmax = F.softmax(gates_logits, dim=-1)
        importance_loss = self._importance_auxiliary_loss(gates_softmax)

        if self.deterministic or self.noise_std == 0.0:
            gshard_loss = self._gshard_auxiliary_loss(gates_softmax)
            metrics = {
                "auxiliary_loss": weighted_sum(
                    (self.gshard_loss_weight, gshard_loss),
                    (self.importance_loss_weight, importance_loss)),
                "gshard_loss": gshard_loss,
                "importance_loss": importance_loss,
            }
            return gates_softmax, metrics
        else:
            noise_std = (1.0 / self.num_experts) * self.noise_std
            logits_noise = noise_std * torch.randn_like(gates_logits)
            gates_logits_noisy = gates_logits + logits_noise
            gates_softmax_noisy = F.softmax(gates_logits_noisy, dim=-1)

            load_loss = self._load_auxiliary_loss(
                gates_logits, gates_logits_noisy, noise_std, self.num_selected_experts
            )
            gshard_loss = self._gshard_auxiliary_loss(gates_softmax_noisy)

            metrics = {
                "auxiliary_loss": weighted_sum(
                    (self.gshard_loss_weight, gshard_loss),
                    (self.importance_loss_weight, importance_loss),
                    (self.load_loss_weight, load_loss)),
                "gshard_loss": gshard_loss,
                "importance_loss": importance_loss,
                "load_loss": load_loss,
            }
            return gates_softmax, metrics

    def _gshard_auxiliary_loss(self, gates):
        num_experts = gates.shape[-1]
        mean_gates_per_expert = gates.mean(dim=0)
        top1_expert = F.one_hot(torch.argmax(gates, dim=-1), num_classes=num_experts).float().mean(dim=0)
        aux_loss = torch.mean(top1_expert * mean_gates_per_expert) * (num_experts ** 2)
        return aux_loss

    def _importance_auxiliary_loss(self, gates):
        importance_per_expert = gates.sum(dim=tuple(range(gates.ndim - 1)))
        std_importance = torch.std(importance_per_expert)
        mean_importance = torch.mean(importance_per_expert)
        return (std_importance / mean_importance) ** 2

    def _load_auxiliary_loss(self, logits, logits_noisy, noise_std, num_selected_experts):
        num_experts = logits_noisy.shape[-1]
        topk_vals, topk_idx = torch.topk(logits_noisy, num_selected_experts, dim=-1)
        threshold_per_item = topk_vals[..., -1]
        threshold_per_item = F.one_hot(topk_idx[..., -1], num_classes=num_experts).float() * logits_noisy
        threshold_per_item = threshold_per_item.sum(dim=-1)

        noise_required_to_win = (threshold_per_item[..., None] - logits) / noise_std
        p = 1.0 - torch.distributions.Normal(0, 1).cdf(noise_required_to_win)
        p_mean = p.mean(dim=0)
        return (torch.std(p_mean) / torch.mean(p_mean)) ** 2
