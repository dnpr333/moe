import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from transformers import AutoImageProcessor, ViTConfig, AutoModelForImageClassification
from transformers.models.vit.modeling_vit import ViTLayer

from base import MoELayer, MoESubmodules

class ViTLayerWithMoE(ViTLayer):
    def __init__(self, config, submodules):
        super().__init__(config)
        self.intermediate = None  # remove HuggingFace MLP
        self.output = None
        self.moe = MoELayer(
            config=config,
            submodules=submodules,
            layer_number=0
        )

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask=head_mask
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add attentions if we output them

        hidden_states = attention_output + hidden_states

        residual = hidden_states
        
        hidden_states = self.layernorm_after(hidden_states)

        # print("Before MoE:", hidden_states.shape)
        moe_out, _ = self.moe(hidden_states)
        # print("After MoE:", moe_out.shape)

        hidden_states = residual + moe_out
        
        # print(outputs.shape, type(outputs))
        # print(hidden_states.shape, type(hidden_states))
        return (hidden_states,) + outputs
        # print(f"Hidden states shape: {hidden_states.shape}")

        # return hidden_states

def train(model, trainloader, testloader, device, epochs=5, lr=5e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)

            # print(imgs.shape)
            inputs = image_processor(imgs, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs).logits

            # outputs = model(pixel_values=imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(trainloader):.4f} | Acc: {acc:.2f}%")

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(pixel_values=imgs).logits
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        print(f"Test Acc: {100. * correct / total:.2f}%")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load pretrained ViT
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=10  # CIFAR-100
    )

    # Replace every encoder block with MoE-enabled block
    submodules = MoESubmodules(
        experts=nn.ModuleList([nn.Linear(768, 3072) for _ in range(2)]),  # Example: 4 experts
        shared_experts=None  # No shared experts in this setup
    )
    moe_config = ViTConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        num_hidden_layers=12,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        moe_shared_expert_intermediate_size=None,
        moe_shared_expert_overlap=False,
        recompute_granularity='none',
        recompute_modules=False,
        sequence_parallel=False,
        num_moe_experts=2,
        calculate_per_token_loss=False,
        perform_initialization=False,
        moe_router_topk=2,
        moe_router_load_balancing_type='aux_loss',
        moe_router_score_function='softmax',
        moe_router_enable_expert_bias=False,
        _attn_implementation='eager',
        moe_input_jitter_eps=None, # float
        moe_router_dtype='fp32',
        moe_router_force_load_balancing=False,
        moe_z_loss_coeff=None, # 1e-3
        moe_router_pre_softmax=False, # softmax before topk(router)
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_router_topk_scaling_factor=None,
        moe_expert_capacity_factor=None,
        moe_aux_loss_coeff=1e-2,
        num_layers=0,
        mtp_num_layers=None,
        moe_ffn_hidden_size=3072, # hidden_size * 4
        gated_linear_unit=False,
        add_bias_linear=True,
        activation_func=F.gelu,
        moe_apply_probs_on_input=False,
        moe_permute_fusion=False,
    )

    for i, block in enumerate(model.vit.encoder.layer):
        # print(type(model.vit.encoder.layer[i]))
        model.vit.encoder.layer[i] = ViTLayerWithMoE(moe_config, submodules)

    # print(type(model))
    model.to(device)

    batch_size = 2
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),  # Ensure values are in [0, 1]
        # transforms.Normalize((0.5071, 0.4865, 0.4409),
        #                      (0.2673, 0.2564, 0.2762)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4865, 0.4409),
        #                      (0.2673, 0.2564, 0.2762)),
    ])
    trainset = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=1)
    testset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=1)
    
    train(model, trainloader, testloader, device, epochs=5, lr=5e-5)
