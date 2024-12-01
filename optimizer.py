import torch
import torch.nn as nn

import inspect


def configure_optimizers(model, weight_decay, learning_rate, device):
    # param_dict = {pn: p for pn, p in model.named_parameters()} #
    param_dict = {
        pn: p for pn, p in param_dict.items() if p.requires_grad
    }  # Filter out the parameters that don't require gradients

    # Create optim groups. Any parameters that is 2D will be weight decayed
    decay_parameters = [p for n, p in param_dict.items() if p.dim() >= 2]
    other_parameters = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_parameters = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {
            "params": other_parameters,
            "weight_decay": 0,
        },
    ]

    num_decay_params = sum(p.numel() for p in decay_parameters)
    num_other_params = sum(p.numel() for p in other_parameters)
    print(
        f"Number of parameters: {num_decay_params} weight decay, {num_other_params} no weight decay"
    )

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    used_fused = fused_available and "cuda" in device
    print(f'Using {"fused" if used_fused else "unfused"} AdamW')

    optimizer = torch.optim.AdamW(
        optim_parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=used_fused,
    )
    return optimizer


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = (
    19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
)


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
