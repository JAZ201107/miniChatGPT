import torch
import torch.nn as nn


def generate_sentence(
    model, tokenizer, prefix=None, device="cuda", device_type="cuda", ddp_rank=0.0
):
    model.eval()

    if prefix is None:
        prefix = tokenizer.encode("The meaning of life is")

    num_return_sequences = 4
    max_length = 32

    tokens = tokenizer.encode(prefix)
    tokens = (
        torch.tensor(tokens, dtype=torch.long)
        .unsqueeze(0)
        .repeat(num_return_sequences, 1)
    )
    tokens = tokens.to(device)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while tokens.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(tokens)
            logits = logits[:, -1, :]  # We just need last token logits
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)

            xcol = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = tokens[i].cpu().numpy()
        decoded = tokenizer.decode(tokens)

        print(f"rank {ddp_rank}:", decoded)
