import torch
import time

from model import GPT
from config import GPTConfig
from utils import seed_everything, get_device, count_parameter, print_info
from dataloader import DataLoaderLite

from torch.utils.tensorboard import SummaryWriter
import math
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# NOTE: should run with `torchrun --standalone --nproc_per_node=2 train.py`
if __name__ == "__main__":
    ddp = int(os.environ.get("RANK", -1)) != -1  # True if DDP is used
    if ddp:
        destroy_process_group()
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])  # Each process has a unique RANK
        ddp_local_rank = int(
            os.environ["LOCAL_RANK"]
        )  # This is used in the Multi-node setting: The rank on the GPU on the single node
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # device = get_device()

    total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
    B = 64  # micro batch size
    T = 1024  # sequence length

    assert (
        total_batch_size % (B * T * ddp_world_size) == 0
    ), "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(
        B=4,
        T=32,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
    )
    # split="train",
    # master_process=master_process,

    config = GPTConfig(vocab_size=50304)
    model = GPT(config).to(device)
    model = torch.compile(model)

    # count_parameter(model)
    if ddp:  # Wrap the model into the DDP container
        # raw_model = model.module if ddp else model
        model = DDP(model, device_ids=[ddp_local_rank])

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

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

    writer = SummaryWriter(log_dir="./Experiment/runs")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for step in range(max_steps):
        # t0 = time.time()
        # x, y = train_loader.next_batch()
        # x, y = x.to(device), y.to(device)
        # optimizer.zero_grad()
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        #     logits, loss = model(x, y)
        # loss.backward()
        # optimizer.step()
        # torch.cuda.synchronize()
        # t1 = time.time()
        # dt = (t1 - t0) * 1000
        # print_info(writer, step=i, loss=loss.item(), dt=dt)
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        token_processed = (
            train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        )
        token_per_sec = token_processed / dt
        if master_process:
            print_info(
                writer, step=step, loss=loss_accum, dt=dt, token_per_sec=token_per_sec
            )

    writer.close()

    if ddp:
        destroy_process_group()
