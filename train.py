import torch
import time
import logging

from model import GPT
from config import GPTConfig
from utils import seed_everything, get_device, count_parameter, print_info, set_logger
from dataloader import DataLoaderLite
from optimizer import *
from decode import generate_sentence

from torch.utils.tensorboard import SummaryWriter  # type: ignore
import math
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# NOTE: should run with `torchrun --standalone --nproc_per_node=2 train.py`
if __name__ == "__main__":
    ddp = int(os.environ.get("RANK", -1)) != -1  # True if DDP is used
    if ddp:
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

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    seed_everything(1337)

    total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
    B = 8  # micro batch size
    T = 1024  # sequence length
    assert (
        total_batch_size % (B * T * ddp_world_size) == 0
    ), "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if master_process:
        set_logger("train.log")
        logging.info(f"total desired batch size: {total_batch_size}")
        logging.info(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
        writer = SummaryWriter(log_dir="./Experiment/runs")

    train_loader = DataLoaderLite(
        B=4,
        T=32,
        split="train",
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        master_process=master_process,
    )
    val_loader = DataLoaderLite(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
        master_process=master_process,
    )

    config = GPTConfig(vocab_size=50304)
    model = GPT(config).to(device)
    model = torch.compile(model)

    # count_parameter(model)
    if ddp:  # Wrap the model into the DDP container
        # raw_model = model.module if ddp else model
        model = DDP(model, device_ids=[ddp_local_rank])

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    tokenizer = tiktoken.get_encoding("gpt2")
    for step in range(max_steps):
        if master_process:
            print(f"step: {step}/{max_steps}")
        t0 = time.time()
        last_step = step == max_steps - 1

        if (step > 0 and step % 250 == 0) or last_step:
            generate_sentence(
                model,
                tokenizer=tokenizer,
                device=device,
                device_type=device_type,
                ddp_rank=ddp_rank,
            )
        optimizer.zero_grad()
        loss_accum = 0
        model.train()
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(
                device_type=device_type, dtype=torch.bfloat16
            ):  # Mixed Precision
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == grad_accum_steps - 1
                )  # This line tell the model sync until the last micro_step

            loss.backward()

        if ddp:
            dist.all_reduce(
                loss_accum, op=dist.ReduceOp.AVG
            )  # Average the loss across all processes

        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )  # Gradient Clipping

        # Learning Scheduler
        lr = get_lr(step)
        update_lr(optimizer, lr)

        optimizer.step()
        torch.cuda.synchronize()  # The CPU just send the task, need wait for the GPU to finish
        t1 = time.time()
        dt = (t1 - t0) * 1000
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
