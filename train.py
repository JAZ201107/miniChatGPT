import torch
import time

from model import GPT
from config import GPTConfig
from utils import seed_everything, get_device, count_parameter, print_info
from dataloader import DataLoaderLite

from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":

    config = GPTConfig(vocab_size=50304)
    device = get_device()
    model = GPT(config).to(device)
    model = torch.compile(model)

    train_loader = DataLoaderLite("./data/input.txt", B=4, T=32)
    count_parameter(model)

    writer = SummaryWriter(log_dir="./Experiment/runs")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for i in range(50):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        print_info(writer, step=i, loss=loss.item(), dt=dt)

    writer.close()
