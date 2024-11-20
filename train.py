import torch

from model import GPT
from config import GPTConfig
from utils import seed_everything, get_device, count_parameter, print_info
from dataloader import DataLoaderLite

from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    device = get_device()
    model = GPT(GPTConfig).to(device)

    train_loader = DataLoaderLite("./data/input.txt", B=4, T=32)
    count_parameter(model)

    writer = SummaryWriter(log_dir="./Experiment/runs")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for i in range(50):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        print_info(writer, step=i, loss=loss.item())

    writer.close()
