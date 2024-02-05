import numpy as np
from omegaconf import DictConfig
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from typing import Tuple
import wandb

from modeling.diffusion import DiffusionModel


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss


def convert_numpy(samples):
    grid = make_grid(samples, nrow=4)
    return grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


def train_epoch(model: DiffusionModel,
                dataloader: DataLoader,
                optimizer: Optimizer,
                n_epoch: int,
                cfg: DictConfig):
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    for step, (x, _) in enumerate(pbar):
        train_loss = train_step(model, x, optimizer, cfg.device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        if step % cfg.log_step == 0 and cfg.wandb.use:
            lr = optimizer.param_groups[0]['lr']
            wandb.log({"loss": train_loss, "lr": lr}, step=n_epoch * len(dataloader) + step)
        pbar.set_description(f"loss: {loss_ema:.4f}")


def generate_samples(model: DiffusionModel, device: str, path: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        inits, samples = model.sample(8, (3, 32, 32), device=device)
        save_image(make_grid(samples, nrow=4), path)
        return convert_numpy(inits), convert_numpy(samples)
