import logging
import numpy as np
from omegaconf import DictConfig
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from typing import Tuple
import wandb

from modeling.diffusion import DiffusionModel


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss.item()


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
            wandb.log(
                {
                    "loss": train_loss,
                    "lr": lr,
                    "x": wandb.Image(x[0].detach().permute(1, 2, 0).cpu().numpy())
                },
                step=n_epoch * len(dataloader) + step)
        pbar.set_description(f"loss: {loss_ema:.4f}")


def generate_samples(model: DiffusionModel, device: str, path: str, processing=None) -> Tuple[wandb.Image, wandb.Image]:
    model.eval()
    with torch.no_grad():
        inits, samples = model.sample(8, (3, 32, 32), device=device, processing=processing)
        path1, path2 = path + "_init.png", path + "_sample.png"
        save_image(make_grid(inits, nrow=4), path1)
        save_image(make_grid(samples, nrow=4), path2)
        return wandb.Image(path1), wandb.Image(path2)
