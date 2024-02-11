import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import wandb

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


def main():
    cfg = OmegaConf.load("config.yaml")
    cfg.trainer.device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(OmegaConf.to_yaml(cfg))
    ddpm = DiffusionModel(
        eps_model=UnetModel(cfg.arch.in_channels, cfg.arch.out_channels, hidden_size=cfg.arch.hidden_size),
        betas=(cfg.steps.beta_min, cfg.steps.beta_max),
        num_timesteps=cfg.steps.num_timesteps,
    )
    ddpm = ddpm.to(cfg.trainer.device)
    ddpm.load_state_dict(torch.load("ch.pt"))
    processing = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    generate_samples(ddpm, cfg.trainer.device, "gen", processing)


if __name__ == "__main__":
    main()
