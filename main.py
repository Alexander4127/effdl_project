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


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    cfg.trainer.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.trainer.wandb.use:
        saved_dir = f"saved/{cfg.trainer.wandb.name}"
    else:
        saved_dir = "saved"
    Path(saved_dir).mkdir(exist_ok=True, parents=True)
    Path("samples").mkdir(exist_ok=True)

    if cfg.trainer.wandb.use:
        dict_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(config=dict_cfg, project=cfg.trainer.wandb.project, name=cfg.trainer.wandb.name)
        wandb.log_artifact("config.yaml", name="HydraConfig")

    logging.info(OmegaConf.to_yaml(cfg))
    ddpm = DiffusionModel(
        eps_model=UnetModel(cfg.arch.in_channels, cfg.arch.out_channels, hidden_size=cfg.arch.hidden_size),
        betas=(cfg.steps.beta_min, cfg.steps.beta_max),
        num_timesteps=cfg.steps.num_timesteps,
    )
    ddpm = ddpm.to(cfg.trainer.device)
    if cfg.trainer.wandb.use:
        wandb.watch(ddpm)

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ] +
        ([transforms.RandomHorizontalFlip()] if cfg.augments.flip else [])
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=True
    )
    optim = hydra.utils.instantiate(cfg.optim, ddpm.parameters())

    processing = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    for i in range(cfg.trainer.num_epochs):
        train_epoch(ddpm, dataloader, optim, i, cfg.trainer)
        torch.save(ddpm.state_dict(), f"{saved_dir}/ch{i}.pth")
        inits, samples = generate_samples(ddpm, cfg.trainer.device, f"samples/{i:02d}", processing)
        if cfg.trainer.wandb.use:
            wandb.log({"init": inits, "sample": samples}, step=(i + 1) * len(dataloader))

    if cfg.trainer.wandb.use:
        wandb.finish()


if __name__ == "__main__":
    main()
