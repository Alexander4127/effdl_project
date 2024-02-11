import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, generate_samples, train_epoch
from modeling.unet import UnetModel


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in tqdm(range(50), desc="Train one batch"):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


def test_training():
    class OneSampleDataset(torch.utils.data.Dataset):
        def __init__(self, parent: torch.utils.data.Dataset, size: int):
            assert size <= len(parent)
            self.dataset = parent
            self.size = size
            self.idx = np.random.choice(np.arange(len(parent)), replace=False)

        def __len__(self):
            return self.size

        def __getitem__(self, item):
            return self.dataset[self.idx]

    # note: implement and test a complete training procedure (including sampling)
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        transform=transforms
    )
    bs = 2
    sampled_dataset = OneSampleDataset(dataset, size=2 * bs)
    device = "cpu"
    samples = torch.concat([sampled_dataset[i][0].unsqueeze(0) for i in range(2 * bs)])
    save_image(make_grid(samples, nrow=4), "tmp/dataset.png")

    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(sampled_dataset, batch_size=bs, shuffle=True)

    cfg = OmegaConf.create({"device": device, "log_step": 10, "wandb": {"use": False}})
    for _ in tqdm(range(1000)):
        train_epoch(ddpm, dataloader, optim, 0, cfg)

    processing = Normalize((-1, -1, -1), (2, 2, 2))
    generate_samples(ddpm, device, "tmp/samples", processing)
