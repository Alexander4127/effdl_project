# Simple efficient DL project

This project implements DDPM training on CIFAR10.

Experiments logs and generation results are presented in [wandb](https://wandb.ai/practice-cifar/effdl_project).

In addition, [Hydra](https://hydra.cc/) and [DVC](https://dvc.org/) are supported.

## Installation guide

To get started run the following code
```shell
pip install -r requirements.txt
```

## Testing code

Model and pipeline testing are evaluated by
```shell
pytest tests/
```

Testing coverage is presented below

    Name                    Stmts   Miss  Cover
    -------------------------------------------
    modeling/diffusion.py      38      0   100%
    modeling/training.py       37      2    95%
    modeling/unet.py           68      0   100%
    -------------------------------------------
    TOTAL                     143      2    99%

## Training and evaluation

You can either execute full training/evaluation pipeline with
```shell
python main.py
```
or work step by step with running
```shell
dvc repro
```

In this case generation results will be in `gen_sample.png`, while model checkpoint will be saved in `ch.pt`.
