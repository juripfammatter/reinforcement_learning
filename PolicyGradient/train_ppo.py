import os
from collections import defaultdict

from torch import multiprocessing

import matplotlib.pyplot as plt
import torch
from PGModels.PPO import PPO


def main():
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass

    is_fork = multiprocessing.get_start_method() == "fork"

    device = (
        "cuda"
        if torch.cuda.is_available() and not is_fork
        # else "mps"
        # if torch.backends.mps.is_available() and not is_fork
        else "cpu"
    )

    print(f"Using device: {device}")
    hyperparameters = {
        "num_cells": 256,  # general
        "lr": 3e-4,  # optimizer
        "max_grad_norm": 1.0,  # optimizer
        "gamma": 0.99,  # advantage
        "lmbda": 0.95,  # advantage
        "entropy_eps": 0.01,  # PPO loss
        "clip_epsilon": 0.2,  # PPO loss
        "sub_batch_size": 64,  # replay buffer
        "num_epochs": 10,  # replay buffer
        "frames_per_batch": 1024,  # data collector
        "total_frames": 1024 * 150,  # data collector
    }
    agent = PPO(hyperparameters, device)
    agent.train()


if __name__ == "__main__":
    main()
