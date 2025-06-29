import os
import json
import sys

import torch
from torch import multiprocessing
from PGModels.PPO import PPO


def parse_config(config_file: str) -> dict:
    """Import JSON file and check existence"""
    if os.path.exists(config_file):
        print(f"Using {config_file} as config file")

    with open(config_file, "r") as file:
        config = json.load(file)
    return config


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

    hyperparameters = parse_config(sys.argv[1])
    agent = PPO(hyperparameters, device)
    agent.train()


if __name__ == "__main__":
    main()
