import os
from collections import defaultdict
from multiprocessing import freeze_support

from torch import multiprocessing
import warnings

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv, ParallelEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

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

    def make_env():
        base_env = GymEnv("InvertedDoublePendulum-v4", device=device)

        env = TransformedEnv(
            base_env,
            Compose(
                # normalize observations
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
            ),
        )

        env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

        return env

    env = ParallelEnv(4, make_env)
    env.reset()

    # print("normalization constant shape:", env.transform[0].loc.shape)

    n = 100
    rollout = env.rollout(n)

    print(f"\nrollout of {n} steps:", rollout)
    print("\nShape of the rollout TensorDict:", rollout.batch_size)
    print("\nMax steps:", rollout["step_count"].max().item())

if __name__ == '__main__':
    main()