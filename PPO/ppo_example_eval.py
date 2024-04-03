from collections import defaultdict
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
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

import gymnasium as gym

""" Device """
is_fork = multiprocessing.get_start_method() == "fork"

device = (
    "cuda"
    if torch.cuda.is_available() and not is_fork
    # else "mps"
    # if torch.backends.mps.is_available() and not is_fork
    else "cpu"
)

print(f"Using device: {device}")

""" Hyperparameters """

num_cells = 256  # number of cells in each layer i.e. output dim.

""" Environment """

base_env = GymEnv("InvertedDoublePendulum-v4", device=device)  # , render_mode="human")

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

print("normalization constant shape:", env.transform[0].loc.shape)
print("\nobservation_spec:", env.observation_spec)
print("\nreward_spec:", env.reward_spec)
print("\ninput_spec:", env.input_spec)
print("\naction_spec (as defined by input_spec):", env.action_spec)

""" Policy """
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)


policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.action_spec.space.low,
        "max": env.action_spec.space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

""" load policy """
model_weights_filename = "models/ppo_example_model_weights_300k.pth"
# print(policy_module.state_dict())
# print(actor_net.state_dict())
policy_module.load_state_dict(torch.load(model_weights_filename))
# print(policy_module.state_dict())
# print(actor_net.state_dict())


""" Evaluate """
torch.manual_seed(37)

actor_net.eval()
policy_module.eval()
with torch.no_grad():
    for _ in range(10):
        eval_rollout = env.rollout(1000, policy_module)
        # print(eval_rollout)
        print("reward: ", eval_rollout["next", "reward"].mean().item())
        print("action: ", eval_rollout["action"].mean().item())
        print("step count: ", eval_rollout["step_count"].max().item())

        del eval_rollout
