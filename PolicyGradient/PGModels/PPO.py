import os
from collections import defaultdict

from torch import multiprocessing

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    ParallelEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import typing


class PPO(object):
    def __init__(self, hyperparams: dict, device: str = "cpu"):
        """
        Args:
            hyperparams (dict): Hyperparameters for the algorithm. Needs to contain the following keys:
                - num_cells (int):
                - gamma (float):
                - lmbda (float):
                - clip_epsilon (float):
                - entropy_eps (float):
                - frames_per_batch (int):
                - total_frames (int):
                - lr (float):

            device (str): device to run the algorithm on (default: "cpu")

        """
        self.device = device
        self.hp = hyperparams

        self.env = self._get_env()
        self.policy_module = self._get_policy()
        self.value_function = self._get_value_function()

        # Initializes policy and value net
        print("\nPolicy Net:", self.policy_module(self.env.reset()))
        print("\nValue Net:", self.value_function(self.env.reset()))

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal(m.weight)
                m.bias.data.fill_(0.01)

        self.policy_module.apply(init_weights)
        self.value_function.apply(init_weights)

        self.data_collector = self._get_data_collector()
        self.replay_buffer = self._get_replay_buffer()
        self.advantage_module = self._get_advantage_module()
        self._init_optimizer_and_loss()

    def _get_env(self, render_mode: str | None = None) -> TransformedEnv:
        if render_mode is not None:
            base_env = GymEnv(
                "InvertedDoublePendulum-v4", device=self.device, render_mode=render_mode
            )
        else:
            base_env = GymEnv("InvertedDoublePendulum-v4", device=self.device)

        env = TransformedEnv(
            base_env,
            Compose(
                # normalize observations
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
            ),
        )

        print(f"initializing environment, this could take a while...")
        env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

        return env

    def _get_policy(self) -> ProbabilisticActor:
        actor_net = nn.Sequential(
            nn.LazyLinear(self.hp["num_cells"], device=self.device),
            nn.Tanh(),
            # nn.LazyLinear(self.hp["num_cells"], device=self.device),
            # nn.Tanh(),
            # nn.LazyLinear(self.hp["num_cells"], device=self.device),
            # nn.Tanh(),
            nn.LazyLinear(2 * self.env.action_spec.shape[-1], device=self.device),
            NormalParamExtractor(),
        )

        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )

        policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": self.env.action_spec.space.low,
                "high": self.env.action_spec.space.high,
            },
            return_log_prob=True,
            # we'll need the log-prob for the numerator of the importance weights
        )

        return policy_module

    def _get_value_function(self) -> ValueOperator:
        value_net = nn.Sequential(
            nn.LazyLinear(self.hp["num_cells"], device=self.device),
            nn.Tanh(),
            # nn.LazyLinear(self.hp["num_cells"], device=self.device),
            # nn.Tanh(),
            # nn.LazyLinear(self.hp["num_cells"], device=self.device),
            # nn.Tanh(),
            nn.LazyLinear(1, device=self.device),
        )

        value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        )

        return value_module

    def _get_data_collector(self):
        collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=self.hp["frames_per_batch"],
            total_frames=self.hp["total_frames"],
            split_trajs=False,
            device=self.device,
        )
        return collector

    def _get_replay_buffer(self):
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.hp["frames_per_batch"]),
            sampler=SamplerWithoutReplacement(),
        )
        return replay_buffer

    def _get_advantage_module(self):
        advantage_module = GAE(
            gamma=self.hp["gamma"],
            lmbda=self.hp["lmbda"],
            value_network=self.value_function,
            average_gae=True,
        )
        return advantage_module

    def _get_optimizer(self):
        """
        Requires self.loss_module to be initialized
        """
        optim = torch.optim.Adam(self.loss_module.parameters(), self.hp["lr"])
        return optim

    def _get_loss_module(self):

        loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_function,
            clip_epsilon=self.hp["clip_epsilon"],
            entropy_bonus=bool(self.hp["entropy_eps"]),
            entropy_coef=self.hp["entropy_eps"],
            # these keys match by default, but we set this for completeness
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )
        return loss_module

    def _get_lr_scheduler(self):
        """
        Requires self.optimizer to be initialized
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.hp["total_frames"] // self.hp["frames_per_batch"], 0.0
        )
        return scheduler

    def _init_optimizer_and_loss(self):
        self.loss_module = self._get_loss_module()
        self.optimizer = self._get_optimizer()
        self.lr_scheduler = self._get_lr_scheduler()

    @staticmethod
    def _log_rollout(eval_rollout: torch.Tensor, logs: defaultdict):
        logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
        logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
        logs["eval step_count"].append(eval_rollout["step_count"].max().item())
        eval_str = (
            f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
            f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
            f"eval step-count: {logs['eval step_count'][-1]}"
        )
        return eval_str

    def train(self):
        logs = defaultdict(list)
        pbar = tqdm(total=self.hp["total_frames"] * self.hp["num_epochs"])

        torch.manual_seed(42)
        self.env.set_seed(42)

        # iterate over batches
        for i, tensordict_data in enumerate(self.data_collector):
            for _ in range(self.hp["num_epochs"]):
                # Compute advantage
                self.advantage_module(tensordict_data)

                # Add step to replay buffer
                self.replay_buffer.extend(tensordict_data.reshape(-1).cpu())

                # Calculate Loss and backprop
                for _ in range(
                    self.hp["frames_per_batch"] // self.hp["sub_batch_size"]
                ):
                    subdata = self.replay_buffer.sample(self.hp["sub_batch_size"])
                    loss_vals = self.loss_module(subdata.to(self.device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), self.hp["max_grad_norm"]
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Policy evaluation (every n batches)
                if i % 10 == 0:
                    # Execute rollout without exploration
                    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                        eval_rollout = self.env.rollout(1000, self.policy_module)
                        eval_str = self._log_rollout(eval_rollout, logs)
                        del eval_rollout

                # logging
                pbar.update(tensordict_data.numel())
                pbar.set_description(", ".join([eval_str]))

                # step learning rate
                self.lr_scheduler.step()

        # Sanity check
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.plot(logs["eval reward"])
        plt.title("Evaluation rewards")
        plt.subplot(1, 2, 2)
        plt.plot(logs["eval step_count"])
        plt.title("Evaluation step counts")
        plt.show()
