import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from multiprocessing import Process
import multiprocessing
import os
import time

# from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym


class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 64  # Nothing special with 16, feel free to change
        hidden_space2 = 64  # Nothing special with 32, feel free to change
        hidden_space3 = 64  # Nothing special with 32, feel free to change
        


        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.LeakyReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.LeakyReLU(),
            #nn.Linear(hidden_space2, hidden_space3),
            #nn.LeakyReLU(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, params, use_mps = False):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = params[0] #3e-4 #1e-4  # Learning rate for policy optimization
        self.gamma = params[1] #0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        if use_mps and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("metal acceleration enabled")
        else:
            self.device = torch.device("cpu")

        self.net = Policy_Network(obs_space_dims, action_space_dims).to(self.device)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]), device = self.device, dtype=torch.float32)
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()                                                           # action is chosen from normal distribution resulting from neural net
        prob = distrib.log_prob(action)

        action = action.cpu().numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs, dtype=torch.float32, device=self.device)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean().mul(delta).mul(-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []


def run_dqn(proc, params, export = False, use_mps = False):
    """ Status Info """
    time.sleep(proc/10) # make sure outputs are seperate
    print("Process ",proc,"param:", params, "PID:" ,os.getpid())
    #print(params[0], params [1], int(params [2]))

    """ DQN """
    env = gym.make("InvertedDoublePendulum-v4")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 100)  # Records episode-reward
    
    total_num_episodes = int(params[2])
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    seeds = [1, 3, 5, 42]
    rewards_over_seeds = np.zeros((len(seeds),total_num_episodes))

    for index, seed in enumerate(seeds):  # Fibonacci seeds
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize agent every seed
        agent = REINFORCE(obs_space_dims, action_space_dims, params=params, use_mps = use_mps)
        reward_over_episodes = []

        for episode in range(total_num_episodes):
            rewards_seed = []
            # gymnasium v26 requires users to set seed while resetting the environment
            obs, info = wrapped_env.reset(seed=seed)

            done = False
            while not done:
                action = agent.sample_action(obs)

                # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
                # These represent the next observation, the reward from the step,
                # if the episode is terminated, if the episode is truncated and
                # additional info from the step
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                # reward -= (abs(obs[1]) + abs(obs[0]))
                #reward -= obs[0]-np.sin[obs[1]]
                agent.rewards.append(reward)

                # End the episode when either truncated or terminated is true
                #  - truncated: The episode duration reaches max number of timesteps
                #  - terminated: Any of the state space values is no longer finite.
                done = terminated or truncated

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            # update weights (training)
            agent.update()

            if episode % 100 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print("Thread", proc,"Seed", seed,"Episode:", episode, "Average Reward:", avg_reward)

            rewards_seed.append(reward_over_episodes)

        rewards_over_seeds[index] = np.resize(np.array((rewards_seed)),(total_num_episodes,1)).T

        """ Export model"""
        if export:
            model = agent.net
            optimizer = agent.optimizer
            print("Model's state_dict:")
            for param_tensor in model.state_dict():
                print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            # print("Optimizer's state_dict:")
            # for var_name in optimizer.state_dict():
            #     print(var_name, "\t", optimizer.state_dict()[var_name])
            modelname = str(params[0])+"_"+str(params[1])+"_"+str(int(params[2]))
            path = os.path.join(os.getcwd(), "models/"+modelname)

            # Create the subdirectory
            try:
                os.mkdir(path)
                print(f"Subdirectory '{modelname}' created successfully.")
            except OSError as e:
                print(f"Creation of the directory '{modelname}' failed. {e}")

            torch.save(model.state_dict(), "models/"+modelname+"/invert_double_pend_"+modelname+"_"+str(seed)+".pt")
    
    """ Plotting """
    seed_names = [str(i) for i in seeds]
    df1 = pd.DataFrame(rewards_over_seeds.T, columns=seed_names)

    df1 = df1.reset_index()

    # return df1
    plt.rcParams["figure.figsize"] = (10, 5)
    fig, axs = plt.subplots(1)
    sns.set(style="darkgrid", context="talk", palette="rainbow")

    for seed in seed_names:
        # sns.lineplot(x="index", y=seed, data=df1, 
        #                 linewidth = 0.5
        #                 ).set(
        #     title="REINFORCE for InvertedDoublePendulum-v4"
        # )
        sns.scatterplot(x="index", y=seed, data=df1, 
                        linewidth = 0.5,
                        alpha = 0.1,
                        s=20,
                        edgecolor = None, 
                        label = seed
                        ).set(
            title="Thread "+str(proc)+" lr: "+str(params[0])+" $\gamma: $"+str(params[1]),
            xlabel = "episodes",
            ylabel = "reward",
            ylim = [0,1000]
        )

    axs.legend()
    
    plt.show()


def main(): 
    # first ... last hpyerparam set
    # lr = np.array([1e-6, 1e-5, 3e-5, 6e-5, 1e-4 ])
    # gamma = np.array([0.99, 0.99, 0.99, 0.99, 0.99])
    # episodes = np.array([1e4, 1e4, 1e4, 1e4, 1e4])
    lr = np.array([1e-4 ])
    gamma = np.array([0.99])
    episodes = np.array([5e3])
    export = False
    use_mps = True    # metal acceleration (mac silicon)

    hyperparameters = np.array(([lr],[gamma], [episodes])).T
    procs = []

    # start processes
    for index, param in enumerate(hyperparameters):
        param = param.flatten()
        proc = Process(target= run_dqn, args=(index,param,export,use_mps,))
        procs.append(proc)
        proc.start()

    # cleanup
    for proc in procs:
        proc.join()

if __name__ == '__main__':
    main()  