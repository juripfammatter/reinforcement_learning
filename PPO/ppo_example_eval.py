import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing
from torch import nn
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal

""" Device """
is_fork = multiprocessing.get_start_method() == "fork"

# device = (
#     "cuda"
#     if torch.cuda.is_available() and not is_fork
#     # else "mps"
#     # if torch.backends.mps.is_available() and not is_fork
#     else "cpu"
# )
device = "cpu"

print(f"Using device: {device}")

""" Hyperparameters """

num_cells = 256  # number of cells in each layer i.e. output dim.

""" Environment """

base_env = GymEnv("InvertedDoublePendulum-v4", device=device, render_mode="human")

env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)

# initialize stats -> random motion in render
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

# print("normalization constant shape:", env.transform[0].loc.shape)
# print("\nobservation_spec:", env.observation_spec)
# print("\nreward_spec:", env.reward_spec)
# print("\ninput_spec:", env.input_spec)
# print("\naction_spec (as defined by input_spec):", env.action_spec)

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
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True,
)

""" load policy """
total_frames = 1_000_000
model_weights_filename = f"models/ppo_example_model_weights_{total_frames//1000}k.pth"
policy_module.load_state_dict(torch.load(model_weights_filename, weights_only=True))

""" Evaluate """
torch.manual_seed(37)
env.set_seed(37)

policy_module.eval()
for _ in range(3):
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        env.reset()
        eval_rollout = env.rollout(1000, policy_module)
        print("reward: ", eval_rollout["next", "reward"].mean().item())
        print("action: ", eval_rollout["action"].mean().item())
        print("step count: ", eval_rollout["step_count"].max().item())

        del eval_rollout
