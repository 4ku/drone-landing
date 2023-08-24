import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.base import HardCodedPolicy
from drone_aviary.envs import TakeoffAviary, LandingAviary
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    DummyVecEnv,
    VecTransposeImage,
)
from stable_baselines3.common.monitor import Monitor
import gym

from utils.make_trajectories import (
    generate_trajectories,
)  # Make sure you have this imported

ENV = "takeoff"  # or 'landing'
N_EPISODES = 3
N_EPOCHS = 500
VECTORIZED_ENV = False  # Flag for vectorized env using frame stack


class TakeoffExpert(HardCodedPolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return np.array([1, 1, 1, 1])

    def forward(self, *args):
        pass


class LandingExpert(HardCodedPolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return np.array([-0.02, -0.02, -0.02, -0.02])

    def forward(self, *args):
        pass


class SingleEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)[0]

    def step(self, action):
        observations, rewards, dones, infos = self.env.step([action])
        return observations[0], rewards[0], dones[0], infos[0]


rng = np.random.default_rng(0)

if VECTORIZED_ENV:
    if ENV == "takeoff":
        env = TakeoffAviary(gui=False, record=False)
    elif ENV == "landing":
        env = LandingAviary(gui=False, record=False)
    env = Monitor(env, "./tensorboard/")
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=6)
    env = VecTransposeImage(env)
    env = SingleEnv(env)
    env = RolloutInfoWrapper(env)
    env = DummyVecEnv([lambda: env])
else:
    if ENV == "takeoff":
        env = make_vec_env(
            "TakeoffAviary-v0",
            n_envs=1,
            rng=rng,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        )
    elif ENV == "landing":
        env = make_vec_env(
            "LandingAviary-v0",
            n_envs=1,
            rng=rng,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        )

if ENV == "landing" and VECTORIZED_ENV:
    trajectories = generate_trajectories(
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=N_EPISODES),
        rng=rng,
    )
else:
    if ENV == "takeoff":
        expert = TakeoffExpert(env.observation_space, env.action_space)
    elif ENV == "landing":
        expert = LandingExpert(env.observation_space, env.action_space)
    trajectories = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=N_EPISODES),
        rng=rng,
    )

transitions = rollout.flatten_trajectories(trajectories)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=N_EPOCHS)

reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print("Mean reward:", reward)
