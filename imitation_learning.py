import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from drone_aviary.envs import TakeoffAviary
from imitation.policies.base import HardCodedPolicy


class TakeoffExpert(HardCodedPolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

    def forward(self, *args):
        # Always return the optimal action for takeoff
        return np.array([[1, 1, 1, 1]])
    

# Create a random number generator
rng = np.random.default_rng(0)

# Create the environment
# env = TakeoffAviary(gui=False, record=False)
env = make_vec_env(
    "takeoff-aviary-v0",
    n_envs=1,
    rng=rng,  
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)

# Use the expert policy to collect demonstrations
expert = TakeoffExpert(env.observation_space, env.action_space)


rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    rng=rng
)
transitions = rollout.flatten_trajectories(rollouts)

# Train behavioral cloning model
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng
)
bc_trainer.train(n_epochs=1)

# Evaluate the trained policy
reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print("Reward:", reward)



