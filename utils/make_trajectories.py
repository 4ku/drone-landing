from typing import Sequence

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from imitation.data import types
from imitation.data.rollout import (
    TrajectoryAccumulator,
    GenTrajTerminationFn,
    unwrap_traj,
)
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


def rpmToNormalizedAction(env, rpm):
    """
    Converts RPM values to normalized actions in the range of [-1, 1].

    Parameters
    ----------
    rpm : ndarray
        (4)-shaped array of ints containing RPM values.

    Returns
    -------
    ndarray
        (4)-shaped array of floats containing normalized actions in the range of [-1, 1].
    """
    action = np.where(
        rpm <= env.HOVER_RPM,
        (rpm / env.HOVER_RPM) - 1,
        (rpm - env.HOVER_RPM) / (env.MAX_RPM - env.HOVER_RPM),
    )
    return action

class LandingExpert:
    def __init__(self, env):
        self.env = env
        self.ctrl = DSLPIDControl(drone_model=DroneModel("cf2x"))
        self.INIT_RPY = np.zeros(3)
        self.init_state()
        self.TARGET_POS = np.zeros(3)

    def init_state(self):
        cur_state = np.zeros(16)
        cur_state[:3] = self.env.INIT_XYZS
        cur_state[6] = 1        
        self.cur_state = cur_state

    def get_action(self):
        cur_pos = self.cur_state[:3]
        next_pos = cur_pos + 0.05 * (self.TARGET_POS - cur_pos) / np.linalg.norm(
            self.TARGET_POS - cur_pos
        )

        if np.linalg.norm(next_pos - cur_pos) > np.linalg.norm(self.TARGET_POS - cur_pos):
            next_pos = self.TARGET_POS

        #### Step the simulation ###################################
        rpm_values, _, _ = self.ctrl.computeControlFromState(
            control_timestep=self.env.TIMESTEP * self.env.AGGR_PHY_STEPS,
            state=self.cur_state,
            target_pos=next_pos,
            target_rpy=self.INIT_RPY,
        )
        return [rpmToNormalizedAction(self.env, rpm_values)]

def generate_trajectories(
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    rng: np.random.Generator,
) -> Sequence[types.TrajectoryWithRew]:

    
    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = TrajectoryAccumulator()
    obs = venv.reset()
    
    env = venv.envs[0].env.env.venv.envs[0].env
    expert = LandingExpert(env)

    for env_idx, ob in enumerate(obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.
    active = np.ones(venv.num_envs, dtype=bool)
    assert isinstance(obs, np.ndarray), "Dict/tuple observations are not supported."
    dones = np.zeros(venv.num_envs, dtype=bool)

    n = 0
    while np.any(active):
        acts = expert.get_action()
        obs, rews, dones, infos = venv.step(acts)
        expert.cur_state = infos[0]["state"]

        assert isinstance(obs, np.ndarray)

        if dones[0]:
            expert.init_state()
            n += 1
            print("Finished episode", n)

        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        print(acts)
        # acts = np.array([[-0.02, -0.02, -0.02, -0.02]])
        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts,
            obs,
            rews,
            dones,
            infos,
        )
        trajectories.extend(new_trajs)

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any
            # environments where a trajectory was completed this timestep.
            active &= ~dones

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)  # type: ignore[arg-type]

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    trajectories = [unwrap_traj(traj) for traj in trajectories]

    return trajectories
