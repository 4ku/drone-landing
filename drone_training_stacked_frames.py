import os
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from drone_landing.envs.single_agent_rl.LandingAviary import LandingAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack

# Custom feature extractor for the policy
from utils.policies import CustomPolicy

# Callback for saving a model based on the training reward
from utils.callbacks import SaveOnBestTrainingRewardCallback

# Constants
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False
DEFAULT_EVAL = False
DEFAULT_MODEL_NAME = "landing-aviary-ppo"


log_dir = "./tensorboard/"
monitor_folder = os.path.join(log_dir, "monitor", f"env_{os.getpid()}")
os.makedirs(monitor_folder, exist_ok=True)


def make_env(gui, record, training=True):
    env = gym.make("landing-aviary-v0")
    env = LandingAviary(gui=gui, record=record)
    env = Monitor(env, monitor_folder) if training else env
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=6)
    env.seed(42)
    return env


def train_model(env, checkpoint=None, model_name=DEFAULT_MODEL_NAME):
    if checkpoint is not None and os.path.isfile(checkpoint):
        print("Loading model from checkpoint...")
        model = PPO.load(checkpoint, env)
    else:
        print("Creating new model...")
        model = PPO(
            CustomPolicy,
            env,
            n_steps=128,
            n_epochs=5,
            batch_size=32,
            verbose=1,
            tensorboard_log=log_dir,
        )

    callback = SaveOnBestTrainingRewardCallback(
        check_freq=30000, log_dir=monitor_folder
    )
    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save(os.path.join(model.logger.dir, model_name))


# Function to load the model and evaluate its performance
def evaluate_model(env, checkpoint):
    if checkpoint is None or not os.path.isfile(checkpoint):
        print("Error: checkpoint does not exist")
        return
    model = PPO.load(checkpoint, env)
    obs = env.reset()

    start = time.time()
    total_reward = 0
    CTRL_FREQ = env.venv.envs[0].CTRL_FREQ
    CTRL_TIMESTEP = env.venv.envs[0].CTRL_TIMESTEP
    for i in range(10 * CTRL_FREQ):
        action, _states = model.predict(obs)
        # action = np.array([[1, 1, 1, 1]])
        obs, reward, terminated, info = env.step(action)

        total_reward += reward
        env.render()
        sync(i, start, CTRL_TIMESTEP)
        if terminated:
            print("Episode reward", total_reward)
            time.sleep(5)
            break
    env.close()


def run(
    eval=DEFAULT_EVAL,
    gui=DEFAULT_GUI,
    record_video=DEFAULT_RECORD_VIDEO,
    model_name=DEFAULT_MODEL_NAME,
    checkpoint=None,
):
    if eval:
        env = make_env(gui, record_video, training=False)
        evaluate_model(env, checkpoint)
    else:
        env = make_env(False, False, training=True)
        train_model(env, checkpoint, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single agent reinforcement learning example script using HoverAviary"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=DEFAULT_EVAL,
        help="Whether to evaluate the model (default: False)",
    )
    parser.add_argument(
        "--gui",
        default=DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=DEFAULT_RECORD_VIDEO,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        type=str,
        help='Path to the model (default: "landing-aviary-ppo")',
        metavar="",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to a checkpoint to continue training from (zip file)",
        metavar="",
    )

    ARGS = parser.parse_args()

    run(**vars(ARGS))
