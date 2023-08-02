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

# Custom feature extractor for the policy
from utils.policies import CustomPolicy

# Callback for saving a model based on the training reward
from utils.callbacks import SaveOnBestTrainingRewardCallback

# Constants
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_EVAL = False
DEFAULT_MODEL_NAME = 'landing-aviary-ppo'

# Function to train the model
def train_model(env, checkpoint=None, model_name=DEFAULT_MODEL_NAME):
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(env, log_dir)
    
    if checkpoint is not None and os.path.isfile(checkpoint):
        print("Loading model from checkpoint...")
        model = PPO.load(checkpoint, env)
    else:
        print("Creating new model...")
        model = PPO(CustomPolicy, env, verbose=1, tensorboard_log="./tensorboard/")
    
    callback = SaveOnBestTrainingRewardCallback(check_freq=30000, log_dir=log_dir)
    model.learn(total_timesteps=2_000_000, callback=callback) 
    model.save(os.path.join(model.logger.dir, model_name))


# Function to load the model and evaluate its performance
def evaluate_model(env, gui, record_video, checkpoint):
    if checkpoint is None or not os.path.isfile(checkpoint):
        print("Error: checkpoint does not exist")
        return
    model = PPO.load(checkpoint, env)
    env = LandingAviary(gui=gui, record=record_video)
    obs, info = env.reset(seed=42, options={})
    start = time.time()
    total_reward = 0
    for i in range(10*env.CTRL_FREQ):
        action, _states = model.predict(obs)
        # action = np.array([1,1,1,1])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            print("Episode reward", total_reward)
            time.sleep(5)
            obs, info = env.reset(seed=42, options={})
            break
    env.close()

def run(eval=DEFAULT_EVAL, gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO, model_name=DEFAULT_MODEL_NAME, checkpoint=None):
    env = gym.make("landing-aviary-v0")
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    if eval:
        evaluate_model(env, gui, record_video, checkpoint)
    else:
        train_model(env, checkpoint, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using HoverAviary')
    parser.add_argument('--eval', action='store_true', default=DEFAULT_EVAL, help='Whether to evaluate the model (default: False)')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME, type=str, help='Path to the model (default: "landing-aviary-ppo")', metavar='')
    parser.add_argument('--checkpoint', default=None, type=str, help='Path to a checkpoint to continue training from (zip file)', metavar='')

    ARGS = parser.parse_args()

    run(**vars(ARGS))
