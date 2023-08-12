import os
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from drone_aviary.envs import TakeoffAviary, LandingAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# Custom feature extractor for the policy
from utils.policies import CustomPolicy

# Callback for saving a model based on the training reward
from utils.callbacks import SaveOnBestTrainingRewardCallback


class DroneTrainer:
    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = True
    DEFAULT_OUTPUT_FOLDER = "results"
    DEFAULT_EVAL = False
    # DEFAULT_ENV = "takeoff"  # or 'landing'
    DEFAULT_ENV = "landing"  # or 'landing'
    DEFAULT_MODEL_TYPE = "PPO"

    def __init__(
        self,
        env_type=DEFAULT_ENV,
        model_type=DEFAULT_MODEL_TYPE,
        eval=DEFAULT_EVAL,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        checkpoint=None,
    ):
        self.env_type = env_type
        self.model_type = model_type
        self.eval = eval
        self.gui = gui
        self.record_video = record_video
        self.checkpoint = checkpoint

        self.model_name = env_type + "_" + model_type
        
        self.log_dir = "./tensorboard/" + env_type
        self.monitor_folder = os.path.join(
            self.log_dir, "monitor", f"env_{os.getpid()}"
        )
        os.makedirs(self.monitor_folder, exist_ok=True)

    def make_env(self, training=True):
        if training:
            self.gui = False
            self.record_video = False

        if self.env_type == "takeoff":
            env = gym.make("takeoff-aviary-v0")
            env = TakeoffAviary(gui=self.gui, record=self.record_video)
        elif self.env_type == "landing":
            env = gym.make("landing-aviary-v0")
            env = LandingAviary(gui=self.gui, record=self.record_video)
        else:
            raise ValueError("Invalid environment type. Choose 'takeoff' or 'landing'.")
        env = Monitor(env, self.monitor_folder) if training else env
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=6)
        env.seed(42)
        print("[INFO] Action space:", env.action_space)
        print("[INFO] Observation space:", env.observation_space)
        return env

    def train_model(self):
        env = self.make_env()
        if self.checkpoint is not None and os.path.isfile(self.checkpoint):
            print("Loading model from checkpoint...")
            model = PPO.load(self.checkpoint, env)
        else:
            print("Creating new model...")
            model = PPO(
                CustomPolicy,
                env,
                n_steps=128,
                n_epochs=5,
                batch_size=32,
                verbose=1,
                tensorboard_log=self.log_dir,
            )
        callback = SaveOnBestTrainingRewardCallback(
            check_freq=20000, log_dir=self.monitor_folder
        )
        model.learn(total_timesteps=1_000_000, callback=callback)
        model.save(os.path.join(model.logger.dir, self.model_name))

    def evaluate_model(self):
        env = self.make_env(training=False)
        if self.checkpoint is None or not os.path.isfile(self.checkpoint):
            print("Error: checkpoint does not exist")
            return
        model = PPO.load(self.checkpoint, env)
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

    def run(self):
        if self.eval:
            self.evaluate_model()
        else:
            self.train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single agent reinforcement learning example script using HoverAviary"
    )
    parser.add_argument(
        "--env_type",
        default=DroneTrainer.DEFAULT_ENV,
        type=str,
        help="Type of environment: 'takeoff' or 'landing' (default: 'takeoff')",
        metavar="",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=DroneTrainer.DEFAULT_EVAL,
        help="Whether to evaluate the model (default: False)",
    )
    parser.add_argument(
        "--gui",
        default=DroneTrainer.DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=DroneTrainer.DEFAULT_RECORD_VIDEO,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--model_type",
        default=DroneTrainer.DEFAULT_MODEL_TYPE,
        type=str,
        help='Path to the model (default: env_type + "_PPO")',
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

    trainer = DroneTrainer(**vars(ARGS))
    trainer.run()
