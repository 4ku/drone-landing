import os
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    """
    def __init__(self, check_freq: int, log_file: str):
        super(SaveOnBestTrainingRewardCallback, self).__init__()
        self.check_freq = check_freq
        self.log_file = log_file
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_file), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.best_mean_reward < mean_reward:
                  self.best_mean_reward = mean_reward
                  save_path = os.path.join(self.model.logger.dir, 'best_model')
                  print(f"Saving new best model to {save_path}.zip")
                  self.model.save(save_path)

        return True

