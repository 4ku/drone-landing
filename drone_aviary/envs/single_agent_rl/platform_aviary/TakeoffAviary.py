import numpy as np
from .PlatformAviary import PlatformAviary


def get_random_position():
    return np.array(
        [
            [
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(0.05, 0.05),
            ]
        ]
    )


class TakeoffAviary(PlatformAviary):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 5

    def reset(self, **kwargs):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """
        self.INIT_XYZS = get_random_position()
        return super().reset(**kwargs)

    ################################################################################


    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        if state[2] < 0.03:
            return -5
        return 0.1 * state[2]


    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if state[2] < 0.03:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
