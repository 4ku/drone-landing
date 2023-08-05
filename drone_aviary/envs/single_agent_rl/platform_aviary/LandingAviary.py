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


class LandingAviary(PlatformAviary):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 5
        self.prev_penalty = None

    def reset(self, **kwargs):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """
        self.prev_penalty = None
        self.INIT_XYZS = get_random_position()
        return super().reset(**kwargs)

    ################################################################################

    TARGET_RADIUS = 0.1
    XYZ_PENALTY_FACTOR = 10
    VEL_PENALTY_FACTOR = 20
    INSIDE_RADIUS_BONUS = 60
    VEL = 0
    LANDING_Z_ZONE = 1

    def _computeReward(self):
        """Computes the current reward value.
        Returns
        -------
        float
            The reward.
        """
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(state[:3])
        vel = np.linalg.norm(state[10:13])

        # Penalty based on state
        penalty = -(self.XYZ_PENALTY_FACTOR * (dist + dist**2))

        # Compute reward based on diff between previous penalty and current
        reward = (penalty - self.prev_penalty) if self.prev_penalty is not None else 0

        # Compute less velocity for safe landing
        if state[2] < self.LANDING_Z_ZONE and (self.prev_penalty is not None):
            reward += self.VEL_PENALTY_FACTOR * (self.prev_vel - vel)

        # To faster land (increase time)
        reward -= 0.1

        self.prev_penalty = penalty
        self.prev_vel = vel
        self.VEL = vel

        if state[2] <= 0.05:
            # Add big reward when land safely between the radious
            if np.linalg.norm(state[:3]) < self.TARGET_RADIUS:
                reward += self.INSIDE_RADIUS_BONUS / 2

                if vel <= 0.5:
                    reward += self.INSIDE_RADIUS_BONUS / 2 + 10
                elif vel <= 2:
                    reward += (1 - (vel - 0.5) / 1.5) * self.INSIDE_RADIUS_BONUS / 2

        return reward

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        # if self.step_counter / self.PYB_FREQ >= self.EPISODE_LEN_SEC:
        #     self.done = True
        #     return True

        # state = self._getDroneStateVector(0)
        # # Stop conditions in reaching target point
        # # self.done = state[2] <= 0.05
        # return self.done
        state = self._getDroneStateVector(0)
        if state[2] < 0.03:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
