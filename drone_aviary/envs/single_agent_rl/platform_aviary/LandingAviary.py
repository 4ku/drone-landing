import numpy as np
from .PlatformAviary import PlatformAviary


def get_random_position():
    return np.array(
        [
            [
                np.random.uniform(0, 0),
                np.random.uniform(0, 0),
                np.random.uniform(1, 1),
            ]
        ]
    )


class LandingAviary(PlatformAviary):
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
        self.prev_dist = None
        self.INIT_XYZS = get_random_position()
        return super().reset(**kwargs)

    ################################################################################

    # Constants
    TARGET_RADIUS = 0.1
    XYZ_PENALTY_FACTOR = 40
    VEL_PENALTY_FACTOR = 20
    INSIDE_RADIUS_BONUS = 50
    TIME_PENALTY = 0.1
    SOFT_LANDING_VEL = 0.5
    MEDIUM_LANDING_VEL = 2
    LANDING_DIST = 0.5

    def _computeReward(self):
        """Computes the current reward value."""
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(state[:3])
        vel = np.linalg.norm(state[10:13])
        ang_vel = np.linalg.norm(state[13:16])

        # Reward for reducing distance from target
        reward = (
            0
            if self.prev_dist is None
            else self.XYZ_PENALTY_FACTOR * (self.prev_dist - dist)
        )

        if ang_vel > 1.5:
            reward -= 0.1

        # Penalty for high velocity when close to landing
        if dist < self.LANDING_DIST and self.prev_dist is not None and state[2] > 0.05:
            reward += self.VEL_PENALTY_FACTOR * (self.prev_vel - vel)

        # Time penalty for taking too long to decide
        reward -= self.TIME_PENALTY

        # Reward for landing
        if state[2] <= 0.05:
            # Bigger reward if landed within target radius
            if dist < self.TARGET_RADIUS:
                reward += self.INSIDE_RADIUS_BONUS / 2

                # Additional bonus for soft landing
                if vel <= self.SOFT_LANDING_VEL:
                    reward += self.INSIDE_RADIUS_BONUS / 2 + 10
                elif vel <= self.MEDIUM_LANDING_VEL:
                    reward += (
                        (
                            1
                            - (vel - self.SOFT_LANDING_VEL)
                            / (self.MEDIUM_LANDING_VEL - self.SOFT_LANDING_VEL)
                        )
                        * self.INSIDE_RADIUS_BONUS
                        / 2
                    )

        self.prev_dist = dist
        self.prev_vel = vel

        return reward

    ################################################################################

    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """

        state = self._getDroneStateVector(0)
        if state[2] <= 0.03:
            return True
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
