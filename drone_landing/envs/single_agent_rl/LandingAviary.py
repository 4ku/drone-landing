import os
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from drone_landing.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
    BaseSingleAgentAviary,
)
import pybullet as p


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

class LandingAviary(BaseSingleAgentAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    TARGET_RADIUS = 0.1
    XYZ_PENALTY_FACTOR = 10
    VEL_PENALTY_FACTOR = 20
    INSIDE_RADIUS_BONUS = 60
    VEL = 0
    LANDING_Z_ZONE = 1

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 40,
        ctrl_freq: int = 20,
        gui=False,
        record=False,
        obs: ObservationType = ObservationType.RGB,
        act: ActionType = ActionType.RPM,
    ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """

        super().__init__(
            drone_model=drone_model,
            initial_xyzs=get_random_position(),
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )
        self.EPISODE_LEN_SEC = 1
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
        return super().reset(kwargs)

    def _addObstacles(self):
        """Add obstacles to the environment.
        These obstacles are loaded from standard URDF files included in Bullet.
        """
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)
        p.setAdditionalSearchPath(
            parent_directory, physicsClientId=self.CLIENT
        )

        p.loadURDF(
            "data/aruco.urdf",
            [0, 0, 0.005],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT,
        )
        p.loadURDF(
            "data/platform.urdf",
            [0, 0, 0.002],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT,
        )
        p.loadURDF(
            "data/wall.urdf",
            [5, 0, 5],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT,
        )
        p.loadURDF(
            "data/wall.urdf",
            [-5, 0, 5],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT,
        )
        p.loadURDF(
            "data/wall.urdf",
            [0, 5, 5],
            p.getQuaternionFromEuler([0, 0, np.pi / 2]),
            physicsClientId=self.CLIENT,
        )
        self.k = p.loadURDF(
            "data/wall.urdf",
            [0, -5, 5],
            p.getQuaternionFromEuler([0, 0, np.pi / 2]),
            physicsClientId=self.CLIENT,
        )
        p.loadURDF(
            "data/wall.urdf",
            [0, 0, 10],
            p.getQuaternionFromEuler([0, np.pi / 2, 0]),
            physicsClientId=self.CLIENT,
        )

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        # return -1 * np.linalg.norm(np.array([0, 0, 1])-state[0:3])**2
        return 0.1 * state[2]

    # def _computeReward(self):
    #     """Computes the current reward value.
    #     Returns
    #     -------
    #     float
    #         The reward.
    #     """
    #     state = self._getDroneStateVector(0)
    #     dist = np.linalg.norm(state[:3])
    #     vel = np.linalg.norm(state[10:13])

    #     # Penalty based on state
    #     penalty = -(self.XYZ_PENALTY_FACTOR * (dist + dist**2))

    #     # Compute reward based on diff between previous penalty and current
    #     reward = ((penalty - self.prev_penalty)
    #               if self.prev_penalty is not None
    #               else 0)

    #     # Compute less velocity for safe landing
    #     if state[2] < self.LANDING_Z_ZONE and (self.prev_penalty is not None):
    #         reward += self.VEL_PENALTY_FACTOR * (self.prev_vel - vel)

    #     # To faster land (increase time)
    #     reward -= 0.1

    #     self.prev_penalty = penalty
    #     self.prev_vel = vel
    #     self.VEL = vel

    #     if state[2] <= 0.05:
    #         # Add big reward when land safely between the radious
    #         if np.linalg.norm(state[:3]) < self.TARGET_RADIUS:
    #             reward += self.INSIDE_RADIUS_BONUS/2

    #             if vel <= 0.5:
    #                 reward += self.INSIDE_RADIUS_BONUS/2 + 10
    #             elif vel <= 2:
    #                 reward += (1 - (vel-0.5)/1.5) * self.INSIDE_RADIUS_BONUS/2

    #     return reward
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
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused in this implementation.

        Returns
        -------
        bool
            Always false.

        """
        return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {}

    ################################################################################

    def _clipAndNormalizeState(self, state):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(
                state,
                clipped_pos_xy,
                clipped_pos_z,
                clipped_rp,
                clipped_vel_xy,
                clipped_vel_z,
            )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = (
            state[13:16] / np.linalg.norm(state[13:16])
            if np.linalg.norm(state[13:16]) != 0
            else state[13:16]
        )

        norm_and_clipped = np.hstack(
            [
                normalized_pos_xy,
                normalized_pos_z,
                state[3:7],
                normalized_rp,
                normalized_y,
                normalized_vel_xy,
                normalized_vel_z,
                normalized_ang_vel,
                state[16:20],
            ]
        ).reshape(
            20,
        )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(
        self,
        state,
        clipped_pos_xy,
        clipped_pos_z,
        clipped_rp,
        clipped_vel_xy,
        clipped_vel_z,
    ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
                    state[0], state[1]
                ),
            )
        if not (clipped_pos_z == np.array(state[2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(
                    state[2]
                ),
            )
        if not (clipped_rp == np.array(state[7:9])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                    state[7], state[8]
                ),
            )
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                    state[10], state[11]
                ),
            )
        if not (clipped_vel_z == np.array(state[12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(
                    state[12]
                ),
            )
