import torch as th
import numpy as np
from typing import Any
from modules.envs.env_base import MotornetEnv


class CClkCurvedReach(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _configure_task(
        self,
        batch_size,
        testing,
        reach_conds,
        speed_cond,
        delay_cond,
        custom_delay,
        fingertip,
    ):
        self._configure_timing(
            batch_size,
            testing,
            speed_cond,
            delay_cond,
            custom_delay,
            train_movement_times=list(np.arange(50, 150, 5)),
            test_movement_times=list(np.arange(50, 150, 10)),
            speed_denominator=150,
        )
        self._set_static_inputs(batch_size, rule_index=2)

        points = self._arc_points(np.pi, 2 * np.pi, self.movement_time)
        self.traj = self._rotated_trajectory(
            points, fingertip, batch_size, reach_conds, testing
        )
        self._set_visual_input(batch_size, self.traj[:, -1:, :])
        self.hidden_goal = self.traj[:, 0, :].clone()
