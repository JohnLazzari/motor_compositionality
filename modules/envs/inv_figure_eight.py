import torch as th
import numpy as np
from typing import Any
from modules.envs.env_base import MotornetEnv


class InvFigure8(MotornetEnv):
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
            train_movement_times=list(np.arange(100, 300, 40)),
            test_movement_times=list(np.arange(100, 300, 20)),
            speed_denominator=300,
            set_half_movement=True,
        )
        self._set_static_inputs(batch_size, rule_index=9)

        points = self._figure_eight_points(self.half_movement_time, sign=-1)
        self.traj = self._rotated_trajectory(
            points, fingertip, batch_size, reach_conds, testing
        )
        self._set_visual_input(batch_size, self.traj[:, self.half_movement_time, :])
        self.hidden_goal = self.traj[:, 0, :].clone()
