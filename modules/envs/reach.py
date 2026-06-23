import torch as th
import numpy as np
from typing import Any
from modules.envs.env_base import MotornetEnv


class Reach(MotornetEnv):
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
            train_movement_times=list(np.arange(50, 150, 20)),
            test_movement_times=list(np.arange(50, 150, 10)),
            speed_denominator=150,
        )
        self._set_static_inputs(batch_size, rule_index=0)

        angles = self._direction_angles(testing)
        points = self._unit_circle_points(angles)
        point_idx = self._condition_indices(reach_conds, points.size(0), batch_size)
        goal = points[point_idx] * 0.25 + fingertip

        self.traj = self._line_trajectory(
            fingertip, goal, self.movement_time, batch_size
        )
        self._set_visual_input(batch_size, self.traj[:, -1:, :])
        self.hidden_goal = self.traj[:, 0, :].clone()
