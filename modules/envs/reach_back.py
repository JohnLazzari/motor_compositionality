import torch as th
import numpy as np
from typing import Any
from modules.envs.env_base import MotornetEnv


class ReachBack(MotornetEnv):
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
            train_movement_times=[100, 200, 300],
            test_movement_times=list(np.arange(100, 300, 20)),
            speed_denominator=300,
            set_half_movement=True,
        )
        self._set_static_inputs(batch_size, rule_index=5)

        angles = self._direction_angles(testing)
        points = self._unit_circle_points(angles)
        point_idx = self._condition_indices(reach_conds, points.size(0), batch_size)
        goal = points[point_idx] * 0.25 + fingertip

        forward_traj = self._line_trajectory(
            fingertip, goal, self.half_movement_time, batch_size
        )
        backward_traj = self._line_trajectory(
            goal, fingertip, self.half_movement_time, batch_size
        )
        self.traj = th.cat([forward_traj, backward_traj], dim=1)
        self._set_visual_input(batch_size, self.traj[:, self.half_movement_time, :])
        self.hidden_goal = self.traj[:, 0, :].clone()
