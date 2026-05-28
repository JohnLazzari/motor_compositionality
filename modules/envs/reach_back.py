import torch as th
import numpy as np
from typing import Any
from modules.envs.env_base import MotornetEnv


class ReachBack(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(
        self,
        *,
        testing: bool = False,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        (
            _,
            batch_size,
            reach_conds,
            speed_cond,
            delay_cond,
            custom_delay,
            deterministic,
            joint_state,
        ) = self._reset_trial_options(seed, options)
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

        fingertip = self._fingertip_from_joint_state(joint_state)
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
        return self._finalize_reset(batch_size, deterministic)
