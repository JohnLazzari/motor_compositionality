import torch as th
import numpy as np
from typing import Any
import random
from modules.envs.env_base import MotornetEnv


class Figure8(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(
        self,
        *,
        testing: bool = False,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        # ------------------------------------- START OPTION AND EFFECTOR DEFINITIONS

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get("batch_size", 1)
        reach_conds = options.get("reach_conds", None)
        speed_cond = options.get("speed_cond", None)
        delay_cond = options.get("delay_cond", None)
        custom_delay = options.get("custom_delay", None)
        joint_state = (
            th.tensor(
                [
                    self.effector.pos_range_bound[0] * 0.5
                    + self.effector.pos_upper_bound[0]
                    + 0.1,
                    self.effector.pos_range_bound[1] * 0.5
                    + self.effector.pos_upper_bound[1]
                    + 0.5,
                    0,
                    0,
                ]
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        deterministic: bool = options.get("deterministic", False)
        self.effector.reset(
            options={"batch_size": batch_size, "joint_state": joint_state}
        )

        # ------------------------------------- END

        # ------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

        self.stable_time = 25
        self.hold_time = 25

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        self.delay_time = self.choose_delay(delay_cond, custom_delay)

        # Set up different speeds, use same delay and movement time across batch to keep timesteps the same
        movement_times = list(np.arange(100, 300, 20)) if testing else [100, 200, 300]

        if speed_cond is None:
            self.movement_time = random.choice(movement_times)
        else:
            self.movement_time = movement_times[speed_cond]

        self.half_movement_time = int(self.movement_time / 2)

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time + self.delay_time),
            "movement": (
                self.stable_time + self.delay_time,
                self.stable_time + self.delay_time + self.movement_time,
            ),
            "hold": (
                self.stable_time + self.delay_time + self.movement_time,
                self.stable_time
                + self.delay_time
                + self.movement_time
                + self.hold_time,
            ),
        }

        self.speed_scalar = th.cat(
            [
                th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
                1
                - (self.movement_time / 300)
                * th.ones(
                    size=(
                        batch_size,
                        self.epoch_bounds["hold"][1] - self.epoch_bounds["stable"][1],
                        1,
                    )
                ),
            ],
            dim=1,
        )

        # Set duration
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

        # ------------------------------------- END

        # ------------------------------------- START STATIC NETWORK INPUT (except visual input)

        self.go_cue = th.cat(
            [
                th.zeros(size=(batch_size, self.epoch_bounds["delay"][1], 1)),
                th.ones(
                    size=(
                        batch_size,
                        self.epoch_bounds["hold"][1] - self.epoch_bounds["movement"][0],
                        1,
                    )
                ),
            ],
            dim=1,
        )

        # Now we need rule input
        self.rule_input = th.zeros(size=(batch_size, 10))

        self.rule_input[:, 8] = 1

        # ------------------------------------- END

        # ------------------------------------- START KINEMATIC TRAJECTORY

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        x_points_forward = th.linspace(0, 1, self.half_movement_time)
        y_points_forward = th.sin(th.linspace(0, 2 * np.pi, self.half_movement_time))

        x_points_back = th.linspace(1, 0, self.half_movement_time)
        y_points_back = -th.sin(th.linspace(2 * np.pi, 0, self.half_movement_time))

        # Compute (x, y) coordinates for each angle
        points_forward = (
            th.stack([x_points_forward, y_points_forward], dim=1)
            * 0.25
            * th.tensor([1, 0.5])
        )
        points_back = (
            th.stack([x_points_back, y_points_back], dim=1) * 0.25 * th.tensor([1, 0.5])
        )

        points = th.cat([points_forward, points_back], dim=0)

        # Generate 8 equally spaced angles
        rot_angle = (
            th.linspace(0, 2 * np.pi, 33)[:-1]
            if testing
            else th.linspace(0, 2 * np.pi, 9)[:-1]
        )
        rotated_points = th.zeros(size=(batch_size, self.movement_time, 2))

        # Might be slow because I have to loop through everything
        if reach_conds is None:
            point_idx = th.randint(0, rot_angle.size(0), (batch_size,))
        else:
            if isinstance(reach_conds, (int, float)):
                point_idx = torch.tensor([reach_conds])
            elif isinstance(reach_conds, (th.Tensor, np.ndarray)):
                point_idx = reach_conds

        batch_angles = rot_angle[point_idx]
        for i, theta in enumerate(batch_angles):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # Create the 2D rotation matrix
            R = th.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj

        self.traj = rotated_points + fingertip[:, None, :]

        # We want to start target onset after stable epoch
        self.vis_inp = th.cat(
            [
                # [batch_size, stability timesteps, xy]
                th.zeros(
                    size=(
                        batch_size,
                        self.epoch_bounds["stable"][1],
                        self.traj.shape[-1],
                    )
                ),
                # [batch_size, delay->hold timesteps, xy]
                self.traj[:, self.half_movement_time, :]
                .unsqueeze(1)
                .repeat(
                    1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1
                ),
            ],
            dim=1,
        )

        self.hidden_goal = self.traj[:, 0, :].clone()

        # ------------------------------------- END

        # ------------------------------------- START MOTORNET OBSERVATIONS

        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(
            self.obs_buffer["proprioception"]
        )
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(0, deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.hidden_goal
            if self.differentiable
            else self.detach(self.hidden_goal),
        }

        # ------------------------------------- END

        return obs, info
