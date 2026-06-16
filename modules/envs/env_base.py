import torch as th
import numpy as np
from typing import Any
from motornet import environment as env
import random


class MotornetEnv(env.Environment):
    """A reach to a random target from a random starting position.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass. This is the network that will perform
        the task.
        name: `String`, the name of the task object instance.
        deriv_weight: `Float`, the weight of the muscle activation's derivative contribution to the default muscle L2
        loss.
        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """

    def __init__(self, *args, **kwargs):
        # MotorNet builds spaces by calling reset() during its constructor.
        self.zero_feedback = kwargs.pop("zero_feedback", False)
        if type(self.zero_feedback) is not bool:
            raise TypeError("zero_feedback must be a boolean")
        self.stable_time = 25
        self.hold_time = 25
        super().__init__(*args, **kwargs)
        self.obs_noise = [0.0] * self._get_obs_size()
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None
        # Initialize empty 3d trajectory
        self.traj = th.empty(size=(1, 1, 1))

    def get_obs(
        self, t, action=None, deterministic: bool = False
    ) -> th.Tensor | np.ndarray:
        """
        Returns a `(batch_size, n_features)` `tensor` containing the (potientially time-delayed) observations.
        By default, this is the task goal, followed by the output of the :meth:`get_proprioception()` method,
        the output of the :meth:`get_vision()` method, and finally the last :attr:`action_frame_stacking` action sets,
        if a non-zero `action_frame_stacking` keyword argument was passed at initialization of this class instance.
        `.i.i.d.` Gaussian noise is added to each element in the `tensor`,
        using the :attr:`obs_noise` attribute.
        """
        self.update_obs_buffer(action=action)

        assert self.go_cue is not None

        obs_as_list = [
            self.rule_input,
            self.speed_scalar[:, t],
            self.go_cue[:, t],
            self.vis_inp[:, t],
        ]
        if not self.zero_feedback:
            obs_as_list.append(self.obs_buffer["vision"][0])
            obs_as_list.append(self.obs_buffer["proprioception"][0])
        obs_as_list += self.obs_buffer["action"][: self.action_frame_stacking]

        obs = th.cat(obs_as_list, dim=-1)

        if deterministic is False:
            noise = self.obs_noise
            if len(noise) != obs.shape[-1]:
                noise = [0.0] * obs.shape[-1]
            obs = self.apply_noise(obs, noise=noise)

        return obs if self.differentiable else self.detach(obs)

    def _get_obs_size(self):
        base_size = 14
        feedback_size = 0
        if not self.zero_feedback:
            feedback_size += self.skeleton.space_dim
            feedback_size += 2 * self.effector.n_muscles
        action_size = self.effector.n_muscles * self.action_frame_stacking
        return base_size + feedback_size + action_size

    def step(
        self,
        t,
        action: th.Tensor | np.ndarray,
        **kwargs,
    ) -> tuple[th.Tensor | np.ndarray, None | np.ndarray, bool, dict[str, Any]]:
        """
        Perform one simulation step. This method is likely to be overwritten by any subclass to implement user-defined
        computations, such as reward value calculation for reinforcement learning, custom truncation or termination
        conditions, or time-varying goals.

        Args:
        action: `Tensor` or `numpy.ndarray`, the input drive to the actuators.
        deterministic: `Boolean`, whether observation, action, proprioception, and vision noise are applied.
        **kwargs: This is passed as-is to the :meth:`motornet.effector.Effector.step()` call. This is maily useful to pass
        `endpoint_load` or `joint_load` kwargs.

        Returns:
        - The observation vector as `tensor` or `numpy.ndarray`, if the :class:`Environment` is set as differentiable or
            not, respectively. It has dimensionality `(batch_size, n_features)`.
        - A `numpy.ndarray` with the reward information for the step, with dimensionality `(batch_size, 1)`. This is
            `None` if the :class:`Environment` is set as differentiable. By default this always returns `0.` in the
            :class:`Environment`.
        - A `boolean` indicating if the simulation has been terminated or truncated. If the :class:`Environment` is set as
            differentiable, this returns `True` when the simulation time reaches `max_ep_duration` provided at
            initialization.
        - A `boolean` indicating if the simulation has been truncated early or not. This always returns `False` if the
            :class:`Environment` is set as differentiable.
        - A `dictionary` containing this step's information.
        """

        action = (
            action
            if th.is_tensor(action)
            else th.tensor(action, dtype=th.float32).to(self.device)
        )
        noisy_action = action

        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = (
            None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        )
        terminated = bool(t >= self.max_ep_duration)
        t_delay_shifted = t - self.epoch_bounds["movement"][0]

        """
            Each stage of the trial is given here
            Trajectory only specifies the movement kinematics, the stable, delay, and hold periods 
            simply repeat the first and last hand positions
        """
        if t < self.epoch_bounds["delay"][1]:
            self.hidden_goal = self.traj[:, 0, :]
        elif (
            t >= self.epoch_bounds["movement"][0]
            and t < self.epoch_bounds["movement"][1]
        ):
            self.hidden_goal = self.traj[:, t_delay_shifted, :].clone()
        elif t >= self.epoch_bounds["hold"][0]:
            self.hidden_goal = self.traj[:, -1, :].clone()

        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal
            if self.differentiable
            else self.detach(self.hidden_goal),
        }

        return obs, reward, terminated, info

    def choose_delay(self, delay_cond, custom_delay):
        delay_times = [25, 50, 75]
        if delay_cond is not None:
            delay_time = delay_times[delay_cond]
        elif custom_delay is not None:
            # Should be an integer
            delay_time = custom_delay
        else:
            delay_time = random.choice(delay_times)
        return delay_time

    def _reset_trial_options(self, seed, options):
        """Parse reset options, reset the effector, and return common values."""
        self._set_generator(seed=seed)
        options = {} if options is None else options
        batch_size: int = options.get("batch_size", 1)
        joint_state = self._initial_joint_state(batch_size)
        deterministic: bool = options.get("deterministic", False)
        self.effector.reset(
            options={"batch_size": batch_size, "joint_state": joint_state}
        )
        return (
            options,
            batch_size,
            options.get("reach_conds", None),
            options.get("speed_cond", None),
            options.get("delay_cond", None),
            options.get("custom_delay", None),
            deterministic,
            joint_state,
        )

    def _initial_joint_state(self, batch_size):
        """Return the default repeated initial joint state used by all tasks."""
        return (
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

    def _configure_timing(
        self,
        batch_size,
        testing,
        speed_cond,
        delay_cond,
        custom_delay,
        train_movement_times,
        test_movement_times,
        speed_denominator,
        set_half_movement=False,
    ):
        """Configure epoch bounds, speed scalar, and episode duration."""
        self.delay_time = self.choose_delay(delay_cond, custom_delay)

        movement_times = test_movement_times if testing else train_movement_times
        if speed_cond is None:
            self.movement_time = random.choice(movement_times)
        else:
            self.movement_time = movement_times[speed_cond]

        if set_half_movement:
            self.half_movement_time = int(self.movement_time / 2)

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

        speed_scale = 1 - (self.movement_time / speed_denominator)
        self.speed_scalar = th.cat(
            [
                th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
                speed_scale
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
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

    def _set_static_inputs(self, batch_size, rule_index):
        """Set the go cue and one-hot rule input for a reset."""
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
        self.rule_input = th.zeros(size=(batch_size, 10))
        self.rule_input[:, rule_index] = 1

    def _fingertip_from_joint_state(self, joint_state):
        """Convert a joint state to the fingertip position used as trajectory origin."""
        return self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

    def _direction_angles(self, testing):
        """Return the direction grid used for train or test resets."""
        if testing:
            return th.linspace(0, 2 * np.pi, 33)[:-1]
        return th.linspace(0, 2 * np.pi, 9)[:-1]

    def _condition_indices(self, reach_conds, num_conditions, batch_size):
        """Return condition indices from explicit options or random sampling."""
        if reach_conds is None:
            return th.randint(0, num_conditions, (batch_size,))
        if isinstance(reach_conds, (int, float)):
            return th.tensor([reach_conds])
        if isinstance(reach_conds, (th.Tensor, np.ndarray)):
            return reach_conds
        raise TypeError

    def _unit_circle_points(self, angles):
        """Return unit-circle xy points for a tensor of angles."""
        return th.stack([th.cos(angles), th.sin(angles)], dim=1)

    def _arc_points(self, start_angle, end_angle, movement_time):
        """Return scaled arc trajectory points between two angles."""
        traj_points = th.linspace(start_angle, end_angle, movement_time)
        points = self._unit_circle_points(traj_points)
        return (points + th.tensor([[1, 0]])) * 0.25 * 0.5

    def _sinusoid_points(self, movement_time, sign=1):
        """Return scaled sinusoidal trajectory points."""
        x_points = th.linspace(0, 1, movement_time)
        y_points = sign * th.sin(th.linspace(0, 2 * np.pi, movement_time))
        return th.stack([x_points, y_points], dim=1) * 0.25 * th.tensor([1, 0.5])

    def _figure_eight_points(self, half_movement_time, sign=1):
        """Return scaled figure-eight trajectory points."""
        x_points_forward = th.linspace(0, 1, half_movement_time)
        y_points_forward = sign * th.sin(th.linspace(0, 2 * np.pi, half_movement_time))

        x_points_back = th.linspace(1, 0, half_movement_time)
        y_points_back = -sign * th.sin(th.linspace(2 * np.pi, 0, half_movement_time))

        points_forward = (
            th.stack([x_points_forward, y_points_forward], dim=1)
            * 0.25
            * th.tensor([1, 0.5])
        )
        points_back = (
            th.stack([x_points_back, y_points_back], dim=1) * 0.25 * th.tensor([1, 0.5])
        )
        return th.cat([points_forward, points_back], dim=0)

    def _rotated_trajectory(self, points, fingertip, batch_size, reach_conds, testing):
        """Rotate points according to condition choices and center them on fingertip."""
        rot_angle = self._direction_angles(testing)
        rotated_points = th.zeros(size=(batch_size, points.shape[0], 2))
        point_idx = self._condition_indices(reach_conds, rot_angle.size(0), batch_size)

        for i, theta in enumerate(rot_angle[point_idx]):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            rotation = th.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            rotated_points[i] = (rotation @ points.T).T

        return rotated_points + fingertip[:, None, :]

    def _line_trajectory(self, start, end, steps, batch_size):
        """Return a batched straight-line trajectory from start to end."""
        weights = th.linspace(0, 1, steps).repeat(batch_size, 1)
        x_points = start[:, None, 0] + weights * (end[:, None, 0] - start[:, None, 0])
        y_points = start[:, None, 1] + weights * (end[:, None, 1] - start[:, None, 1])
        return th.stack([x_points, y_points], dim=-1)

    def _set_visual_input(self, batch_size, target):
        """Set visual input from a stable zero period and repeated target."""
        if target.dim() == 2:
            target = target.unsqueeze(1)
        self.vis_inp = th.cat(
            [
                th.zeros(
                    size=(
                        batch_size,
                        self.epoch_bounds["stable"][1],
                        self.traj.shape[-1],
                    )
                ),
                target.repeat(
                    1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1
                ),
            ],
            dim=1,
        )

    def _kinematic_targets(self):
        """Return the full target trajectory without stepping a MotorNet effector."""
        return th.cat(
            [
                self.traj[:, :1, :].repeat(1, self.epoch_bounds["delay"][1], 1),
                self.traj,
                self.traj[:, -1:, :].repeat(1, self.hold_time, 1),
            ],
            dim=1,
        )

    def _supervised_observations(self, inp_size):
        """Build observations with arm feedback and previous-action inputs set to zero."""
        obs = th.cat(
            [
                self.rule_input[:, None, :].repeat(1, self.speed_scalar.shape[1], 1),
                self.speed_scalar,
                self.go_cue,
                self.vis_inp,
            ],
            dim=-1,
        )
        if obs.shape[-1] > inp_size:
            raise ValueError(
                f"inp_size must be at least {obs.shape[-1]} for kinematic training"
            )

        padding = th.zeros((*obs.shape[:-1], inp_size - obs.shape[-1]))
        return th.cat([obs, padding], dim=-1)

    @classmethod
    def generate_supervised_trial(
        cls,
        batch_size,
        testing=False,
        reach_conds=None,
        speed_cond=None,
        delay_cond=None,
        custom_delay=None,
        inp_size=28,
    ):
        """Generate task inputs and xy targets without constructing an arm."""
        task = cls.__new__(cls)
        task.stable_time = 25
        task.hold_time = 25
        fingertip = th.zeros(size=(batch_size, 2))
        task._configure_task(
            batch_size,
            testing,
            reach_conds,
            speed_cond,
            delay_cond,
            custom_delay,
            fingertip,
        )
        return (
            task._supervised_observations(inp_size),
            task._kinematic_targets(),
            task.epoch_bounds,
        )

    def _finalize_reset(self, batch_size, deterministic):
        """Initialize observation buffers and return the reset observation/info."""
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
        return obs, info

    def reset(
        self,
        *,
        testing: bool = False,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """
        Uses the :meth:`Environment.reset()` method of the parent class :class:`Environment` that can be overwritten to
        change the returned data. Here the goals (`i.e.`, the targets) are drawn from a random uniform distribution across
        the full joint space.
        """
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
        fingertip = self._fingertip_from_joint_state(joint_state)
        self._configure_task(
            batch_size,
            testing,
            reach_conds,
            speed_cond,
            delay_cond,
            custom_delay,
            fingertip,
        )
        return self._finalize_reset(batch_size, deterministic)

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
        """Configure the task timing, inputs, and xy trajectory."""
        raise NotImplementedError
