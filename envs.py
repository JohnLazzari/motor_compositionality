import torch as th
import numpy as np
from typing import Any
from motornet import environment as env
from itertools import product
import random

# Comments in first environment (below) should follow for others
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
        super().__init__(*args, **kwargs)
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None

    def get_obs(self, t, action=None, deterministic: bool = False) -> th.Tensor | np.ndarray:
        """
        Returns a `(batch_size, n_features)` `tensor` containing the (potientially time-delayed) observations.
        By default, this is the task goal, followed by the output of the :meth:`get_proprioception()` method, 
        the output of the :meth:`get_vision()` method, and finally the last :attr:`action_frame_stacking` action sets,
        if a non-zero `action_frame_stacking` keyword argument was passed at initialization of this class instance.
        `.i.i.d.` Gaussian noise is added to each element in the `tensor`,
        using the :attr:`obs_noise` attribute.
        """
        self.update_obs_buffer(action=action)

        obs_as_list = [
        self.rule_input,
        self.speed_scalar[:, t],
        self.go_cue[:, t],
        self.vis_inp[:, t],
        self.obs_buffer["vision"][0],
        self.obs_buffer["proprioception"][0],
        ] + self.obs_buffer["action"][:self.action_frame_stacking]
        
        obs = th.cat(obs_as_list, dim=-1)

        if deterministic is False:
            obs = self.apply_noise(obs, noise=self.obs_noise)

        return obs if self.differentiable else self.detach(obs)

    def step(
        self,
        t,
        action: th.Tensor | np.ndarray,
        **kwargs,
    ) -> tuple[th.Tensor | np.ndarray, bool, bool, dict[str, Any]]:
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
        
        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(t >= self.max_ep_duration)
        t_delay_shifted = t - self.epoch_bounds["movement"][0]

        """
            Each stage of the trial is given here
            Trajectory only specifies the movement kinematics, the stable, delay, and hold periods simply repeat the first and last hand positions
        """
        if t < self.epoch_bounds["delay"][1]:
            self.hidden_goal = self.traj[:, 0, :]
        elif t >= self.epoch_bounds["movement"][0] and t < self.epoch_bounds["movement"][1]:
            self.hidden_goal = self.traj[:, t_delay_shifted, :].clone()
        elif t >= self.epoch_bounds["hold"][0]:
            self.hidden_goal = self.traj[:, -1, :].clone()

        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
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

    def reset(self, *, testing: bool = False, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """
        Uses the :meth:`Environment.reset()` method of the parent class :class:`Environment` that can be overwritten to 
        change the returned data. Here the goals (`i.e.`, the targets) are drawn from a random uniform distribution across
        the full joint space.
        """
        raise NotImplementedError





class DlyHalfReach(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *, testing: bool = False, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        #------------------------------------- START OPTION AND EFFECTOR DEFINITIONS 

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        speed_cond = options.get('speed_cond', None)
        delay_cond = options.get('delay_cond', None)
        custom_delay = options.get('custom_delay', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        #------------------------------------- END


        #------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

        self.stable_time = 25
        self.hold_time = 25

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        self.delay_time = self.choose_delay(delay_cond, custom_delay)

        # Set up different speeds, use same delay and movement time across batch to keep timesteps the same
        movement_times = list(np.arange(50, 150, 10)) if testing else [50, 100, 150]

        if speed_cond is None:
            self.movement_time = random.choice(movement_times)
        else:
            self.movement_time = movement_times[speed_cond]

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time+self.delay_time),
            "movement": (self.stable_time+self.delay_time, self.stable_time+self.delay_time+self.movement_time),
            "hold": (self.stable_time+self.delay_time+self.movement_time, self.stable_time+self.delay_time+self.movement_time+self.hold_time),
        }

        self.speed_scalar = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
            1 - (self.movement_time / 150) * th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["stable"][1], 1))
        ], dim=1)

        # Set duration
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

        #------------------------------------- END


        #------------------------------------- START STATIC NETWORK INPUT (NOT FEEDBACK)

        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["delay"][1], 1)),
            th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["movement"][0], 1))
        ], dim=1)

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )

        self.rule_input[:, 0] = 1

        #------------------------------------- END


        #------------------------------------- START KINEMATIC TRAJECTORY

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        # Generate 8 equally spaced angles
        angles = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in angles], dim=0)

        # this wont work yet cause everything else has shape batch_size (or I can assert reach_conds and batch_size are same shape)
        if reach_conds is None:
            point_idx = th.randint(0, points.size(0), (batch_size,))
        else:
            if isinstance(reach_conds, (int, float)):
                point_idx = torch.tensor([reach_conds])
            elif isinstance(reach_conds, (th.Tensor, np.ndarray)):
                point_idx = reach_conds

        goal = points[point_idx] * 0.25 + fingertip

        # Draw a line from fingertip to goal 
        x_points = fingertip[:, None, 0] + th.linspace(0, 1, steps=self.movement_time).repeat(batch_size, 1) * (goal[:, None, 0] - fingertip[:, None, 0]) 
        y_points = fingertip[:, None, 1] + th.linspace(0, 1, steps=self.movement_time).repeat(batch_size, 1) * (goal[:, None, 1] - fingertip[:, None, 1]) 

        self.traj = th.stack([x_points, y_points], dim=-1)

        # We want to start target onset after stable epoch
        self.vis_inp = th.cat([
            # [batch_size, stability timesteps, xy]
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], self.traj.shape[-1])),
            # [batch_size, delay->hold timesteps, xy]
            self.traj[:, -1:, :].repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        ], dim=1)

        self.hidden_goal = self.traj[:, 0, :].clone()

        #------------------------------------- END


        #------------------------------------- START MOTORNET OBSERVATIONS

        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(0, deterministic=deterministic)

        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        #------------------------------------- END

        return obs, info




class DlyHalfCircleClk(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *, testing: bool = False, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        #------------------------------------- START OPTION AND EFFECTOR DEFINITIONS 

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        speed_cond = options.get('speed_cond', None)
        delay_cond = options.get('delay_cond', None)
        custom_delay = options.get('custom_delay', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        #------------------------------------- END


        #------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

        self.stable_time = 25
        self.hold_time = 25

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        self.delay_time = self.choose_delay(delay_cond, custom_delay)

        # Set up different speeds, use same delay and movement time across batch to keep timesteps the same
        movement_times = list(np.arange(50, 150, 10)) if testing else [50, 100, 150]

        if speed_cond is None:
            self.movement_time = random.choice(movement_times)
        else:
            self.movement_time = movement_times[speed_cond]

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time+self.delay_time),
            "movement": (self.stable_time+self.delay_time, self.stable_time+self.delay_time+self.movement_time),
            "hold": (self.stable_time+self.delay_time+self.movement_time, self.stable_time+self.delay_time+self.movement_time+self.hold_time),
        }

        self.speed_scalar = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
            1 - (self.movement_time / 150) * th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["stable"][1], 1))
        ], dim=1)

        # Set duration
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

        #------------------------------------- END


        #------------------------------------- START STATIC NETWORK INPUT (except visual input)

        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["delay"][1], 1)),
            th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["movement"][0], 1))
        ], dim=1)

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )

        self.rule_input[:, 1] = 1

        #------------------------------------- END


        #------------------------------------- START KINEMATIC TRAJECTORY

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        traj_points = th.linspace(np.pi, 0, self.movement_time)

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in traj_points], dim=0)
        points = (points + th.tensor([[1, 0]])) * 0.25 * 0.5

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
        rotated_points = th.zeros(size=(batch_size, self.movement_time, 2))

        # Might be slow because I have to loop through everything
        if reach_conds is None:
            point_idx = th.randint(0, rot_angle.size(0), (batch_size,))
        else:
            if isinstance(reach_conds, (int, float)):
                point_idx = torch.tensor([reach_conds])
            elif isinstance(reach_conds, (th.Tensor, np.ndarray)):
                point_idx = reach_conds

        # Rotate the points based on the chosen angles
        batch_angles = rot_angle[point_idx]
        for i, theta in enumerate(batch_angles):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # Create the 2D rotation matrix
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        # Create full trajectory (center at fingertip)
        self.traj = rotated_points + fingertip[:, None, :]

        # We want to start target onset after stable epoch
        self.vis_inp = th.cat([
            # [batch_size, stability timesteps, xy]
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], self.traj.shape[-1])),
            # [batch_size, delay->hold timesteps, xy]
            self.traj[:, -1:, :].repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        ], dim=1)

        self.hidden_goal = self.traj[:, 0, :].clone()

        #------------------------------------- END


        #------------------------------------- START MOTORNET OBSERVATIONS

        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(0, deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        #------------------------------------- END

        return obs, info




class DlyHalfCircleCClk(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *, testing: bool = False, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        #------------------------------------- START OPTION AND EFFECTOR DEFINITIONS 

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        speed_cond = options.get('speed_cond', None)
        delay_cond = options.get('delay_cond', None)
        custom_delay = options.get('custom_delay', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        #------------------------------------- END


        #------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

        self.stable_time = 25
        self.hold_time = 25

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        self.delay_time = self.choose_delay(delay_cond, custom_delay)

        # Set up different speeds, use same delay and movement time across batch to keep timesteps the same
        movement_times = list(np.arange(50, 150, 10)) if testing else [50, 100, 150]

        if speed_cond is None:
            self.movement_time = random.choice(movement_times)
        else:
            self.movement_time = movement_times[speed_cond]

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time+self.delay_time),
            "movement": (self.stable_time+self.delay_time, self.stable_time+self.delay_time+self.movement_time),
            "hold": (self.stable_time+self.delay_time+self.movement_time, self.stable_time+self.delay_time+self.movement_time+self.hold_time),
        }

        self.speed_scalar = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
            1 - (self.movement_time / 150) * th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["stable"][1], 1))
        ], dim=1)

        # Set duration
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

        #------------------------------------- END


        #------------------------------------- START STATIC NETWORK INPUT (except visual input)

        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["delay"][1], 1)),
            th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["movement"][0], 1))
        ], dim=1)

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )

        self.rule_input[:, 2] = 1

        #------------------------------------- END


        #------------------------------------- START KINEMATIC TRAJECTORY

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        traj_points = th.linspace(np.pi, 2*np.pi, self.movement_time)

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in traj_points], dim=0)
        points = (points + th.tensor([[1, 0]])) * 0.25 * 0.5

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
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
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        self.traj = rotated_points + fingertip[:, None, :]

        # We want to start target onset after stable epoch
        self.vis_inp = th.cat([
            # [batch_size, stability timesteps, xy]
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], self.traj.shape[-1])),
            # [batch_size, delay->hold timesteps, xy]
            self.traj[:, -1:, :].repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        ], dim=1)

        self.hidden_goal = self.traj[:, 0, :].clone()

        #------------------------------------- END


        #------------------------------------- START MOTORNET OBSERVATIONS

        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(0, deterministic=deterministic)

        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        #------------------------------------- END

        return obs, info




class DlySinusoid(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *, testing: bool = False, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        #------------------------------------- START OPTION AND EFFECTOR DEFINITIONS 

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        speed_cond = options.get('speed_cond', None)
        delay_cond = options.get('delay_cond', None)
        custom_delay = options.get('custom_delay', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        #------------------------------------- END


        #------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

        self.stable_time = 25
        self.hold_time = 25

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        self.delay_time = self.choose_delay(delay_cond, custom_delay)

        # Set up different speeds, use same delay and movement time across batch to keep timesteps the same
        movement_times = list(np.arange(50, 150, 10)) if testing else [50, 100, 150]

        if speed_cond is None:
            self.movement_time = random.choice(movement_times)
        else:
            self.movement_time = movement_times[speed_cond]

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time+self.delay_time),
            "movement": (self.stable_time+self.delay_time, self.stable_time+self.delay_time+self.movement_time),
            "hold": (self.stable_time+self.delay_time+self.movement_time, self.stable_time+self.delay_time+self.movement_time+self.hold_time),
        }

        self.speed_scalar = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
            1 - (self.movement_time / 150) * th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["stable"][1], 1))
        ], dim=1)

        # Set duration
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

        #------------------------------------- END


        #------------------------------------- START STATIC NETWORK INPUT (except visual input)

        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["delay"][1], 1)),
            th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["movement"][0], 1))
        ], dim=1)

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )

        self.rule_input[:, 3] = 1

        #------------------------------------- END


        #------------------------------------- START KINEMATIC TRAJECTORY

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        # x and y coordinates for movement, x is in 0-1 range, y is similar 
        x_points = th.linspace(0, 1, self.movement_time)
        y_points = th.sin(th.linspace(0, 2*np.pi, self.movement_time))

        # Compute (x, y) coordinates for each angle
        # Circle y is scaled by 0.25 and 0.5 (this is so that the x coordinate has a length of 0.25, but this looks good)
        # Due to this, additionally scale only the y component of the sinusoid by 0.5 to get it in a better range
        points = th.stack([x_points, y_points], dim=1) * 0.25 * th.tensor([1, 0.5])

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
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
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        self.traj = rotated_points + fingertip[:, None, :]

        # We want to start target onset after stable epoch
        self.vis_inp = th.cat([
            # [batch_size, stability timesteps, xy]
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], self.traj.shape[-1])),
            # [batch_size, delay->hold timesteps, xy]
            self.traj[:, -1:, :].repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        ], dim=1)

        self.hidden_goal = self.traj[:, 0, :].clone()

        #------------------------------------- END


        #------------------------------------- START MOTORNET OBSERVATIONS

        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(0, deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        #------------------------------------- END

        return obs, info




class DlySinusoidInv(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *, testing: bool = False, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        #------------------------------------- START OPTION AND EFFECTOR DEFINITIONS 

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        speed_cond = options.get('speed_cond', None)
        delay_cond = options.get('delay_cond', None)
        custom_delay = options.get('custom_delay', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        #------------------------------------- END


        #------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

        self.stable_time = 25
        self.hold_time = 25

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        self.delay_time = self.choose_delay(delay_cond, custom_delay)

        # Set up different speeds, use same delay and movement time across batch to keep timesteps the same
        movement_times = list(np.arange(50, 150, 10)) if testing else [50, 100, 150]

        if speed_cond is None:
            self.movement_time = random.choice(movement_times)
        else:
            self.movement_time = movement_times[speed_cond]

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time+self.delay_time),
            "movement": (self.stable_time+self.delay_time, self.stable_time+self.delay_time+self.movement_time),
            "hold": (self.stable_time+self.delay_time+self.movement_time, self.stable_time+self.delay_time+self.movement_time+self.hold_time),
        }

        self.speed_scalar = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
            1 - (self.movement_time / 150) * th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["stable"][1], 1))
        ], dim=1)

        # Set duration
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1
        
        #------------------------------------- END


        #------------------------------------- START STATIC NETWORK INPUT (except visual input)

        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["delay"][1], 1)),
            th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["movement"][0], 1))
        ], dim=1)

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )

        self.rule_input[:, 4] = 1

        #------------------------------------- END


        #------------------------------------- START KINEMATIC TRAJECTORY

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        x_points = th.linspace(0, 1, self.movement_time)
        y_points = -th.sin(th.linspace(0, 2*np.pi, self.movement_time))

        # Compute (x, y) coordinates for each angle
        points = th.stack([x_points, y_points], dim=1) * 0.25 * th.tensor([1, 0.5])

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
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
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        self.traj = rotated_points + fingertip[:, None, :]

        # We want to start target onset after stable epoch
        self.vis_inp = th.cat([
            # [batch_size, stability timesteps, xy]
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], self.traj.shape[-1])),
            # [batch_size, delay->hold timesteps, xy]
            self.traj[:, -1:, :].repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        ], dim=1)

        self.hidden_goal = self.traj[:, 0, :].clone()

        #------------------------------------- END


        #------------------------------------- START MOTORNET OBSERVATIONS

        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(0, deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        #------------------------------------- END

        return obs, info




class DlyFullReach(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *, testing: bool = False, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        #------------------------------------- START OPTION AND EFFECTOR DEFINITIONS 

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        speed_cond = options.get('speed_cond', None)
        delay_cond = options.get('delay_cond', None)
        custom_delay = options.get('custom_delay', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        #------------------------------------- END


        #------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

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

        self.half_movement_time = int(self.movement_time/2)

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time+self.delay_time),
            "movement": (self.stable_time+self.delay_time, self.stable_time+self.delay_time+self.movement_time),
            "hold": (self.stable_time+self.delay_time+self.movement_time, self.stable_time+self.delay_time+self.movement_time+self.hold_time),
        }

        self.speed_scalar = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
            1 - (self.movement_time / 300) * th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["stable"][1], 1))
        ], dim=1)

        # Set duration
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

        #------------------------------------- END


        #------------------------------------- START STATIC NETWORK INPUT (except visual input)

        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["delay"][1], 1)),
            th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["movement"][0], 1))
        ], dim=1)

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )

        self.rule_input[:, 5] = 1

        #------------------------------------- END


        #------------------------------------- START KINEMATIC TRAJECTORY

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        # Generate 8 equally spaced angles
        angles = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in angles], dim=0)

        # this wont work yet cause everything else has shape batch_size (or I can assert reach_conds and batch_size are same shape)
        if reach_conds is None:
            point_idx = th.randint(0, points.size(0), (batch_size,))
        else:
            if isinstance(reach_conds, (int, float)):
                point_idx = torch.tensor([reach_conds])
            elif isinstance(reach_conds, (th.Tensor, np.ndarray)):
                point_idx = reach_conds

        goal = points[point_idx] * 0.25 + fingertip

        # Draw a line from fingertip to goal 
        x_points_ext = fingertip[:, None, 0] + th.linspace(0, 1, steps=self.half_movement_time).repeat(batch_size, 1) * (goal[:, None, 0] - fingertip[:, None, 0]) 
        y_points_ext = fingertip[:, None, 1] + th.linspace(0, 1, steps=self.half_movement_time).repeat(batch_size, 1) * (goal[:, None, 1] - fingertip[:, None, 1]) 

        # Draw a line from goal to fingertip
        x_points_ret = goal[:, None, 0] + th.linspace(0, 1, steps=self.half_movement_time).repeat(batch_size, 1) * (fingertip[:, None, 0] - goal[:, None, 0]) 
        y_points_ret = goal[:, None, 1] + th.linspace(0, 1, steps=self.half_movement_time).repeat(batch_size, 1) * (fingertip[:, None, 1] - goal[:, None, 1]) 

        # Concatenate reaching forward then backward along time axis
        forward_traj = th.stack([x_points_ext, y_points_ext], dim=-1)
        backward_traj = th.stack([x_points_ret, y_points_ret], dim=-1)
        self.traj = th.cat([forward_traj, backward_traj], dim=1)

        # We want to start target onset after stable epoch
        self.vis_inp = th.cat([
            # [batch_size, stability timesteps, xy]
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], self.traj.shape[-1])),
            # [batch_size, delay->hold timesteps, xy]
            self.traj[:, self.half_movement_time, :].unsqueeze(1).repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        ], dim=1)

        self.hidden_goal = self.traj[:, 0, :].clone()

        #------------------------------------- END


        #------------------------------------- START MOTORNET OBSERVATIONS

        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(0, deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        #------------------------------------- END

        return obs, info




class DlyFullCircleClk(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *, testing: bool = False, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        #------------------------------------- START OPTION AND EFFECTOR DEFINITIONS 

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        speed_cond = options.get('speed_cond', None)
        delay_cond = options.get('delay_cond', None)
        custom_delay = options.get('custom_delay', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        #------------------------------------- END


        #------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

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

        self.half_movement_time = int(self.movement_time/2)

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time+self.delay_time),
            "movement": (self.stable_time+self.delay_time, self.stable_time+self.delay_time+self.movement_time),
            "hold": (self.stable_time+self.delay_time+self.movement_time, self.stable_time+self.delay_time+self.movement_time+self.hold_time),
        }

        self.speed_scalar = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
            1 - (self.movement_time / 300) * th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["stable"][1], 1))
        ], dim=1)

        # Set duration
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

        #------------------------------------- END


        #------------------------------------- START STATIC NETWORK INPUT (except visual input)

        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["delay"][1], 1)),
            th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["movement"][0], 1))
        ], dim=1)

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )

        self.rule_input[:, 6] = 1

        #------------------------------------- END


        #------------------------------------- START KINEMATIC TRAJECTORY

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        traj_points = th.linspace(np.pi, -np.pi, self.movement_time)

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in traj_points], dim=0)
        points = (points + th.tensor([[1, 0]])) * 0.25 * 0.5

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
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
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        self.traj = rotated_points + fingertip[:, None, :]

        # We want to start target onset after stable epoch
        self.vis_inp = th.cat([
            # [batch_size, stability timesteps, xy]
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], self.traj.shape[-1])),
            # [batch_size, delay->hold timesteps, xy]
            self.traj[:, self.half_movement_time, :].unsqueeze(1).repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        ], dim=1)

        self.hidden_goal = self.traj[:, 0, :].clone()

        #------------------------------------- END


        #------------------------------------- START MOTORNET OBSERVATIONS

        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(0, deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        #------------------------------------- END

        return obs, info




class DlyFullCircleCClk(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *, testing: bool = False, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        #------------------------------------- START OPTION AND EFFECTOR DEFINITIONS 

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        speed_cond = options.get('speed_cond', None)
        delay_cond = options.get('delay_cond', None)
        custom_delay = options.get('custom_delay', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        #------------------------------------- END


        #------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

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

        self.half_movement_time = int(self.movement_time/2)

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time+self.delay_time),
            "movement": (self.stable_time+self.delay_time, self.stable_time+self.delay_time+self.movement_time),
            "hold": (self.stable_time+self.delay_time+self.movement_time, self.stable_time+self.delay_time+self.movement_time+self.hold_time),
        }

        self.speed_scalar = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
            1 - (self.movement_time / 300) * th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["stable"][1], 1))
        ], dim=1)

        # Set duration
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

        #------------------------------------- END


        #------------------------------------- START STATIC NETWORK INPUT (except visual input)

        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["delay"][1], 1)),
            th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["movement"][0], 1))
        ], dim=1)

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )

        self.rule_input[:, 7] = 1

        #------------------------------------- END


        #------------------------------------- START KINEMATIC TRAJECTORY

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        traj_points = th.linspace(np.pi, 3*np.pi, self.movement_time)

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in traj_points], dim=0)
        points = (points + th.tensor([[1, 0]])) * 0.25 * 0.5

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
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
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        self.traj = rotated_points + fingertip[:, None, :]

        # We want to start target onset after stable epoch
        self.vis_inp = th.cat([
            # [batch_size, stability timesteps, xy]
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], self.traj.shape[-1])),
            # [batch_size, delay->hold timesteps, xy]
            self.traj[:, self.half_movement_time, :].unsqueeze(1).repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        ], dim=1)

        self.hidden_goal = self.traj[:, 0, :].clone()

        #------------------------------------- END


        #------------------------------------- START MOTORNET OBSERVATIONS

        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(0, deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        #------------------------------------- END

        return obs, info




class DlyFigure8(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *, testing: bool = False, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        #------------------------------------- START OPTION AND EFFECTOR DEFINITIONS 

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        speed_cond = options.get('speed_cond', None)
        delay_cond = options.get('delay_cond', None)
        custom_delay = options.get('custom_delay', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        #------------------------------------- END


        #------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

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

        self.half_movement_time = int(self.movement_time/2)

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time+self.delay_time),
            "movement": (self.stable_time+self.delay_time, self.stable_time+self.delay_time+self.movement_time),
            "hold": (self.stable_time+self.delay_time+self.movement_time, self.stable_time+self.delay_time+self.movement_time+self.hold_time),
        }

        self.speed_scalar = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
            1 - (self.movement_time / 300) * th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["stable"][1], 1))
        ], dim=1)

        # Set duration
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

        #------------------------------------- END


        #------------------------------------- START STATIC NETWORK INPUT (except visual input)

        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["delay"][1], 1)),
            th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["movement"][0], 1))
        ], dim=1)

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )

        self.rule_input[:, 8] = 1

        #------------------------------------- END


        #------------------------------------- START KINEMATIC TRAJECTORY

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        x_points_forward = th.linspace(0, 1, self.half_movement_time)
        y_points_forward = th.sin(th.linspace(0, 2*np.pi, self.half_movement_time))

        x_points_back = th.linspace(1, 0, self.half_movement_time)
        y_points_back = -th.sin(th.linspace(2*np.pi, 0, self.half_movement_time))

        # Compute (x, y) coordinates for each angle
        points_forward = th.stack([x_points_forward, y_points_forward], dim=1) * 0.25 * th.tensor([1, 0.5])
        points_back = th.stack([x_points_back, y_points_back], dim=1) * 0.25 * th.tensor([1, 0.5])

        points = th.cat([points_forward, points_back], dim=0)

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
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
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        self.traj = rotated_points + fingertip[:, None, :]

        # We want to start target onset after stable epoch
        self.vis_inp = th.cat([
            # [batch_size, stability timesteps, xy]
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], self.traj.shape[-1])),
            # [batch_size, delay->hold timesteps, xy]
            self.traj[:, self.half_movement_time, :].unsqueeze(1).repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        ], dim=1)

        self.hidden_goal = self.traj[:, 0, :].clone()

        #------------------------------------- END


        #------------------------------------- START MOTORNET OBSERVATIONS

        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(0, deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        #------------------------------------- END

        return obs, info




class DlyFigure8Inv(MotornetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *, testing: bool = False, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        #------------------------------------- START OPTION AND EFFECTOR DEFINITIONS 

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        speed_cond = options.get('speed_cond', None)
        delay_cond = options.get('delay_cond', None)
        custom_delay = options.get('custom_delay', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        #------------------------------------- END


        #------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

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

        self.half_movement_time = int(self.movement_time/2)

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time+self.delay_time),
            "movement": (self.stable_time+self.delay_time, self.stable_time+self.delay_time+self.movement_time),
            "hold": (self.stable_time+self.delay_time+self.movement_time, self.stable_time+self.delay_time+self.movement_time+self.hold_time),
        }

        self.speed_scalar = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
            1 - (self.movement_time / 300) * th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["stable"][1], 1))
        ], dim=1)

        # Set duration
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

        #------------------------------------- END


        #------------------------------------- START STATIC NETWORK INPUT (except visual input)

        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["delay"][1], 1)),
            th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["movement"][0], 1))
        ], dim=1)

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )

        self.rule_input[:, 9] = 1

        #------------------------------------- END


        #------------------------------------- START KINEMATIC TRAJECTORY

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        x_points_forward = th.linspace(0, 1, self.half_movement_time)
        y_points_forward = -th.sin(th.linspace(0, 2*np.pi, self.half_movement_time))

        x_points_back = th.linspace(1, 0, self.half_movement_time)
        y_points_back = th.sin(th.linspace(2*np.pi, 0, self.half_movement_time))

        # Compute (x, y) coordinates for each angle
        points_forward = th.stack([x_points_forward, y_points_forward], dim=1) * 0.25 * th.tensor([1, 0.5])
        points_back = th.stack([x_points_back, y_points_back], dim=1) * 0.25 * th.tensor([1, 0.5])

        points = th.cat([points_forward, points_back], dim=0)

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
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
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        self.traj = rotated_points + fingertip[:, None, :]

        # We want to start target onset after stable epoch
        self.vis_inp = th.cat([
            # [batch_size, stability timesteps, xy]
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], self.traj.shape[-1])),
            # [batch_size, delay->hold timesteps, xy]
            self.traj[:, self.half_movement_time, :].unsqueeze(1).repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        ], dim=1)

        self.hidden_goal = self.traj[:, 0, :].clone()

        #------------------------------------- END


        #------------------------------------- START MOTORNET OBSERVATIONS

        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(0, deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        #------------------------------------- END

        return obs, info





class ComposableEnv(env.Environment):
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
        super().__init__(*args, **kwargs)
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None

    def get_obs(self, t, action=None, deterministic: bool = False) -> th.Tensor | np.ndarray:
        """
        Returns a `(batch_size, n_features)` `tensor` containing the (potientially time-delayed) observations.
        By default, this is the task goal, followed by the output of the :meth:`get_proprioception()` method, 
        the output of the :meth:`get_vision()` method, and finally the last :attr:`action_frame_stacking` action sets,
        if a non-zero `action_frame_stacking` keyword argument was passed at initialization of this class instance.
        `.i.i.d.` Gaussian noise is added to each element in the `tensor`,
        using the :attr:`obs_noise` attribute.
        """
        self.update_obs_buffer(action=action)

        obs_as_list = [
        self.rule_input,
        self.speed_scalar[:, t],
        self.go_cue[:, t],
        self.vis_inp[:, t],
        self.obs_buffer["vision"][0],
        self.obs_buffer["proprioception"][0],
        ] + self.obs_buffer["action"][:self.action_frame_stacking]
        
        obs = th.cat(obs_as_list, dim=-1)

        if deterministic is False:
            obs = self.apply_noise(obs, noise=self.obs_noise)

        return obs if self.differentiable else self.detach(obs)
    
    def halfreach_forward_motif(self, joint_state, batch_size, testing, reach_conds=None):
        
        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        # Generate 8 equally spaced angles
        angles = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in angles], dim=0)

        # this wont work yet cause everything else has shape batch_size (or I can assert reach_conds and batch_size are same shape)
        if reach_conds is None:
            point_idx = th.randint(0, points.size(0), (batch_size,))
        else:
            # tensor that will specify which of the 8 conditions to get
            point_idx = reach_conds

        goal = points[point_idx] * 0.25 + fingertip

        # Draw a line from fingertip to goal 
        x_points = fingertip[:, None, 0] + th.linspace(0, 1, steps=self.half_movement_time).repeat(batch_size, 1) * (goal[:, None, 0] - fingertip[:, None, 0]) 
        y_points = fingertip[:, None, 1] + th.linspace(0, 1, steps=self.half_movement_time).repeat(batch_size, 1) * (goal[:, None, 1] - fingertip[:, None, 1]) 

        traj = th.stack([x_points, y_points], dim=-1)

        return traj

    def halfcircleclk_forward_motif(self, joint_state, batch_size, testing, reach_conds=False):

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        traj_points = th.linspace(np.pi, 0, self.half_movement_time)

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in traj_points], dim=0)
        points = (points + th.tensor([[1, 0]])) * 0.25 * 0.5

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
        rotated_points = th.zeros(size=(batch_size, self.half_movement_time, 2))

        # Might be slow because I have to loop through everything
        if reach_conds is None:
            point_idx = th.randint(0, rot_angle.size(0), (batch_size,))
        else:
            # tensor that will specify which of the 8 conditions to get
            point_idx = th.tensor([reach_conds])

        # Rotate the points based on the chosen angles
        batch_angles = rot_angle[point_idx]
        for i, theta in enumerate(batch_angles):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # Create the 2D rotation matrix
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        # Create full trajectory (center at fingertip)
        traj = rotated_points + fingertip[:, None, :]

        return traj

    def halfcirclecclk_forward_motif(self, joint_state, batch_size, testing, reach_conds=False):

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        traj_points = th.linspace(np.pi, 2*np.pi, self.half_movement_time)

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in traj_points], dim=0)
        points = (points + th.tensor([[1, 0]])) * 0.25 * 0.5

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
        rotated_points = th.zeros(size=(batch_size, self.half_movement_time, 2))

        # Might be slow because I have to loop through everything
        if reach_conds is None:
            point_idx = th.randint(0, rot_angle.size(0), (batch_size,))
        else:
            # tensor that will specify which of the 8 conditions to get
            point_idx = reach_conds

        batch_angles = rot_angle[point_idx]
        for i, theta in enumerate(batch_angles):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # Create the 2D rotation matrix
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        traj = rotated_points + fingertip[:, None, :]

        return traj


    def sinusoid_forward_motif(self, joint_state, batch_size, testing, reach_conds=False):

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        # x and y coordinates for movement, x is in 0-1 range, y is similar 
        x_points = th.linspace(0, 1, self.half_movement_time)
        y_points = th.sin(th.linspace(0, 2*np.pi, self.half_movement_time))

        # Compute (x, y) coordinates for each angle
        # Circle y is scaled by 0.25 and 0.5 (this is so that the x coordinate has a length of 0.25, but this looks good)
        # Due to this, additionally scale only the y component of the sinusoid by 0.5 to get it in a better range
        points = th.stack([x_points, y_points], dim=1) * 0.25 * th.tensor([1, 0.5])

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
        rotated_points = th.zeros(size=(batch_size, self.half_movement_time, 2))

        # Might be slow because I have to loop through everything
        if reach_conds is None:
            point_idx = th.randint(0, rot_angle.size(0), (batch_size,))
        else:
            # tensor that will specify which of the 8 conditions to get
            point_idx = reach_conds

        batch_angles = rot_angle[point_idx]
        for i, theta in enumerate(batch_angles):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # Create the 2D rotation matrix
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        traj = rotated_points + fingertip[:, None, :]

        return traj

    def sinusoidinv_forward_motif(self, joint_state, batch_size, testing, reach_conds=False):

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        x_points = th.linspace(0, 1, self.half_movement_time)
        y_points = -th.sin(th.linspace(0, 2*np.pi, self.half_movement_time))

        # Compute (x, y) coordinates for each angle
        points = th.stack([x_points, y_points], dim=1) * 0.25 * th.tensor([1, 0.5])

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
        rotated_points = th.zeros(size=(batch_size, self.half_movement_time, 2))

        # Might be slow because I have to loop through everything
        if reach_conds is None:
            point_idx = th.randint(0, rot_angle.size(0), (batch_size,))
        else:
            # tensor that will specify which of the 8 conditions to get
            point_idx = reach_conds

        batch_angles = rot_angle[point_idx]
        for i, theta in enumerate(batch_angles):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # Create the 2D rotation matrix
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        traj = rotated_points + fingertip[:, None, :]

        return traj

    def fullreach_backward_motif(self, joint_state, batch_size, testing, reach_conds=False):

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        # Generate 8 equally spaced angles
        angles = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in angles], dim=0)

        # this wont work yet cause everything else has shape batch_size (or I can assert reach_conds and batch_size are same shape)
        if reach_conds is None:
            point_idx = th.randint(0, points.size(0), (batch_size,))
        else:
            # tensor that will specify which of the 8 conditions to get
            point_idx = reach_conds

        goal = points[point_idx] * 0.25 + fingertip

        # Draw a line from goal to fingertip
        x_points_ret = goal[:, None, 0] + th.linspace(0, 1, steps=self.half_movement_time).repeat(batch_size, 1) * (fingertip[:, None, 0] - goal[:, None, 0]) 
        y_points_ret = goal[:, None, 1] + th.linspace(0, 1, steps=self.half_movement_time).repeat(batch_size, 1) * (fingertip[:, None, 1] - goal[:, None, 1]) 

        # Concatenate reaching forward then backward along time axis
        traj = th.stack([x_points_ret, y_points_ret], dim=-1)

        return traj

    def fullcircleclk_backward_motif(self, joint_state, batch_size, testing, reach_conds=False):

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        traj_points = th.linspace(0, -np.pi, self.half_movement_time)

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in traj_points], dim=0)
        points = (points + th.tensor([[1, 0]])) * 0.25 * 0.5

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
        rotated_points = th.zeros(size=(batch_size, self.half_movement_time, 2))

        # Might be slow because I have to loop through everything
        if reach_conds is None:
            point_idx = th.randint(0, rot_angle.size(0), (batch_size,))
        else:
            # tensor that will specify which of the 8 conditions to get
            point_idx = reach_conds

        batch_angles = rot_angle[point_idx]
        for i, theta in enumerate(batch_angles):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # Create the 2D rotation matrix
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        traj = rotated_points + fingertip[:, None, :]

        return traj

    def fullcirclecclk_backward_motif(self, joint_state, batch_size, testing, reach_conds=False):

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        traj_points = th.linspace(2*np.pi, 3*np.pi, self.half_movement_time)

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in traj_points], dim=0)
        points = (points + th.tensor([[1, 0]])) * 0.25 * 0.5

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
        rotated_points = th.zeros(size=(batch_size, self.half_movement_time, 2))

        # Might be slow because I have to loop through everything
        if reach_conds is None:
            point_idx = th.randint(0, rot_angle.size(0), (batch_size,))
        else:
            # tensor that will specify which of the 8 conditions to get
            point_idx = reach_conds

        batch_angles = rot_angle[point_idx]
        for i, theta in enumerate(batch_angles):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # Create the 2D rotation matrix
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        traj = rotated_points + fingertip[:, None, :]

        return traj

    def figure8_backward_motif(self, joint_state, batch_size, testing, reach_conds=False):

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        x_points_back = th.linspace(1, 0, self.half_movement_time)
        y_points_back = -th.sin(th.linspace(2*np.pi, 0, self.half_movement_time))

        # Compute (x, y) coordinates for each angle
        points = th.stack([x_points_back, y_points_back], dim=1) * 0.25 * th.tensor([1, 0.5])

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
        rotated_points = th.zeros(size=(batch_size, self.half_movement_time, 2))

        # Might be slow because I have to loop through everything
        if reach_conds is None:
            point_idx = th.randint(0, rot_angle.size(0), (batch_size,))
        else:
            # tensor that will specify which of the 8 conditions to get
            point_idx = reach_conds

        batch_angles = rot_angle[point_idx]
        for i, theta in enumerate(batch_angles):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # Create the 2D rotation matrix
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        traj = rotated_points + fingertip[:, None, :]

        return traj

    def figure8inv_backward_motif(self, joint_state, batch_size, testing, reach_conds=False):

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        x_points_back = th.linspace(1, 0, self.half_movement_time)
        y_points_back = th.sin(th.linspace(2*np.pi, 0, self.half_movement_time))

        # Compute (x, y) coordinates for each angle
        points = th.stack([x_points_back, y_points_back], dim=1) * 0.25 * th.tensor([1, 0.5])

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1] if testing else th.linspace(0, 2 * np.pi, 9)[:-1]
        rotated_points = th.zeros(size=(batch_size, self.half_movement_time, 2))

        # Might be slow because I have to loop through everything
        if reach_conds is None:
            point_idx = th.randint(0, rot_angle.size(0), (batch_size,))
        else:
            # tensor that will specify which of the 8 conditions to get
            point_idx = reach_conds

        batch_angles = rot_angle[point_idx]
        for i, theta in enumerate(batch_angles):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # Create the 2D rotation matrix
            R = th.tensor([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
            rotated_traj = (R @ points.T).T
            rotated_points[i] = rotated_traj
        
        traj = rotated_points + fingertip[:, None, :]

        return traj


    def step(
        self,
        t,
        action: th.Tensor | np.ndarray,
        **kwargs,
    ) -> tuple[th.Tensor | np.ndarray, bool, bool, dict[str, Any]]:
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
        
        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(t >= self.max_ep_duration)
        t_delay_shifted = t - self.epoch_bounds["movement"][0]

        """
            Each stage of the trial is given here
            Trajectory only specifies the movement kinematics, the stable, delay, and hold periods simply repeat the first and last hand positions
        """
        if t < self.epoch_bounds["delay"][1]:
            self.hidden_goal = self.traj[:, 0, :]
        elif t >= self.epoch_bounds["movement"][0] and t < self.epoch_bounds["movement"][1]:
            self.hidden_goal = self.traj[:, t_delay_shifted, :].clone()
        elif t >= self.epoch_bounds["hold"][0]:
            self.hidden_goal = self.traj[:, -1, :].clone()

        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        return obs, reward, terminated, info

    def reset(self, *, testing: bool = False, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """
        Uses the :meth:`Environment.reset()` method of the parent class :class:`Environment` that can be overwritten to 
        change the returned data. Here the goals (`i.e.`, the targets) are drawn from a random uniform distribution across
        the full joint space.
        """

        #------------------------------------- START OPTION AND EFFECTOR DEFINITIONS 

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        speed_cond = options.get('speed_cond', None)
        delay_cond = options.get('delay_cond', None)
        forward_key = options.get('forward_key', None)
        backward_key = options.get('backward_key', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        self.forward_motifs = {
            "forward_halfreach": self.halfreach_forward_motif,
            "forward_halfcircleclk": self.halfcircleclk_forward_motif,
            "forward_halfcirclecclk": self.halfcirclecclk_forward_motif,
            "forward_sinusoid": self.sinusoid_forward_motif,
            "forward_sinusoidinv": self.sinusoidinv_forward_motif
        }

        self.backward_motifs = {
            "backward_fullreach": self.fullreach_backward_motif,
            "backward_fullcircleclk": self.fullcircleclk_backward_motif,
            "backward_fullcirclecclk": self.fullcirclecclk_backward_motif,
            "backward_figure8": self.figure8_backward_motif,
            "backward_figure8inv": self.figure8inv_backward_motif
        }

        self.combination_idx = list(product(self.forward_motifs, self.backward_motifs))
        self.combination_idx.remove(("forward_halfreach", "backward_fullreach"))
        self.combination_idx.remove(("forward_halfcircleclk", "backward_fullcircleclk"))
        self.combination_idx.remove(("forward_halfcirclecclk", "backward_fullcirclecclk"))
        self.combination_idx.remove(("forward_sinusoid", "backward_figure8"))
        self.combination_idx.remove(("forward_sinusoidinv", "backward_figure8inv"))

        #------------------------------------- END


        #------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

        self.stable_time = 25
        self.hold_time = 25

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        delay_times = [25, 50, 75]
        if delay_cond is None:
            self.delay_time = random.choice(delay_times)
        else:
            self.delay_time = delay_times[delay_cond]

        # Set up different speeds, use same delay and movement time across batch to keep timesteps the same
        movement_times = list(np.arange(100, 300, 20)) if testing else [100, 200, 300]

        if speed_cond is None:
            self.movement_time = random.choice(movement_times)
        else:
            self.movement_time = movement_times[speed_cond]

        self.half_movement_time = int(self.movement_time/2)

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time+self.delay_time),
            "movement": (self.stable_time+self.delay_time, self.stable_time+self.delay_time+self.movement_time),
            "hold": (self.stable_time+self.delay_time+self.movement_time, self.stable_time+self.delay_time+self.movement_time+self.hold_time),
        }

        self.speed_scalar = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], 1)),
            1 - (self.movement_time / 300) * th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["stable"][1], 1))
        ], dim=1)

        # Set duration
        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

        #------------------------------------- END


        #------------------------------------- START STATIC NETWORK INPUT (NOT FEEDBACK)

        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["delay"][1], 1)),
            th.ones(size=(batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["movement"][0], 1))
        ], dim=1)

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 20)
        )


        #------------------------------------- END


        #------------------------------------- START KINEMATIC TRAJECTORY

        # choose which forward and backward motif to use
        if forward_key is None:

            forward_key_list = list(self.forward_motifs.keys()).copy()
            forward_key = random.choice(forward_key_list)
            forward_key_idx = forward_key_list.index(forward_key)

            forward_motif = self.forward_motifs[forward_key]
            forward_traj = forward_motif(joint_state, batch_size, testing, reach_conds=reach_conds)

        else:
            forward_traj = self.forward_motifs[forward_key](joint_state, batch_size, testing, reach_conds=reach_conds)

        if backward_key is None:

            # Ensure that the backward motif is not same as forward
            backward_key_list = list(self.backward_motifs.keys()).copy()
            backward_key_list.pop(forward_key_idx)

            backward_key = random.choice(backward_key_list)
            backward_motif = self.backward_motifs[backward_key]
            backward_traj = backward_motif(joint_state, batch_size, testing, reach_conds=reach_conds)

        else:
            backward_traj = self.backward_motifs[backward_key](joint_state, batch_size, testing, reach_conds=reach_conds)
        
        self.traj = th.cat([forward_traj, backward_traj], dim=1)

        # We want to start target onset after stable epoch
        self.vis_inp = th.cat([
            # [batch_size, stability timesteps, xy]
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], self.traj.shape[-1])),
            # [batch_size, delay->hold timesteps, xy]
            self.traj[:, self.half_movement_time, :].unsqueeze(1).repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        ], dim=1)

        rule_idx = self.combination_idx.index((forward_key, backward_key))
        self.rule_input[:, rule_idx] = 1
        self.hidden_goal = self.traj[:, 0, :].clone()

        #------------------------------------- END


        #------------------------------------- START MOTORNET OBSERVATIONS

        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(0, deterministic=deterministic)

        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        #------------------------------------- END

        return obs, info