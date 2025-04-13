import torch as th
import numpy as np
from typing import Any
from motornet import environment as env
import random

# Comments in first environment (below) should follow for others

class DlyHalfReach(env.Environment):
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
        self.go_cue[:, t].unsqueeze(1),
        self.vis_inp,
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
        self.hidden_goal = self.initial_pos.clone() if t < self.delay_time else self.traj[:, t - self.delay_time, :].clone()
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        return obs, reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """
        Uses the :meth:`Environment.reset()` method of the parent class :class:`Environment` that can be overwritten to 
        change the returned data. Here the goals (`i.e.`, the targets) are drawn from a random uniform distribution across
        the full joint space.
        """
        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        delay_times = [50, 75, 100]
        self.delay_time = random.choice(delay_times)
        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.delay_time)),
            th.ones(size=(batch_size, 100))
        ], dim=-1)
        # Set duration
        self.max_ep_duration = self.go_cue.shape[-1] - 1

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        # Generate 8 equally spaced angles
        angles = th.linspace(0, 2 * np.pi, 33)[:-1]
        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in angles], dim=0)
        # this wont work yet cause everything else has shape batch_size (or I can assert reach_conds and batch_size are same shape)
        if reach_conds is None:
            point_idx = th.randint(0, points.size(0), (batch_size,))
        else:
            # tensor that will specify which of the 8 conditions to get
            point_idx = reach_conds

        goal = points[point_idx] * 0.25 + fingertip
        self.vis_inp = goal.clone()

        # Draw a line from fingertip to goal 
        x_points = fingertip[:, None, 0] + th.linspace(0, 1, steps=100).repeat(batch_size, 1) * (goal[:, None, 0] - fingertip[:, None, 0]) 
        y_points = fingertip[:, None, 1] + th.linspace(0, 1, steps=100).repeat(batch_size, 1) * (goal[:, None, 1] - fingertip[:, None, 1]) 

        self.traj = th.stack([x_points, y_points], dim=-1)
        self.initial_pos = self.states["fingertip"].clone()
        self.hidden_goal = self.initial_pos.clone()

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )
        self.rule_input[:, 0] = 1

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
        return obs, info




class DlyHalfCircleClk(env.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None

    def get_obs(self, t, action=None, deterministic: bool = False) -> th.Tensor | np.ndarray:
        self.update_obs_buffer(action=action)

        obs_as_list = [
        self.rule_input,
        self.go_cue[:, t].unsqueeze(1),
        self.vis_inp,
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

        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(t >= self.max_ep_duration)
        self.hidden_goal = self.initial_pos.clone() if t < self.delay_time else self.traj[:, t - self.delay_time, :].clone()
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        return obs, reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        delay_times = [50, 75, 100]
        self.delay_time = random.choice(delay_times)
        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.delay_time)),
            th.ones(size=(batch_size, int(1 / self.dt)))
        ], dim=-1)
        # Set duration
        self.max_ep_duration = self.go_cue.shape[-1] - 1

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        traj_points = th.linspace(np.pi, 0, 100)
        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in traj_points], dim=0)
        points = (points + th.tensor([[1, 0]])) * 0.25 * 0.5

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1]
        rotated_points = th.zeros(size=(batch_size, int(1 / self.dt), 2))
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
        
        self.traj = rotated_points + fingertip[:, None, :]
        self.vis_inp = self.traj[:, -1, :].clone()
        self.initial_pos = self.states["fingertip"].clone()
        self.hidden_goal = self.initial_pos.clone()

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )
        self.rule_input[:, 1] = 1

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
        return obs, info




class DlyHalfCircleCClk(env.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None

    def get_obs(self, t, action=None, deterministic: bool = False) -> th.Tensor | np.ndarray:

        self.update_obs_buffer(action=action)

        obs_as_list = [
        self.rule_input,
        self.go_cue[:, t].unsqueeze(1),
        self.vis_inp,
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
        
        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(t >= self.max_ep_duration)
        self.hidden_goal = self.initial_pos.clone() if t < self.delay_time else self.traj[:, t - self.delay_time, :].clone()
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        return obs, reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        delay_times = [50, 75, 100]
        self.delay_time = random.choice(delay_times)
        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.delay_time)),
            th.ones(size=(batch_size, int(1 / self.dt)))
        ], dim=-1)
        # Set duration
        self.max_ep_duration = self.go_cue.shape[-1] - 1

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        traj_points = th.linspace(np.pi, 2*np.pi, 100)
        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in traj_points], dim=0)
        points = (points + th.tensor([[1, 0]])) * 0.25 * 0.5

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1]
        rotated_points = th.zeros(size=(batch_size, int(1 / self.dt), 2))
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
        
        self.traj = rotated_points + fingertip[:, None, :]
        self.vis_inp = self.traj[:, -1, :].clone()
        self.initial_pos = self.states["fingertip"].clone()
        self.hidden_goal = self.initial_pos.clone()

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )
        self.rule_input[:, 2] = 1

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
        return obs, info




class DlySinusoid(env.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None

    def get_obs(self, t, action=None, deterministic: bool = False) -> th.Tensor | np.ndarray:

        self.update_obs_buffer(action=action)

        obs_as_list = [
        self.rule_input,
        self.go_cue[:, t].unsqueeze(1),
        self.vis_inp,
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

        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(t >= self.max_ep_duration)
        self.hidden_goal = self.initial_pos.clone() if t < self.delay_time else self.traj[:, t - self.delay_time, :].clone()
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        return obs, reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        delay_times = [50, 75, 100]
        self.delay_time = random.choice(delay_times)
        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.delay_time)),
            th.ones(size=(batch_size, int(1 / self.dt)))
        ], dim=-1)
        # Set duration
        self.max_ep_duration = self.go_cue.shape[-1] - 1

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        x_points = th.linspace(0, 1, 100)
        y_points = th.sin(th.linspace(0, 2*np.pi, 100))
        # Compute (x, y) coordinates for each angle
        # Circle y is scaled by 0.25 and 0.5 (this is so that the x coordinate has a length of 0.25, but this looks good)
        # Due to this, additionally scale only the y component of the sinusoid by 0.5 to get it in a better range
        points = th.stack([x_points, y_points], dim=1) * 0.25 * th.tensor([1, 0.5])

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1]
        rotated_points = th.zeros(size=(batch_size, int(1 / self.dt), 2))
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
        
        self.traj = rotated_points + fingertip[:, None, :]
        self.vis_inp = self.traj[:, -1, :].clone()
        self.initial_pos = self.states["fingertip"].clone()
        self.hidden_goal = self.initial_pos.clone()

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )
        self.rule_input[:, 3] = 1

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
        return obs, info




class DlySinusoidInv(env.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None

    def get_obs(self, t, action=None, deterministic: bool = False) -> th.Tensor | np.ndarray:

        self.update_obs_buffer(action=action)

        obs_as_list = [
        self.rule_input,
        self.go_cue[:, t].unsqueeze(1),
        self.vis_inp,
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
        
        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(t >= self.max_ep_duration)
        self.hidden_goal = self.initial_pos.clone() if t < self.delay_time else self.traj[:, t - self.delay_time, :].clone()
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        return obs, reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        delay_times = [50, 75, 100]
        self.delay_time = random.choice(delay_times)
        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.delay_time)),
            th.ones(size=(batch_size, int(1 / self.dt)))
        ], dim=-1)
        # Set duration
        self.max_ep_duration = self.go_cue.shape[-1] - 1

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        x_points = th.linspace(0, 1, 100)
        y_points = -th.sin(th.linspace(0, 2*np.pi, 100))
        # Compute (x, y) coordinates for each angle
        points = th.stack([x_points, y_points], dim=1) * 0.25 * th.tensor([1, 0.5])

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1]
        rotated_points = th.zeros(size=(batch_size, int(1 / self.dt), 2))
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
        
        self.traj = rotated_points + fingertip[:, None, :]
        self.vis_inp = self.traj[:, -1, :].clone()
        self.initial_pos = self.states["fingertip"].clone()
        self.hidden_goal = self.initial_pos.clone()

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )
        self.rule_input[:, 4] = 1

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
        return obs, info




class DlyFullReach(env.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None

    def get_obs(self, t, action=None, deterministic: bool = False) -> th.Tensor | np.ndarray:

        self.update_obs_buffer(action=action)

        obs_as_list = [
        self.rule_input,
        self.go_cue[:, t].unsqueeze(1),
        self.vis_inp,
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

        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(t >= self.max_ep_duration)
        self.hidden_goal = self.initial_pos.clone() if t < self.delay_time else self.traj[:, t - self.delay_time, :].clone()
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        return obs, reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        delay_times = [50, 75, 100]
        self.delay_time = random.choice(delay_times)
        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.delay_time)),
            th.ones(size=(batch_size, int(2 / self.dt)))
        ], dim=-1)
        # Set duration
        self.max_ep_duration = self.go_cue.shape[-1] - 1

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        # Generate 8 equally spaced angles
        angles = th.linspace(0, 2 * np.pi, 33)[:-1]
        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in angles], dim=0)
        # this wont work yet cause everything else has shape batch_size (or I can assert reach_conds and batch_size are same shape)
        if reach_conds is None:
            point_idx = th.randint(0, points.size(0), (batch_size,))
        else:
            # tensor that will specify which of the 8 conditions to get
            point_idx = reach_conds

        goal = points[point_idx] * 0.25 + fingertip
        self.vis_inp = goal.clone()

        # Draw a line from fingertip to goal 
        x_points_ext = fingertip[:, None, 0] + th.linspace(0, 1, steps=100).repeat(batch_size, 1) * (goal[:, None, 0] - fingertip[:, None, 0]) 
        y_points_ext = fingertip[:, None, 1] + th.linspace(0, 1, steps=100).repeat(batch_size, 1) * (goal[:, None, 1] - fingertip[:, None, 1]) 

        # Draw a line from goal to fingertip
        x_points_ret = goal[:, None, 0] + th.linspace(0, 1, steps=100).repeat(batch_size, 1) * (fingertip[:, None, 0] - goal[:, None, 0]) 
        y_points_ret = goal[:, None, 1] + th.linspace(0, 1, steps=100).repeat(batch_size, 1) * (fingertip[:, None, 1] - goal[:, None, 1]) 

        # Concatenate reaching forward then backward along time axis
        forward_traj = th.stack([x_points_ext, y_points_ext], dim=-1)
        backward_traj = th.stack([x_points_ret, y_points_ret], dim=-1)
        self.traj = th.cat([forward_traj, backward_traj], dim=1)

        self.initial_pos = self.states["fingertip"].clone()
        self.hidden_goal = self.initial_pos.clone()

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )
        self.rule_input[:, 5] = 1

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
        return obs, info




class DlyFullCircleClk(env.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None

    def get_obs(self, t, action=None, deterministic: bool = False) -> th.Tensor | np.ndarray:

        self.update_obs_buffer(action=action)

        obs_as_list = [
        self.rule_input,
        self.go_cue[:, t].unsqueeze(1),
        self.vis_inp,
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

        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(t >= self.max_ep_duration)
        self.hidden_goal = self.initial_pos.clone() if t <= self.delay_time else self.traj[:, t - self.delay_time, :].clone()
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        return obs, reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        delay_times = [50, 75, 100]
        self.delay_time = random.choice(delay_times)
        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.delay_time)),
            th.ones(size=(batch_size, int(2 / self.dt)))
        ], dim=-1)
        # Set duration
        self.max_ep_duration = self.go_cue.shape[-1] - 1

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        traj_points = th.linspace(np.pi, -np.pi, 200)
        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in traj_points], dim=0)
        points = (points + th.tensor([[1, 0]])) * 0.25 * 0.5

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1]
        rotated_points = th.zeros(size=(batch_size, int(2 / self.dt), 2))
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
        
        self.traj = rotated_points + fingertip[:, None, :]
        self.vis_inp = self.traj[:, 100, :].clone()
        self.initial_pos = self.states["fingertip"].clone()
        self.hidden_goal = self.initial_pos.clone()

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )
        self.rule_input[:, 6] = 1

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
        return obs, info




class DlyFullCircleCClk(env.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None

    def get_obs(self, t, action=None, deterministic: bool = False) -> th.Tensor | np.ndarray:

        self.update_obs_buffer(action=action)

        obs_as_list = [
        self.rule_input,
        self.go_cue[:, t].unsqueeze(1),
        self.vis_inp,
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
        
        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(t >= self.max_ep_duration)
        self.hidden_goal = self.initial_pos.clone() if t <= self.delay_time else self.traj[:, t - self.delay_time, :].clone()
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        return obs, reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        delay_times = [50, 75, 100]
        self.delay_time = random.choice(delay_times)
        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.delay_time)),
            th.ones(size=(batch_size, int(2 / self.dt)))
        ], dim=-1)
        # Set duration
        self.max_ep_duration = self.go_cue.shape[-1] - 1

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]
        traj_points = th.linspace(np.pi, 3*np.pi, 200)
        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in traj_points], dim=0)
        points = (points + th.tensor([[1, 0]])) * 0.25 * 0.5

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1]
        rotated_points = th.zeros(size=(batch_size, int(2 / self.dt), 2))
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
        
        self.traj = rotated_points + fingertip[:, None, :]
        self.vis_inp = self.traj[:, 100, :].clone()
        self.initial_pos = self.states["fingertip"].clone()
        self.hidden_goal = self.initial_pos.clone()

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )
        self.rule_input[:, 7] = 1

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
        return obs, info




class DlyFigure8(env.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None

    def get_obs(self, t, action=None, deterministic: bool = False) -> th.Tensor | np.ndarray:

        self.update_obs_buffer(action=action)

        obs_as_list = [
        self.rule_input,
        self.go_cue[:, t].unsqueeze(1),
        self.vis_inp,
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

        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(t >= self.max_ep_duration)
        self.hidden_goal = self.initial_pos.clone() if t <= self.delay_time else self.traj[:, t - self.delay_time, :].clone()
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        return obs, reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        delay_times = [50, 75, 100]
        self.delay_time = random.choice(delay_times)
        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.delay_time)),
            th.ones(size=(batch_size, int(2 / self.dt)))
        ], dim=-1)
        # Set duration
        self.max_ep_duration = self.go_cue.shape[-1] - 1

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        x_points_forward = th.linspace(0, 1, 100)
        y_points_forward = th.sin(th.linspace(0, 2*np.pi, 100))

        x_points_back = th.linspace(1, 0, 100)
        y_points_back = -th.sin(th.linspace(2*np.pi, 0, 100))

        # Compute (x, y) coordinates for each angle
        points_forward = th.stack([x_points_forward, y_points_forward], dim=1) * 0.25 * th.tensor([1, 0.5])
        points_back = th.stack([x_points_back, y_points_back], dim=1) * 0.25 * th.tensor([1, 0.5])

        points = th.cat([points_forward, points_back], dim=0)

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1]
        rotated_points = th.zeros(size=(batch_size, int(2 / self.dt), 2))
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
        
        self.traj = rotated_points + fingertip[:, None, :]
        self.vis_inp = self.traj[:, 100, :].clone()
        self.initial_pos = self.states["fingertip"].clone()
        self.hidden_goal = self.initial_pos.clone()

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )
        self.rule_input[:, 8] = 1

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
        return obs, info




class DlyFigure8Inv(env.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim  # target info is noiseless
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None

    def get_obs(self, t, action=None, deterministic: bool = False) -> th.Tensor | np.ndarray:
        self.update_obs_buffer(action=action)

        obs_as_list = [
        self.rule_input,
        self.go_cue[:, t].unsqueeze(1),
        self.vis_inp,
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
        
        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(t >= self.max_ep_duration)
        self.hidden_goal = self.initial_pos.clone() if t <= self.delay_time else self.traj[:, t - self.delay_time, :].clone()
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }

        return obs, reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        reach_conds = options.get('reach_conds', None)
        joint_state = th.tensor([self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1, 
                                self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)
        deterministic: bool = options.get('deterministic', False)
        
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        # Set up max_ep_timesteps separately for each one sampled
        # Set go cue time, randomly sample from a distribution, say (50, 75, 100)
        delay_times = [50, 75, 100]
        self.delay_time = random.choice(delay_times)
        self.go_cue = th.cat([
            th.zeros(size=(batch_size, self.delay_time)),
            th.ones(size=(batch_size, int(2 / self.dt)))
        ], dim=-1)
        # Set duration
        self.max_ep_duration = self.go_cue.shape[-1] - 1

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        x_points_forward = th.linspace(0, 1, 100)
        y_points_forward = -th.sin(th.linspace(0, 2*np.pi, 100))

        x_points_back = th.linspace(1, 0, 100)
        y_points_back = th.sin(th.linspace(2*np.pi, 0, 100))

        # Compute (x, y) coordinates for each angle
        points_forward = th.stack([x_points_forward, y_points_forward], dim=1) * 0.25 * th.tensor([1, 0.5])
        points_back = th.stack([x_points_back, y_points_back], dim=1) * 0.25 * th.tensor([1, 0.5])

        points = th.cat([points_forward, points_back], dim=0)

        # Generate 8 equally spaced angles
        rot_angle = th.linspace(0, 2 * np.pi, 33)[:-1]
        rotated_points = th.zeros(size=(batch_size, int(2 / self.dt), 2))
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
        
        self.traj = rotated_points + fingertip[:, None, :]
        self.vis_inp = self.traj[:, 100, :].clone()
        self.initial_pos = self.states["fingertip"].clone()
        self.hidden_goal = self.initial_pos.clone()

        # Now we need rule input
        self.rule_input = th.zeros(
            size=(batch_size, 10)
        )
        self.rule_input[:, 9] = 1

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
        return obs, info