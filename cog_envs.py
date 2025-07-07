import torch as th
import numpy as np
from typing import Any
from motornet import environment as env
from itertools import product
import random

# Comments in first environment (below) should follow for others
class CogMotorEnv(env.Environment):
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
        self.go_cue[:, t],
        self.stim_1[:, t],
        self.stim_2[:, t],
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
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError
    

class Go(CogMotorEnv):
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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """
        Uses the :meth:`Environment.reset()` method of the parent class :class:`Environment` that can be overwritten to 
        change the returned data. Here the goals (`i.e.`, the targets) are drawn from a random uniform distribution across
        the full joint space.
        """

        #------------------------------------- START OPTION AND EFFECTOR DEFINITIONS 

        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        delay_cond = options.get('delay_cond', None)
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
        delay_times = [50, 75, 100]
        if delay_cond is None:
            self.delay_time = random.choice(delay_times)
        else:
            self.delay_time = delay_times[delay_cond]
        
        self.movement_time = 50

        # By here we should have the lengths of all task epochs
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, self.stable_time+self.delay_time),
            "movement": (self.stable_time+self.delay_time, self.stable_time+self.delay_time+self.movement_time),
            "hold": (self.stable_time+self.delay_time+self.movement_time, self.stable_time+self.delay_time+self.movement_time+self.hold_time),
        }

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

        self.rule_input[:, 0] = 1

        #------------------------------------- END


        #------------------------------------- START KINEMATIC TRAJECTORY

        # Get fingertip position for the target
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        # Generate 8 equally spaced angles
        angles = th.linspace(0, 2 * np.pi, 33)[:-1]

        # this wont work yet cause everything else has shape batch_size (or I can assert reach_conds and batch_size are same shape)
        #stim_idx = th.randint(0, 2)
        stim_idx = 0
        point_idx = th.randint(0, angles.size(0), (batch_size,))

        # Compute (x, y) coordinates for each angle
        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in angles], dim=0)
        
        if stim_idx == 0:
            stim_1 = 0.8 * np.exp(-0.5 * ((8 * np.abs(angles.unsqueeze(1) - angles[point_idx].unsqueeze(0))) / np.pi)**2)
            stim_2 = th.zeros_like(stim_1)
        else:
            stim_2 = 0.8 * np.exp(-0.5 * ((8 * np.abs(angles.unsqueeze(1) - angles[point_idx].unsqueeze(0))) / np.pi)**2)
            stim_1 = th.zeros_like(stim_2)

        stim_1 = th.tensor(stim_1, dtype=th.float32).T  
        stim_2 = th.tensor(stim_2, dtype=th.float32).T  

        goal = points[point_idx] * 0.25 + fingertip

        # Draw a line from fingertip to goal 
        x_points = fingertip[:, None, 0] + th.linspace(0, 1, steps=self.movement_time).repeat(batch_size, 1) * (goal[:, None, 0] - fingertip[:, None, 0]) 
        y_points = fingertip[:, None, 1] + th.linspace(0, 1, steps=self.movement_time).repeat(batch_size, 1) * (goal[:, None, 1] - fingertip[:, None, 1]) 

        self.traj = th.stack([x_points, y_points], dim=-1)

        # We want to start target onset after stable epoch
        #self.stim_1 = th.cat([
            # [batch_size, stability timesteps, xy]
            #th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], stim_1.shape[-1])),
            # [batch_size, delay->hold timesteps, xy]
            #stim_1.repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        #], dim=1)

        self.stim_1 = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], stim_1.shape[-1])),
            stim_1.unsqueeze(1).repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
        ], dim=1)

        self.stim_2 = th.cat([
            th.zeros(size=(batch_size, self.epoch_bounds["stable"][1], stim_2.shape[-1])),
            stim_2.unsqueeze(1).repeat(1, self.epoch_bounds["hold"][1] - self.epoch_bounds["delay"][0], 1)
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

class DelayGo(CogMotorEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self._set_generator(seed=seed)
        options = {} if options is None else options

        batch_size: int = options.get('batch_size', 1)
        
        deterministic: bool = options.get('deterministic', False)

        joint_state = th.tensor([
            self.effector.pos_range_bound[0] * 0.5 + self.effector.pos_upper_bound[0] + 0.1,
            self.effector.pos_range_bound[1] * 0.5 + self.effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(batch_size, 1)

        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})
       
        #------------------------------------- START EPOCH DURATION AND BOUNDS DEFINITIONS

        # Epoch times
        fix_time = 50
        stim_time = 50
        delay_time = random.choice([50, 75, 100])
        go_time = 50

        self.epoch_bounds = {
            "fixed_point": (0, fix_time),
            "stimulus": (fix_time, fix_time + stim_time),
            "delay": (fix_time + stim_time, fix_time + stim_time + delay_time),
            "go": (fix_time + stim_time + delay_time, fix_time + stim_time + delay_time + go_time),
        }

        self.max_ep_duration = self.epoch_bounds["go"][1] - 1
        
        
        #------------------------------------- START STATIC NETWORK INPUT (NOT FEEDBACK)

        self.go_cue = th.cat([
            th.zeros(batch_size, self.epoch_bounds["delay"][1], 1),
            th.ones(batch_size, self.epoch_bounds["go"][1] - self.epoch_bounds["delay"][1], 1)
        ], dim=1)

        # Rule input 
        self.rule_input = th.zeros(batch_size, 20)
        self.rule_input[:, 1] = 1

        #------------------------------------- END

        #------------------------------------- START KINEMATIC TRAJECTORY
        
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        angles = th.linspace(0, 2 * np.pi, 33)[:-1]

        point_idx = th.randint(0, angles.size(0), (batch_size,))
        stim_idx = th.randint(0, 2, (batch_size,))

        points = th.stack([th.tensor([np.cos(angle), np.sin(angle)]) for angle in angles], dim=0)
        goal = points[point_idx] * 0.25 + fingertip

        # Trajectory toward goal
        x_points = fingertip[:, None, 0] + th.linspace(0, 1, steps=go_time).repeat(batch_size, 1) * (goal[:, None, 0] - fingertip[:, None, 0])
        y_points = fingertip[:, None, 1] + th.linspace(0, 1, steps=go_time).repeat(batch_size, 1) * (goal[:, None, 1] - fingertip[:, None, 1])

        self.traj = th.stack([x_points, y_points], dim=-1)

        # ----- Stimuli 1 & 2 -----
        stim_1 = th.zeros(batch_size, angles.shape[0])
        stim_2 = th.zeros(batch_size, angles.shape[0])
        
        for b in range(batch_size):
            stim_strength = 0.8 * np.exp(-0.5 * ((8 * np.abs(angles - angles[point_idx[b]])) / np.pi) ** 2)
            if stim_idx[b] == 0:
                stim_1[b] = th.tensor(stim_strength)
            else:
                stim_2[b] = th.tensor(stim_strength)

        # Temporal stimulus inputs
        self.stim_1 = th.cat([
            th.zeros(batch_size, self.epoch_bounds["fixed_point"][1], stim_1.shape[-1]),
            stim_1.unsqueeze(1).repeat(1, self.epoch_bounds["go"][1] - self.epoch_bounds["stimulus"][0], 1)
        ], dim=1)

        self.stim_2 = th.cat([
            th.zeros(batch_size, self.epoch_bounds["fixed_point"][1], stim_2.shape[-1]),
            stim_2.unsqueeze(1).repeat(1, self.epoch_bounds["go"][1] - self.epoch_bounds["stimulus"][0], 1)
        ], dim=1)

        self.hidden_goal = self.traj[:, 0, :].clone()
        
        # ----- MotorNet observations -----
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
    
class ReactGo(CogMotorEnv):
   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        deterministic: bool = options.get('deterministic', False)

        # ------------------------------------- SET EPOCH DURATIONS

        self.stable_time = 50  
        self.movement_time = 40  
        self.hold_time = 30  

        stim_on = self.stable_time
        stim_off = stim_on + self.movement_time

        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "stim": (stim_on, stim_off),
            "hold": (stim_off, stim_off + self.hold_time),
        }

        self.max_ep_duration = self.epoch_bounds["hold"][1] - 1

        #------------------------------------- START STATIC NETWORK INPUT (NOT FEEDBACK)

        self.go_cue = th.cat([
            th.zeros((batch_size, self.epoch_bounds["stim"][0], 1)),
            th.ones((batch_size, self.epoch_bounds["hold"][1] - self.epoch_bounds["stim"][0], 1))
        ], dim=1)

        self.rule_input = th.zeros(batch_size, 20)
        self.rule_input[:, 2] = 1

        #------------------------------------- START KINEMATIC TRAJECTORY

        fingertip = self.joint2cartesian(self.effector.init_joint_state(batch_size))[..., :2]
        angles = th.linspace(0, 2 * np.pi, 33)[:-1]
        point_idx = th.randint(0, angles.size(0), (batch_size,))
        stim_angle = angles[point_idx]

        points = th.stack([th.cos(stim_angle), th.sin(stim_angle)], dim=-1)

        radius = 0.3

        goal = fingertip + radius * response_points
        self.hidden_goal = goal.clone()

        # Movement trajectory: linear from fingertip to goal
        traj_steps = th.linspace(0, 1, self.movement_time).to(goal.device)
        x_points = fingertip[:, None, 0] + traj_steps[None, :] * (goal[:, None, 0] - fingertip[:, None, 0])
        y_points = fingertip[:, None, 1] + traj_steps[None, :] * (goal[:, None, 1] - fingertip[:, None, 1])
        self.traj = th.stack([x_points, y_points], dim=-1)

        # ------------------------------------- OBSERVATION BUFFER INIT

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
