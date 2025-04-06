import torch as th
import numpy as np
from typing import Any
from motornet import environment as env

class RandomReach(env.Environment):
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
        self.max_ep_duration = 1
        self.dt = 0.01
        self.go_cue = None

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
        self.go_cue[:, t].unsqueeze(1),
        self.goal,
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
        
        self.elapsed += self.dt

        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(self.elapsed >= self.max_ep_duration)
        self.goal = self.goal.clone()
        self.hidden_goal = self.goal.clone()
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
        joint_state: th.Tensor | np.ndarray | None = options.get('joint_state', None)
        deterministic: bool = options.get('deterministic', False)
        
        if joint_state is not None:
            joint_state_shape = np.shape(self.detach(joint_state))
            if joint_state_shape[0] > 1:
                batch_size = joint_state_shape[0]
        else:
            joint_state = self.q_init

        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        self.go_cue = th.ones(size=(batch_size, int(1 / self.dt)))
        self.goal = self.joint2cartesian(self.effector.draw_random_uniform_states(batch_size)).chunk(2, dim=-1)[0]
        self.hidden_goal = self.goal.clone()
        self.elapsed = 0.

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


class DlyRandomReach(env.Environment):
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
        self.max_ep_duration = 2
        self.dt = 0.01
        self.go_cue = None

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
        self.go_cue[:, t].unsqueeze(1),
        self.goal,
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
        
        self.elapsed += self.dt

        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(self.elapsed >= self.max_ep_duration)
        self.goal = self.goal.clone()
        self.hidden_goal = self.goal.clone() if self.elapsed >= 1 else self.states["fingertip"].clone()
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
        joint_state: th.Tensor | np.ndarray | None = options.get('joint_state', None)
        deterministic: bool = options.get('deterministic', False)
        
        if joint_state is not None:
            joint_state_shape = np.shape(self.detach(joint_state))
            if joint_state_shape[0] > 1:
                batch_size = joint_state_shape[0]
        else:
            joint_state = self.q_init

        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        self.go_cue = th.cat([
            th.zeros(size=(batch_size, int(1 / self.dt))),
            th.ones(size=(batch_size, int(1 / self.dt)))
        ], dim=-1)

        self.goal = self.joint2cartesian(self.effector.draw_random_uniform_states(batch_size)).chunk(2, dim=-1)[0]
        self.hidden_goal = self.states["fingertip"].clone()
        self.elapsed = 0.

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


class Maze(env.Environment):
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
        self.all_goals = None
        # timestep info
        self.max_ep_duration = 2.99 # floating point error at 3
        self.dt = 0.01

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
        self.go_cue[:, t].unsqueeze(1),
        self.goal,
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
        
        self.elapsed += self.dt

        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32).to(self.device)
        noisy_action = action
        
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(t, action=noisy_action)
        reward = None if self.differentiable else np.zeros((self.detach(action.shape[0]), 1))
        terminated = bool(self.elapsed >= self.max_ep_duration)

        # Change goal based on time to keep sequence lengths the same
        if self.elapsed <= 1:
            self.goal = self.all_goals[:, 0, :].clone()
            self.hidden_goal = self.goal.clone()
        elif self.elapsed >= 1 and self.elapsed <= 2:
            self.goal = self.all_goals[:, 1, :].clone()
            self.hidden_goal = self.goal.clone()
        elif self.elapsed >= 2:
            self.goal = self.all_goals[:, 2, :].clone()
            self.hidden_goal = self.goal.clone()

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
        joint_state: th.Tensor | np.ndarray | None = options.get('joint_state', None)
        deterministic: bool = options.get('deterministic', False)
        
        if joint_state is not None:
            joint_state_shape = np.shape(self.detach(joint_state))
            if joint_state_shape[0] > 1:
                batch_size = joint_state_shape[0]
        else:
            joint_state = self.q_init

        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        self.go_cue = th.ones(size=(batch_size, int(3 / self.dt)))

        self.all_goals = self.joint2cartesian(self.effector.draw_random_uniform_states(batch_size*3)).chunk(2, dim=-1)[0]
        self.all_goals = self.all_goals.reshape((-1, 3, 2))
        self.goal = self.all_goals[:, 0, :].clone()
        self.hidden_goal = self.goal.clone()
        self.elapsed = 0.

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