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
        super().__init__(*args, **kwargs)
        self.obs_noise[: self.skeleton.space_dim] = [
            0.0
        ] * self.skeleton.space_dim  # target info is noiseless
        self.hidden_goal = None
        # timestep info
        self.dt = 0.01
        self.go_cue = None
        self.initial_pos = None

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
            self.obs_buffer["vision"][0],
            self.obs_buffer["proprioception"][0],
        ] + self.obs_buffer["action"][: self.action_frame_stacking]

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
            Trajectory only specifies the movement kinematics, the stable, delay, and hold periods simply repeat the first and last hand positions
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
        raise NotImplementedError
