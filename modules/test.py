import torch
import motornet as mn
import os

from modules.models import RNNPolicy, GRUPolicy
from modules.envs.reach import Reach
from modules.envs.clk_curved_reach import ClkCurvedReach
from modules.envs.cclk_curved_reach import CClkCurvedReach
from modules.envs.sinusoid import Sinusoid
from modules.envs.inv_sinusoid import InvSinusoid
from modules.envs.reach_back import ReachBack
from modules.envs.clk_cycle import ClkCycle
from modules.envs.cclk_cycle import CClkCycle
from modules.envs.figure_eight import Figure8
from modules.envs.inv_figure_eight import InvFigure8
from modules.multitask_training import MultitaskTrainer
from utils.exp_utils import load_pickle, load_torch_checkpoint


class Test:
    def __init__(
        self,
        model_path,
        model_name,
        noise_level_act=None,
        noise_level_inp=None,
        device="cpu",
        add_new_rule_inputs=False,
        num_new_inputs=10,
    ):
        """
        Trial object, stores functions to run a single trial and get data
        Can be used for testing with a loaded model (pass in model info to init)
        or can be used during training by passing in the corresponding rnn to self.trial
        """

        # test specific params for loading model
        self.model_path = model_path
        self.model_name = model_name
        self.model_file = f"{model_name}.pth"

        mult_train = load_pickle(f"{self.model_path}/mult_train.pkl")

        # params used during training
        self.network = mult_train.network
        self.inp_size = mult_train.inp_size
        self.hid_size = mult_train.hid_size
        self.activation_name = mult_train.activation_name
        self.noise_level_act = (
            mult_train.noise_level_act if noise_level_act is None else noise_level_act
        )
        self.noise_level_inp = (
            mult_train.noise_level_inp if noise_level_inp is None else noise_level_inp
        )
        self.rec_constrained = mult_train.rec_constrained
        self.inp_constrained = mult_train.inp_constrained
        self.resevoir = getattr(mult_train, "resevoir", False)
        self.sparsity = getattr(mult_train, "sparsity", None)
        self.spectral_radius = getattr(mult_train, "spectral_radius", None)
        self.dt = mult_train.dt
        self.t_const = mult_train.t_const
        self.lr = mult_train.lr
        self.batch_size = mult_train.batch_size
        self.epochs = mult_train.epochs
        self.save_iter = mult_train.save_iter
        self.l1_rate_scale = mult_train.l1_rate_scale
        self.l1_weight_scale = mult_train.l1_weight_scale
        self.l1_muscle_act_scale = mult_train.l1_muscle_act_scale
        self.simple_dynamics_weight = mult_train.simple_dynamics_weight
        self.zero_feedback = mult_train.zero_feedback
        self.single_env = mult_train.single_env
        self.device = device
        self.add_new_rule_inputs = add_new_rule_inputs
        self.num_new_inputs = num_new_inputs

        self.policy = self.load_model(
            model_path,
            self.model_file,
            self.network,
            self.inp_size,
            self.hid_size,
            self.activation_name,
            self.rec_constrained,
            self.inp_constrained,
            self.resevoir,
            self.sparsity,
            self.spectral_radius,
            self.dt,
            self.t_const,
            self.noise_level_act,
            self.noise_level_inp,
            device,
            add_new_rule_inputs,
            num_new_inputs,
        )

        self.full_env_dict = {
            "Reach": Reach,
            "ClkCurvedReach": ClkCurvedReach,
            "CClkCurvedReach": CClkCurvedReach,
            "Sinusoid": Sinusoid,
            "InvSinusoid": InvSinusoid,
            "ReachBack": ReachBack,
            "ClkCycle": ClkCycle,
            "CClkCycle": CClkCycle,
            "Figure8": Figure8,
            "InvFigure8": InvFigure8,
        }

    def trial(
        self,
        options,
        env,
        noise=False,
        rule_input=None,
    ):
        """
        Runs a test episode in the specified environment using a trained RNN or GRU policy.

        Parameters:
        -----------
        model_path : str
            Path to the trained model directory.
        model_file : str
            Filename of the model checkpoint.
        options : dict
            Dictionary of environment and test configuration options (e.g., batch size).
        env : callable
            Environment constructor. Should accept an effector keyword argument.
        stim : torch.Tensor, optional
            Tensor to silence or stimulate specific hidden units. Default is None.
        noise : bool, optional
            Whether to inject noise into the network. Default is False.
        noise_act : float, optional
            Standard deviation of noise added to activations. Default is 0.1.
        noise_inp : float, optional
            Standard deviation of noise added to inputs. Default is 0.01.
        add_new_rule_inputs : bool, optional
            Whether to add additional rule inputs to the RNN. Default is False.
        num_new_inputs : int, optional
            Number of new rule inputs to add if `add_new_rule_inputs` is True. Default is 10.

        Returns:
        --------
        trial_data : dict
            Dictionary containing recorded trajectories and internal states for the episode, including:
                - 'h', 'x': hidden and internal states
                - 'action': actions taken by the model
                - 'obs': observations received
                - 'xy', 'tg': fingertip positions and target positions
                - 'muscle_acts': muscle activations
                - 'epoch_bounds': dictionary from the environment
        """

        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        env = env(
            effector=effector,
            zero_feedback=self.zero_feedback,
            single_env=self.single_env,
        )

        self.batch_size = options["batch_size"]

        # initialize batch
        x = torch.zeros(size=(self.batch_size, self.hid_size))
        h = torch.zeros(size=(self.batch_size, self.hid_size))

        obs, info = env.reset(testing=True, options=options)
        terminated = False
        trial_data = {}
        timesteps = 0

        trial_data["h"] = []
        trial_data["x"] = []
        trial_data["action"] = []
        trial_data["muscle_acts"] = []
        trial_data["obs"] = []
        trial_data["xy"] = []
        trial_data["tg"] = []

        # simulate whole episode
        while not terminated:  # will run until `max_ep_duration` is reached
            if rule_input is not None:
                obs = self._replace_rule_input(rule_input, obs)

            with torch.no_grad():
                # Check if silencing units
                x, h, action = self.policy(obs, x, h, noise=noise)
                # Take step in motornet environment
                obs, _, terminated, info = env.step(timesteps, action=action)

            timesteps += 1

            # Save all information regarding episode step
            trial_data["h"].append(h.unsqueeze(1))  # trajectories
            trial_data["x"].append(x.unsqueeze(1))  # trajectories
            trial_data["action"].append(action.unsqueeze(1))  # targets
            trial_data["obs"].append(obs.unsqueeze(1))  # targets
            trial_data["xy"].append(
                info["states"]["fingertip"][:, None, :]
            )  # trajectories
            trial_data["tg"].append(info["goal"][:, None, :])  # targets
            trial_data["muscle_acts"].append(
                info["states"]["muscle"][:, 0].unsqueeze(1)
            )

        # Concatenate all data into single tensor
        for key in trial_data:
            trial_data[key] = torch.cat(trial_data[key], dim=1)

        loss = MultitaskTrainer.l1_dist(
            trial_data["xy"], trial_data["tg"]
        )  # L1 loss on position
        trial_data["test_loss"] = loss

        trial_data["epoch_bounds"] = env.epoch_bounds

        return trial_data

    @staticmethod
    def load_model(
        model_path,
        model_file,
        network,
        inp_size,
        hid_size,
        activation_name,
        rec_constrained,
        inp_constrained,
        resevoir,
        sparsity,
        spectral_radius,
        dt,
        t_const,
        noise_act,
        noise_inp,
        device,
        add_new_rule_inputs=False,
        num_new_inputs=10,
    ):
        # Loading in model
        if network == "rnn":
            policy = RNNPolicy(
                inp_size,
                hid_size,
                6,  # Number of muscles in effector
                activation_name=activation_name,
                noise_level_act=noise_act,
                noise_level_inp=noise_inp,
                rec_constrained=rec_constrained,
                inp_constrained=inp_constrained,
                resevoir=resevoir,
                sparsity=sparsity,
                spectral_radius=spectral_radius,
                dt=dt,
                t_const=t_const,
                device=device,
                add_new_rule_inputs=add_new_rule_inputs,
                num_new_inputs=num_new_inputs,
            )

            checkpoint = load_torch_checkpoint(
                os.path.join(model_path, model_file),
                map_location=torch.device("cpu"),
            )
            policy.load_state_dict(checkpoint["agent_state_dict"])

        elif network == "gru":
            policy = GRUPolicy(inp_size, hid_size, 6, batch_first=True)

            checkpoint = load_torch_checkpoint(
                os.path.join(model_path, model_file),
                map_location=torch.device("cpu"),
            )
            policy.load_state_dict(checkpoint["agent_state_dict"])

        else:
            raise ValueError("Not a valid architecture")

        return policy

    @staticmethod
    def _replace_rule_input(composite_inp, obs):
        B, _ = obs.shape
        # This will not account for greater than two
        if composite_inp.dim() < 2:
            composite_inp = composite_inp.repeat(B, 1)
        obs_new_rule = torch.cat([composite_inp, obs[:, 10:]], dim=1)
        return obs_new_rule

    @staticmethod
    def get_epoch(trial_data, epoch, mode, offset=0):
        """
        gathers an epoch from trial data, which should contain the information given from
        running self.trial (is a dictionary).
        """
        if epoch == "delay":
            env_h = trial_data[mode][
                :,
                trial_data["epoch_bounds"]["delay"][0] : trial_data["epoch_bounds"][
                    "delay"
                ][1],
            ]

        elif epoch == "movement":
            env_h = trial_data[mode][
                :,
                trial_data["epoch_bounds"]["movement"][0] + offset : trial_data[
                    "epoch_bounds"
                ]["movement"][1],
            ]
        else:
            raise ValueError("not valid epoch")

        return env_h
