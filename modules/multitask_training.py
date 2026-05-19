import torch
import motornet as mn
import numpy as np
import random
import os
import pickle

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
from utils.plot_utils import create_dir
from utils.exp_utils import save_pickle

DEF_HP = {
    "network": "rnn",
    "inp_size": 28,
    "hid_size": 256,
    "activation_name": "softplus",
    "noise_level_act": 0.1,
    "noise_level_inp": 0.01,
    "rec_constrained": False,
    "inp_constrained": False,
    "dt": 10,
    "t_const": 20,
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 75_000,
    "save_iter": 500,
    "l1_rate": 0.001,
    "l1_weight": 0.001,
    "l1_muscle_act": 0.01,
    "simple_dynamics_weight": 0.001,
}


class MultitaskTrainer:
    def __init__(
        self,
        network: str = DEF_HP["network"],
        inp_size: int = DEF_HP["inp_size"],
        hid_size: int = DEF_HP["hid_size"],
        activation_name: str = DEF_HP["activation_name"],
        noise_level_act: float = DEF_HP["noise_level_act"],
        noise_level_inp: float = DEF_HP["noise_level_inp"],
        rec_constrained: bool = DEF_HP["rec_constrained"],
        inp_constrained: bool = DEF_HP["inp_constrained"],
        dt: int = DEF_HP["dt"],
        t_const: int = DEF_HP["t_const"],
        lr: float = DEF_HP["lr"],
        batch_size: int = DEF_HP["batch_size"],
        epochs: int = DEF_HP["epochs"],
        save_iter: int = DEF_HP["save_iter"],
        l1_rate: float = DEF_HP["l1_rate"],
        l1_weight: float = DEF_HP["l1_weight"],
        l1_muscle_act: float = DEF_HP["l1_muscle_act"],
        simple_dynamics_weight: float = DEF_HP["simple_dynamics_weight"],
        save_model: bool = True,
    ):
        self.network = network
        self.inp_size = inp_size
        self.hid_size = hid_size
        self.activation_name = activation_name
        self.noise_level_act = noise_level_act
        self.noise_level_inp = noise_level_inp
        self.rec_constrained = rec_constrained
        self.inp_constrained = inp_constrained
        self.dt = dt
        self.t_const = t_const
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_iter = save_iter
        self.l1_rate_scale = l1_rate
        self.l1_weight_scale = l1_weight
        self.l1_muscle_act_scale = l1_muscle_act
        self.simple_dynamics_weight = simple_dynamics_weight
        self.save_model = save_model

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

    def train(
        self,
        model_path,
        model_file,
        env_dict=None,
        load_model=False,
        load_optim=False,
        load_model_path=None,
        load_model_file=None,
        transfer=False,
    ):
        # create model path for saving model and hp
        create_dir(model_path)

        save_pickle(f"{model_path}/mult_train.pkl", self)
        device = torch.device("cpu")
        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

        mrnn_params = {
            "inp_size": self.inp_size,
            "hid_size": self.hid_size,
            "output_dim": effector.n_muscles,
            "activation_name": self.activation_name,
            "noise_level_act": self.noise_level_act,
            "noise_level_inp": self.noise_level_inp,
            "rec_constrained": self.rec_constrained,
            "inp_constrained": self.inp_constrained,
            "dt": self.dt,
            "t_const": self.t_const,
            "device": device,
        }

        if transfer:
            mrnn_params["add_new_rule_inputs"] = True
            mrnn_params["num_new_inputs"] = 10

        if self.network == "rnn":
            policy = RNNPolicy(**mrnn_params)
        elif self.network == "gru":
            policy = GRUPolicy(
                self.inp_size, self.hid_size, effector.n_muscles, batch_first=True
            )
        else:
            raise ValueError("Not a valid architecture")

        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)

        # default loading paths to given paths if None
        if load_model_path is None:
            load_model_path = model_path
        if load_model_file is None:
            load_model_file = model_file

        if load_model:
            # Load in a previous model
            checkpoint = torch.load(
                os.path.join(load_model_path, load_model_file),
                map_location=torch.device("cpu"),
            )
            policy.load_state_dict(checkpoint["agent_state_dict"])
            if load_optim:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if env_dict is None:
            env_dict = self.full_env_dict
        env_list = [env for env in env_dict.keys()]

        # initialize loss lists
        env_losses = self._empty_loss_dict()
        total_losses = []

        env_test_losses = self._empty_loss_dict()
        total_test_losses = []

        interval = 100
        best_test_loss = np.inf

        probs = [1 / len(env_list)] * len(env_list)

        for batch in range(self.epochs):
            # initialize batch
            x = torch.zeros(size=(self.batch_size, self.hid_size))
            h = torch.zeros(size=(self.batch_size, self.hid_size))

            env_name = random.choices(env_list, probs)[0]
            env = env_dict[env_name](effector=effector)

            # Get first timestep
            obs, info = env.reset(options={"batch_size": self.batch_size})
            terminated = False

            # initial positions and targets
            xy = [info["states"]["fingertip"][:, None, :]]
            tg = [info["goal"][:, None, :]]
            muscle_acts = [info["states"]["muscle"][:, 0].unsqueeze(1)]
            hs = [h.unsqueeze(1)]

            timestep = 0
            # simulate whole episode
            while not terminated:  # will run until `max_ep_duration` is reached
                x, h, action = policy(obs, x, h)
                obs, _, terminated, info = env.step(timestep, action=action)

                fingertip = info["states"]["fingertip"][:, None, :]
                goal = info["goal"][:, None, :]
                muscle_states = info["states"]["muscle"][:, 0].unsqueeze(1)

                xy.append(fingertip)  # trajectories
                tg.append(goal)  # targets
                muscle_acts.append(muscle_states)
                hs.append(h.unsqueeze(1))

                timestep += 1

            # concatenate into a (batch_size, n_timesteps, xy) tensor
            xy = torch.cat(xy, dim=1)
            tg = torch.cat(tg, dim=1)
            muscle_acts = torch.cat(muscle_acts, dim=1)
            hs = torch.cat(hs, dim=1)

            # Implement loss function
            loss = self.l1_dist(xy, tg)  # L1 loss on position
            loss += self.l1_rate(hs, self.l1_rate_scale)
            loss += self.l1_weight(policy, self.l1_weight_scale)
            loss += self.l1_muscle_act(muscle_acts, self.l1_muscle_act_scale)
            if self.activation_name != "tanh":
                loss += self.simple_dynamics(
                    hs, policy.mrnn, weight=self.simple_dynamics_weight
                )

            # backward pass & update weights
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=1.0
            )  # important!
            optimizer.step()

            # add losses for individual envs
            env_losses[env_name].append(loss.item())
            total_losses.append(loss.item())

            if (batch % interval == 0) and (batch != 0):
                print(
                    "Batch {}/{} Done, mean policy loss: {}".format(
                        batch, self.epochs, sum(total_losses[-interval:]) / interval
                    )
                )

                # separately saving the total losses as well as the env losses
                save_pickle(os.path.join(model_path, "env_losses.pkl"), env_losses)
                np.savetxt(os.path.join(model_path, "total_losses.txt"), total_losses)

            if batch % self.save_iter == 0:
                # Get test loss
                test_loss_envs, test_loss = self.eval(policy)

                self._update_loss_dict(test_loss_envs, env_test_losses)
                total_test_losses.append(test_loss)

                # separately saving the validation total losses as well as the env losses
                save_pickle(
                    os.path.join(model_path, "val_env_losses.pkl"), env_test_losses
                )
                np.savetxt(
                    os.path.join(model_path, "val_total_losses.txt"), total_test_losses
                )

                # If current test loss is better than previous, save model and update best loss
                if test_loss <= best_test_loss:
                    best_test_loss = test_loss
                    torch.save(
                        {
                            "agent_state_dict": policy.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        model_path + "/" + model_file,
                    )
                    print("Model Saved!")
                    print(f"Directory: {model_path}/{model_file}")
                    print("\n")

    def eval(self, policy, env_dict=None):
        if env_dict is None:
            env_dict = self.full_env_dict

        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

        # Currently 10 speed conds during testing, 3 for training
        speed_conds = list(np.arange(0, 10))

        total_test_loss = 0
        condition_losses = self._empty_loss_dict()

        for env in env_dict:
            condition_loss = 0
            for speed in speed_conds:
                # initialize batch
                # use 32 to get every possible direction
                x = torch.zeros(size=(32, self.hid_size))
                h = torch.zeros(size=(32, self.hid_size))

                cur_env = env_dict[env](effector=effector)

                # Get first timestep
                obs, info = cur_env.reset(
                    testing=True,
                    options={
                        "batch_size": 32,
                        "reach_conds": np.arange(0, 32),
                        "speed_cond": speed,
                    },
                )
                terminated = False

                # initial positions and targets
                xy = [info["states"]["fingertip"][:, None, :]]
                tg = [info["goal"][:, None, :]]

                timestep = 0
                # simulate whole episode
                while not terminated:  # will run until `max_ep_duration` is reached
                    with torch.no_grad():
                        x, h, action = policy(obs, x, h)
                        obs, _, terminated, info = cur_env.step(timestep, action=action)

                    xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
                    tg.append(info["goal"][:, None, :])  # targets

                    timestep += 1

                # concatenate into a (batch_size, n_timesteps, xy) tensor
                xy = torch.cat(xy, dim=1)
                tg = torch.cat(tg, dim=1)

                # Implement loss function
                loss = self.l1_dist(xy, tg)  # L1 loss on position
                condition_loss += loss.item()
            condition_loss /= len(speed_conds)
            condition_losses[env].append(condition_loss)
            total_test_loss += condition_loss
        total_test_loss /= len(env_dict)

        print("\n")
        print("Eval Results:")
        for env in condition_losses:
            print(f"Total Loss for Environment {env}| {condition_losses[env][0]}")
        print(f"Total Testing Loss: {total_test_loss}")
        print("\n")

        print("Validation losses saved!")

        return condition_losses, total_test_loss

    @staticmethod
    def l1_dist(x, y):
        """L1 loss"""
        return torch.mean(torch.sum(torch.abs(x - y), dim=-1))

    @staticmethod
    def l1_weight(rnn, scale: float):
        l1 = 0
        for _, param in rnn.named_parameters():
            l1 += torch.mean(torch.abs(torch.flatten(param)))
        l1 *= scale
        return l1

    @staticmethod
    def l1_rate(act, scale: float):
        l1 = scale * torch.mean(torch.abs(torch.flatten(act)))
        return l1

    @staticmethod
    def l1_muscle_act(act, scale: float):
        l1 = scale * torch.mean(torch.sum(torch.abs(act), dim=-1))
        return l1

    @staticmethod
    def simple_dynamics(act, mrnn, weight=1e-2):
        W_rec, W_rec_mask, W_rec_sign = mrnn.gen_w(mrnn.region_dict)
        if mrnn.rec_constrained:
            W_rec = mrnn.apply_dales_law(W_rec, W_rec_mask, W_rec_sign)

        if mrnn.activation_name == "softplus":
            derivative = 1 / (1 + torch.exp(-act))
        elif mrnn.activation_name == "relu":
            derivative = torch.where(act > 0, 1.0, 0.0)
        else:
            raise ValueError("Not implemented for activation")

        d_act = torch.mean(derivative, dim=(1, 0))

        update = W_rec * d_act**2
        update = weight * torch.norm(update)

        return update

    def _empty_loss_dict(self):
        # initialize loss lists
        env_losses = {}
        for env in self.full_env_dict:
            env_losses[env] = []
        return env_losses

    def _update_loss_dict(self, new_dict, old_dict):
        for key in new_dict.keys():
            assert key in old_dict
            old_dict[key].extend(new_dict[key])
