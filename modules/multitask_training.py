import torch
import motornet as mn
import numpy as np
import random
import os
import pickle

from modules.models import RNNPolicy, GRUPolicy, RNNMusclePolicy
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
from utils.exp_utils import load_pickle, save_pickle, load_torch_checkpoint

DEF_HP = {
    "network": "rnn",
    "inp_size": 28,
    "hid_size": 256,
    "activation_name": "softplus",
    "noise_level_act": 0.1,
    "noise_level_inp": 0.01,
    "rec_constrained": False,
    "inp_constrained": False,
    "resevoir": False,
    "sparsity": None,
    "spectral_radius": None,
    "dt": 10,
    "t_const": 20,
    "lr": 0.001,
    "batch_size": 4,
    "epochs": 100_000,
    "save_iter": 500,
    "l1_rate": 1e-3,
    "l1_weight": 1e-3,
    "l1_muscle_act": 1e-3,
    "simple_dynamics_weight": 1e-3,
    "zero_feedback": False,
    "single_env": False,
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
        resevoir: bool = DEF_HP["resevoir"],
        sparsity: float | None = DEF_HP["sparsity"],
        spectral_radius: float | None = DEF_HP["spectral_radius"],
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
        zero_feedback: bool = DEF_HP["zero_feedback"],
        single_env: bool = DEF_HP["single_env"],
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
        self.resevoir = resevoir
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
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
        self.zero_feedback = zero_feedback
        self.single_env = single_env
        self.save_model = save_model
        self.training_mode = "arm"

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

        # Save the training parameters from this training object
        self.training_mode = "arm"
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
            "resevoir": self.resevoir,
            "sparsity": self.sparsity,
            "spectral_radius": self.spectral_radius,
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
            checkpoint = load_torch_checkpoint(
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
        env_losses = self._empty_loss_dict(env_dict)
        total_losses = []

        env_test_losses = self._empty_loss_dict(env_dict)
        total_test_losses = []

        interval = 100
        best_test_loss = np.inf

        probs = [1 / len(env_list)] * len(env_list)

        for batch in range(self.epochs):
            # initialize batch
            x = torch.zeros(size=(self.batch_size, self.hid_size))
            h = torch.zeros(size=(self.batch_size, self.hid_size))

            env_name = random.choices(env_list, probs)[0]
            env = env_dict[env_name](
                effector=effector,
                zero_feedback=self.zero_feedback,
                single_env=self.single_env,
            )

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

            # saving losses
            if (batch % interval == 0) and (batch != 0):
                print(
                    "Batch {}/{} Done, mean policy loss: {}".format(
                        batch, self.epochs, sum(total_losses[-interval:]) / interval
                    )
                )

                # separately saving the total losses as well as the env losses
                save_pickle(os.path.join(model_path, "env_losses.pkl"), env_losses)
                np.savetxt(os.path.join(model_path, "total_losses.txt"), total_losses)

            # saving model
            if batch % self.save_iter == 0:
                # Get test loss
                test_loss_envs, test_loss = self.eval(policy, env_dict=env_dict)

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

    def train_kinematics(
        self,
        model_path,
        model_file,
        load_model=False,
        load_optim=False,
        load_model_path=None,
        load_model_file=None,
        data_path=None,
        probs=None,
    ):
        """Train an RNN to predict muscle activity from saved supervised inputs."""
        if self.network != "rnn":
            raise ValueError("Kinematic training currently supports only RNNPolicy")

        create_dir(model_path)
        self.training_mode = "kinematics"
        self.zero_feedback = True
        save_pickle(f"{model_path}/mult_train.pkl", self)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training kinematics on device: {device}")

        if data_path is None:
            data_path = os.path.join(model_path, "muscle_act_data.pkl")
        muscle_data = load_pickle(data_path)

        env_list = list(muscle_data["tasks"])

        if probs is None:
            probs = [1 / len(env_list)] * len(env_list)

        train_direction_indices = self._even_condition_indices(
            len(muscle_data["reach_conds"]), 16
        )
        train_speed_conds = [
            int(muscle_data["speed_conds"][i])
            for i in self._even_condition_indices(len(muscle_data["speed_conds"]), 5)
        ]

        sample_env = env_list[0]
        sample_speed = train_speed_conds[0]
        train_delay_conds = self._kinematic_delay_conds(
            muscle_data, sample_env, sample_speed
        )
        sample_delay_cond = train_delay_conds[0]
        sample_condition_data = self._kinematic_condition_data(
            muscle_data, sample_env, sample_speed, sample_delay_cond
        )
        sample_obs = sample_condition_data["obs"]
        sample_target = sample_condition_data["action"]
        if sample_obs.shape[-1] != self.inp_size:
            raise ValueError(
                f"inp_size={self.inp_size} does not match supervised obs size {sample_obs.shape[-1]}"
            )

        policy = RNNMusclePolicy(
            inp_size=self.inp_size,
            hid_size=self.hid_size,
            output_dim=sample_target.shape[-1],
            activation_name=self.activation_name,
            noise_level_act=self.noise_level_act,
            noise_level_inp=self.noise_level_inp,
            rec_constrained=self.rec_constrained,
            inp_constrained=self.inp_constrained,
            resevoir=self.resevoir,
            sparsity=self.sparsity,
            spectral_radius=self.spectral_radius,
            dt=self.dt,
            t_const=self.t_const,
            device=device,
            output_activation="sigmoid",
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)

        if load_model_path is None:
            load_model_path = model_path
        if load_model_file is None:
            load_model_file = model_file
        if load_model:
            checkpoint = load_torch_checkpoint(
                os.path.join(load_model_path, load_model_file),
                map_location=device,
            )
            policy.load_state_dict(checkpoint["agent_state_dict"])
            if load_optim:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        env_losses = self._empty_loss_dict(env_list)
        total_losses = []
        env_test_losses = self._empty_loss_dict(env_list)
        total_test_losses = []
        interval = 100
        best_test_loss = np.inf

        for batch in range(self.epochs):
            env_name = random.choices(env_list, probs)[0]
            speed = random.choice(train_speed_conds)
            delay_cond = random.choice(train_delay_conds)
            condition_data = self._kinematic_condition_data(
                muscle_data, env_name, speed, delay_cond
            )
            predictions, targets, hs = self._run_kinematic_trial(
                policy,
                condition_data,
                train_direction_indices,
                device,
                noise=True,
            )

            mse_loss = torch.nn.functional.mse_loss(predictions, targets)
            loss = mse_loss
            loss += self.l1_rate(hs, self.l1_rate_scale)
            loss += self.l1_weight(policy, self.l1_weight_scale)
            if self.activation_name != "tanh":
                loss += self.simple_dynamics(
                    hs, policy.mrnn, weight=self.simple_dynamics_weight
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            env_losses[env_name].append(loss.item())
            total_losses.append(loss.item())

            if (batch % interval == 0) and (batch != 0):
                print(
                    "Batch {}/{} Done, mean policy loss: {}".format(
                        batch, self.epochs, sum(total_losses[-interval:]) / interval
                    )
                )
                save_pickle(os.path.join(model_path, "env_losses.pkl"), env_losses)
                np.savetxt(os.path.join(model_path, "total_losses.txt"), total_losses)

            if batch % self.save_iter == 0:
                test_loss_envs, test_loss = self.eval_kinematics(
                    policy, muscle_data, env_list=env_list
                )
                self._update_loss_dict(test_loss_envs, env_test_losses)
                total_test_losses.append(test_loss)
                save_pickle(
                    os.path.join(model_path, "val_env_losses.pkl"), env_test_losses
                )
                np.savetxt(
                    os.path.join(model_path, "val_total_losses.txt"), total_test_losses
                )

                if test_loss <= best_test_loss:
                    best_test_loss = test_loss
                    torch.save(
                        {
                            "agent_state_dict": policy.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "training_mode": "kinematics",
                            "data_path": data_path,
                            "train_direction_indices": train_direction_indices.tolist(),
                            "train_speed_conds": list(train_speed_conds),
                            "train_delay_conds": list(train_delay_conds),
                        },
                        os.path.join(model_path, model_file),
                    )
                    print("Model Saved!")
                    print(f"Directory: {model_path}/{model_file}")
                    print("\n")

    def eval_kinematics(self, policy, muscle_data, env_list=None):
        """Evaluate supervised muscle prediction on every saved condition."""
        if env_list is None:
            env_list = list(muscle_data["tasks"])

        device = next(policy.parameters()).device
        direction_indices = np.arange(len(muscle_data["reach_conds"]))
        speed_conds = [int(speed) for speed in muscle_data["speed_conds"]]
        total_test_loss = 0
        condition_losses = self._empty_loss_dict(env_list)

        for env_name in env_list:
            condition_loss = 0
            condition_count = 0
            for speed in speed_conds:
                delay_conds = self._kinematic_delay_conds(muscle_data, env_name, speed)
                for delay_cond in delay_conds:
                    condition_data = self._kinematic_condition_data(
                        muscle_data, env_name, speed, delay_cond
                    )
                    with torch.no_grad():
                        predictions, targets, _ = self._run_kinematic_trial(
                            policy,
                            condition_data,
                            direction_indices,
                            device,
                            noise=False,
                        )
                    condition_loss += torch.nn.functional.mse_loss(
                        predictions, targets
                    ).item()
                    condition_count += 1
            condition_loss /= condition_count
            condition_losses[env_name].append(condition_loss)
            total_test_loss += condition_loss
        total_test_loss /= len(env_list)

        print("\n")
        print("Kinematic Eval Results:")
        for env_name in env_list:
            print(
                f"Total Loss for Environment {env_name}| "
                f"{condition_losses[env_name][0]}"
            )
        print(f"Total Testing Loss: {total_test_loss}")
        print("\n")
        print("Validation losses saved!")

        return condition_losses, total_test_loss

    def _run_kinematic_trial(
        self,
        policy,
        condition_data,
        direction_indices,
        device,
        noise,
    ):
        """Run a sequence RNN over saved inputs and return muscle predictions/targets."""
        observations = condition_data["obs"][direction_indices]
        targets = condition_data["action"][direction_indices]
        observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
        targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
        batch_size = observations.shape[0]
        x = torch.zeros(size=(batch_size, self.hid_size), device=device)
        h = torch.zeros(size=(batch_size, self.hid_size), device=device)
        _, hs, predictions = policy(observations, x, h, noise=noise)
        return predictions, targets, hs

    @staticmethod
    def _even_condition_indices(num_conditions, num_samples):
        return np.linspace(0, num_conditions - 1, num_samples, dtype=int)

    def _kinematic_delay_conds(self, muscle_data, env_name, speed):
        speed_data = muscle_data["tasks"][env_name][int(speed)]
        if "obs" in speed_data:
            return [None]
        if "delay_conds" in muscle_data:
            return [int(delay_cond) for delay_cond in muscle_data["delay_conds"]]
        return [int(delay_cond) for delay_cond in sorted(speed_data)]

    def _kinematic_condition_data(self, muscle_data, env_name, speed, delay_cond):
        speed_data = muscle_data["tasks"][env_name][int(speed)]
        if "obs" in speed_data:
            return speed_data
        if delay_cond is None:
            raise ValueError("delay_cond is required for nested muscle data")
        return speed_data[int(delay_cond)]

    def eval(self, policy, env_dict=None):
        if env_dict is None:
            env_dict = self.full_env_dict

        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

        # Currently 10 speed conds during testing, 3 for training
        speed_conds = list(np.arange(0, 10))

        total_test_loss = 0
        condition_losses = self._empty_loss_dict(env_dict)

        for env in env_dict:
            condition_loss = 0
            for speed in speed_conds:
                # initialize batch
                # use 32 to get every possible direction
                x = torch.zeros(size=(32, self.hid_size))
                h = torch.zeros(size=(32, self.hid_size))

                cur_env = env_dict[env](
                    effector=effector,
                    zero_feedback=self.zero_feedback,
                    single_env=self.single_env,
                )

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

    def _empty_loss_dict(self, env_dict):
        # initialize loss lists
        env_losses = {}
        for env in env_dict:
            env_losses[env] = []
        return env_losses

    def _update_loss_dict(self, new_dict, old_dict):
        for key in new_dict.keys():
            assert key in old_dict
            old_dict[key].extend(new_dict[key])
