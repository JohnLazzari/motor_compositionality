import torch
import motornet as mn
import numpy as np
import random
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
    "data_driven": False,
}


class MultitaskTrainer:
    """
    Train recurrent policies on MotorNet reaching tasks.

    Normal mode uses the existing hand-position loss: the policy receives
    environment observations, drives the arm, and compares fingertip position
    against the task target.

    Data-driven mode still uses the same environment and arm rollout. The saved
    dataset only chooses the condition and supplies target muscle activity;
    observations still come from the environment, so feedback remains present.
    """

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
        data_driven: bool = DEF_HP["data_driven"],
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
        self.data_driven = data_driven
        self.data_path = None
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
        data_path=None,
    ):
        """
        Train the policy in normal or data-driven mode.

        In normal mode, `data_path` is ignored and the primary loss is the
        fingertip-to-target loss. In data-driven mode, `data_path` points to a
        `muscle_act_data.pkl` file. Each batch samples a saved task condition,
        resets the environment to that condition, rolls out the arm using
        environment observations, and uses MSE between generated and saved
        muscle activity as the primary loss.
        """

        # create model path for saving model and hp
        create_dir(model_path)

        if self.data_driven:
            if data_path is None:
                data_path = os.path.join(model_path, "muscle_act_data.pkl")
            self.data_path = data_path
        else:
            self.data_path = None

        # Save the training parameters from this training object
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

        muscle_data = None
        if self.data_driven:
            muscle_data = load_pickle(self.data_path)
            env_list = list(muscle_data["tasks"])
        else:
            env_list = [env for env in env_dict.keys()]

        # initialize loss lists
        env_losses = self._empty_loss_dict(env_list)
        total_losses = []

        env_test_losses = self._empty_loss_dict(env_list)
        total_test_losses = []

        interval = 100
        best_test_loss = np.inf

        probs = [1 / len(env_list)] * len(env_list)

        for batch in range(self.epochs):
            env_name = random.choices(env_list, probs)[0]
            env = env_dict[env_name](
                effector=effector,
                zero_feedback=self.zero_feedback,
                single_env=self.single_env,
            )

            target_muscle_acts = None
            if self.data_driven:
                # Pick a saved task/speed/delay/direction condition for this batch.
                condition = self._sample_data_condition(muscle_data, env_name)
                batch_size = len(condition["direction_indices"])
                reset_kwargs = {
                    "testing": False,
                    "options": {
                        "batch_size": batch_size,
                        "reach_conds": condition["env_reach_conds"],
                        "speed_cond": int(condition["env_speed"]),
                        "delay_cond": int(condition["delay_cond"]),
                        "deterministic": True,
                    },
                }
                target_muscle_acts = self._condition_muscle_acts(
                    condition["condition_data"], condition["direction_indices"], device
                )
            else:
                batch_size = self.batch_size
                reset_kwargs = {"options": {"batch_size": batch_size}}

            # initialize batch
            x = torch.zeros(size=(batch_size, self.hid_size))
            h = torch.zeros(size=(batch_size, self.hid_size))

            # Get first timestep
            obs, info = env.reset(**reset_kwargs)
            terminated = False

            # initial positions and targets
            xy = [info["states"]["fingertip"][:, None, :]]
            tg = [info["goal"][:, None, :]]
            muscle_acts = (
                []
                if self.data_driven
                else [info["states"]["muscle"][:, 0].unsqueeze(1)]
            )
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

            # concatenate into a (batch_size, n_timesteps, features) tensor
            xy = torch.cat(xy, dim=1)
            tg = torch.cat(tg, dim=1)
            muscle_acts = torch.cat(muscle_acts, dim=1)
            hs = torch.cat(hs, dim=1)

            # Implement loss function
            if self.data_driven:
                # Compare generated arm muscle activity to saved muscle activity.
                loss = self.l1_dist(muscle_acts, target_muscle_acts)
            else:
                loss = self.l1_dist(xy, tg)  # L1 loss on position
                loss += self.l1_muscle_act(muscle_acts, self.l1_muscle_act_scale)
            loss += self.l1_rate(hs, self.l1_rate_scale)
            loss += self.l1_weight(policy, self.l1_weight_scale)
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
                test_loss_envs, test_loss = self.eval(
                    policy, env_dict=env_dict, muscle_data=muscle_data
                )

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
                    checkpoint = {
                        "agent_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    if self.data_driven:
                        checkpoint["data_path"] = self.data_path
                    torch.save(checkpoint, model_path + "/" + model_file)
                    print("Model Saved!")
                    print(f"Directory: {model_path}/{model_file}")
                    print("\n")

    def eval(self, policy, env_dict=None, muscle_data=None):
        """Evaluate using the same primary loss as train.

        Normal mode evaluates hand-position loss over every task and speed.
        Data-driven mode evaluates muscle-activity MSE over every saved task,
        speed, delay, and reach direction in the loaded dataset.
        """
        if env_dict is None:
            env_dict = self.full_env_dict

        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

        eval_envs = [env for env in env_dict.keys()]

        total_test_loss = 0
        condition_losses = self._empty_loss_dict(eval_envs)

        for env_name in eval_envs:
            condition_loss = 0
            condition_count = 0
            speed_conds = list(np.arange(0, 10))
            for speed in speed_conds:
                if self.data_driven:
                    delay_cond = int(random.choice(muscle_data["delay_conds"]))
                    condition_data = muscle_data["tasks"][env_name][speed][delay_cond]
                    direction_indices = np.arange(
                        condition_data["muscle_acts"].shape[0]
                    )
                    reach_conds = np.asarray(muscle_data["reach_conds"])[
                        direction_indices
                    ]
                    batch_size = len(direction_indices)
                    reset_kwargs = {
                        "testing": True,
                        "options": {
                            "batch_size": batch_size,
                            "reach_conds": reach_conds,
                            "speed_cond": int(speed),
                            "delay_cond": int(delay_cond),
                            "deterministic": True,
                        },
                    }
                    target_muscle_acts = self._condition_muscle_acts(
                        condition_data, direction_indices, torch.device("cpu")
                    )
                else:
                    batch_size = 32
                    reset_kwargs = {
                        "testing": True,
                        "options": {
                            "batch_size": batch_size,
                            "reach_conds": np.arange(0, 32),
                            "speed_cond": speed,
                        },
                    }
                    target_muscle_acts = None

                x = torch.zeros(size=(batch_size, self.hid_size))
                h = torch.zeros(size=(batch_size, self.hid_size))
                cur_env = env_dict[env_name](
                    effector=effector,
                    zero_feedback=self.zero_feedback,
                    single_env=self.single_env,
                )

                obs, info = cur_env.reset(**reset_kwargs)
                terminated = False
                xy = [info["states"]["fingertip"][:, None, :]]
                tg = [info["goal"][:, None, :]]
                muscle_acts = []
                timestep = 0

                while not terminated:
                    with torch.no_grad():
                        if self.network == "rnn":
                            x, h, action = policy(obs, x, h, noise=False)
                        else:
                            x, h, action = policy(obs, x, h)
                        obs, _, terminated, info = cur_env.step(timestep, action=action)
                    xy.append(info["states"]["fingertip"][:, None, :])
                    tg.append(info["goal"][:, None, :])
                    muscle_acts.append(info["states"]["muscle"][:, 0].unsqueeze(1))
                    timestep += 1

                xy = torch.cat(xy, dim=1)
                tg = torch.cat(tg, dim=1)
                if self.data_driven:
                    muscle_acts = torch.cat(muscle_acts, dim=1)
                    loss = self.l1_dist(muscle_acts, target_muscle_acts)
                else:
                    loss = self.l1_dist(xy, tg)

                condition_loss += loss.item()
                condition_count += 1

            condition_loss /= condition_count
            condition_losses[env_name].append(condition_loss)
            total_test_loss += condition_loss
        total_test_loss /= len(eval_envs)

        print("\n")
        for env in condition_losses:
            print(f"Total Loss for Environment {env}| {condition_losses[env][0]}")
        print(f"Total Testing Loss: {total_test_loss}")
        print("\n")
        print("Validation losses saved!")

        return condition_losses, total_test_loss

    def _sample_data_condition(self, muscle_data, env_name):
        """Choose one saved condition and a batch of reach directions."""
        speed_conditions = [int(speed) for speed in muscle_data["speed_conds"][::2]]
        direction_indices_all = np.arange(len(muscle_data["reach_conds"]))[::2]

        speed = random.choice(speed_conditions)
        delay_cond = int(random.choice(muscle_data["delay_conds"]))
        condition_data = muscle_data["tasks"][env_name][speed][delay_cond]
        replace = self.batch_size > len(direction_indices_all)
        direction_indices = np.random.choice(
            direction_indices_all, size=self.batch_size, replace=replace
        )
        reach_conds = np.asarray(muscle_data["reach_conds"])[direction_indices]
        env_reach_conds = reach_conds // 2
        env_speed = speed // 2

        return {
            "direction_indices": direction_indices,
            "delay_cond": delay_cond,
            "condition_data": condition_data,
            "env_speed": env_speed,
            "env_reach_conds": env_reach_conds,
        }

    @staticmethod
    def _condition_muscle_acts(condition_data, direction_indices, device):
        """Return saved target muscle activity for selected directions."""
        target_indices = torch.as_tensor(
            direction_indices, dtype=torch.long, device=device
        )
        return torch.as_tensor(
            condition_data["muscle_acts"], dtype=torch.float32, device=device
        )[target_indices]

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
