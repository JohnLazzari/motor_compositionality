import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm as tqdm

from modules.test import Test
from modules.models import RNNMusclePolicy
from utils.exp_utils import env_dict, load_pickle, load_torch_checkpoint, save_pickle
from utils.plot_utils import save_fig


def _obs_without_feedback(obs):
    return obs[:, :, :14]


def collect_muscle_data(model_name):
    """Collect muscle activity for every task, speed, and direction condition."""
    model_path = f"checkpoints/{model_name}"
    test = Test(model_path, model_name)

    reach_conds = np.arange(0, 32)
    speed_conds = np.arange(0, 10)

    muscle_data = {
        "model_name": model_name,
        "reach_conds": reach_conds,
        "speed_conds": speed_conds,
        "tasks": {},
    }

    for env_name, env in tqdm.tqdm(env_dict.items(), desc="Collecting muscle data"):
        muscle_data["tasks"][env_name] = {}
        for speed in speed_conds:
            options = {
                "batch_size": len(reach_conds),
                "reach_conds": reach_conds,
                "speed_cond": int(speed),
                "deterministic": True,
            }
            trial_data = test.trial(options, env)
            obs = trial_data["obs"]
            if not test.zero_feedback:
                obs = _obs_without_feedback(obs)
            muscle_data["tasks"][env_name][int(speed)] = {
                "muscle_acts": trial_data["muscle_acts"],
                "action": trial_data["action"],
                "obs": obs,
            }

    save_path = os.path.join(model_path, "muscle_act_data.pkl")
    save_pickle(save_path, muscle_data)
    return muscle_data


def _to_numpy(data):
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def load_supervised_model(model_name, device="cpu"):
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    mult_train = load_pickle(os.path.join(model_path, "mult_train.pkl"))
    checkpoint = load_torch_checkpoint(
        os.path.join(model_path, model_file),
        map_location=torch.device(device),
    )

    if getattr(mult_train, "network", None) != "rnn":
        raise ValueError("Supervised model loading only supports RNN checkpoints")
    if getattr(mult_train, "training_mode", None) != "kinematics":
        raise ValueError("Expected a supervised/kinematics checkpoint")

    output_dim = checkpoint["agent_state_dict"]["fc.weight"].shape[0]
    policy = RNNMusclePolicy(
        inp_size=mult_train.inp_size,
        hid_size=mult_train.hid_size,
        output_dim=output_dim,
        activation_name=mult_train.activation_name,
        noise_level_act=mult_train.noise_level_act,
        noise_level_inp=mult_train.noise_level_inp,
        rec_constrained=mult_train.rec_constrained,
        inp_constrained=mult_train.inp_constrained,
        resevoir=getattr(mult_train, "resevoir", False),
        sparsity=getattr(mult_train, "sparsity", None),
        spectral_radius=getattr(mult_train, "spectral_radius", None),
        dt=mult_train.dt,
        t_const=mult_train.t_const,
        device=device,
        output_activation="sigmoid",
    )
    policy.load_state_dict(checkpoint["agent_state_dict"])
    policy.model_name = model_name
    policy.hid_size = mult_train.hid_size
    policy.data_path = checkpoint.get("data_path")
    return policy


def _get_policy_and_hid_size(loaded_model):
    if not isinstance(loaded_model, RNNMusclePolicy):
        raise TypeError("loaded_model must be an RNNMusclePolicy")

    hid_size = getattr(loaded_model, "hid_size", None)
    if hid_size is None:
        hid_size = loaded_model.mrnn.total_num_units
    return loaded_model, hid_size


def _predict_supervised_outputs(loaded_model, obs):
    policy, hid_size = _get_policy_and_hid_size(loaded_model)
    device = next(policy.parameters()).device
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    batch_size = obs.shape[0]
    x = torch.zeros(size=(batch_size, hid_size), device=device)
    h = torch.zeros(size=(batch_size, hid_size), device=device)

    policy.eval()
    with torch.no_grad():
        _, _, outputs = policy(obs, x, h, noise=False)
    return _to_numpy(outputs)


def plot_supervised_outputs(loaded_model, muscle_data, model_name=None):
    """
    Plot supervised model outputs over saved target actions for each condition.

    Each environment/speed pair gets its own directory. Within that directory, one
    figure is saved per direction condition. Target traces are dotted; model
    predictions are solid.
    """
    if model_name is None:
        model_name = getattr(loaded_model, "model_name", None)
    if model_name is None:
        raise ValueError("model_name is required when loaded_model has no model_name")

    save_path = os.path.join("results", model_name, "outputs")
    reach_conds = _to_numpy(muscle_data.get("reach_conds", []))

    for env_name, speed_data in tqdm.tqdm(
        muscle_data["tasks"].items(), desc="Plotting supervised outputs"
    ):
        for speed, data in speed_data.items():
            obs = _to_numpy(data["obs"])
            targets = _to_numpy(data["action"])
            outputs = _predict_supervised_outputs(loaded_model, obs)
            condition_dir = os.path.join(save_path, env_name, f"speed_{speed}")

            colors = plt.cm.tab10(np.linspace(0, 1, targets.shape[-1]))
            for direction_idx in range(targets.shape[0]):
                if len(reach_conds) > direction_idx:
                    direction_label = int(reach_conds[direction_idx])
                else:
                    direction_label = direction_idx

                fig, axes = plt.subplots(
                    targets.shape[-1],
                    1,
                    figsize=(8, 1.6 * targets.shape[-1] + 1.5),
                    sharex=True,
                )
                axes = np.atleast_1d(axes)

                for output_idx, (ax, color) in enumerate(zip(axes, colors)):
                    ax.plot(
                        targets[direction_idx, :, output_idx],
                        color=color,
                        linestyle=":",
                        linewidth=2,
                        label="target",
                    )
                    ax.plot(
                        outputs[direction_idx, :, output_idx],
                        color=color,
                        linewidth=2,
                        label="model",
                    )
                    ax.set_ylabel(
                        f"out {output_idx}", rotation=0, labelpad=25, va="center"
                    )
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                axes[0].legend(frameon=False, ncol=2, fontsize=8)
                axes[-1].set_xlabel("timestep")
                fig.suptitle(f"{env_name} speed {speed} direction {direction_label}")
                save_fig(
                    os.path.join(condition_dir, f"direction_{direction_label}_outputs")
                )


def plot_muscle_data(model_name):
    """Plot saved no-feedback inputs and muscle outputs for each task and speed."""
    model_path = f"checkpoints/{model_name}"
    muscle_data = load_pickle(os.path.join(model_path, "muscle_act_data.pkl"))
    save_path = os.path.join("results", "supervised_data", model_name)

    for env_name, speed_data in tqdm.tqdm(
        muscle_data["tasks"].items(), desc="Plotting muscle data"
    ):
        for speed, data in speed_data.items():
            obs = _to_numpy(data["obs"])
            muscle_acts = _to_numpy(data["muscle_acts"])

            n_inputs = obs.shape[-1]
            n_scalar_inputs = n_inputs - 10
            fig, axes = plt.subplots(
                n_scalar_inputs + 2,
                1,
                figsize=(8, 1.2 * n_scalar_inputs + 5),
                gridspec_kw={"height_ratios": [2] + [1] * n_scalar_inputs + [2]},
                sharex=True,
            )
            rule_ax = axes[0]
            input_axes = axes[1:-1]
            muscle_ax = axes[-1]

            rule_ax.imshow(obs[:, :, :10].mean(axis=0).T, cmap="Purples", aspect="auto")
            rule_ax.set_ylabel("rule", rotation=0, labelpad=25, va="center")
            rule_ax.set_yticks(np.arange(10))
            rule_ax.set_xticks([])
            rule_ax.spines["top"].set_visible(False)
            rule_ax.spines["right"].set_visible(False)
            rule_ax.spines["bottom"].set_visible(False)

            for input_idx, ax in enumerate(input_axes, start=10):
                ax.plot(obs[:, :, input_idx].T, color="0.75", linewidth=0.8, alpha=0.35)
                ax.plot(
                    obs[:, :, input_idx].mean(axis=0),
                    color="black",
                    linewidth=1.5,
                )
                ax.set_ylabel(f"inp {input_idx}", rotation=0, labelpad=25, va="center")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            colors = plt.cm.tab10(np.linspace(0, 1, muscle_acts.shape[-1]))
            for muscle_idx, color in enumerate(colors):
                muscle_ax.plot(
                    muscle_acts[:, :, muscle_idx].T,
                    color=color,
                    linewidth=0.6,
                    alpha=0.12,
                )
                muscle_ax.plot(
                    muscle_acts[:, :, muscle_idx].mean(axis=0),
                    color=color,
                    linewidth=2,
                    label=f"muscle {muscle_idx}",
                )
            muscle_ax.set_ylabel("muscle", rotation=0, labelpad=25, va="center")
            muscle_ax.set_xlabel("timestep")
            muscle_ax.spines["top"].set_visible(False)
            muscle_ax.spines["right"].set_visible(False)
            muscle_ax.legend(frameon=False, ncol=3, fontsize=8)

            fig.suptitle(f"{env_name} speed {speed}")
            save_fig(os.path.join(save_path, f"{env_name}_speed{speed}"))


if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "collect_muscle_data":
        collect_muscle_data(args.model_name)
    elif args.experiment == "plot_muscle_data":
        plot_muscle_data(args.model_name)
    elif args.experiment == "plot_supervised_outputs":
        supervised_model = load_supervised_model(args.model_name)
        data_path = supervised_model.data_path
        muscle_data = load_pickle(data_path)
        plot_supervised_outputs(supervised_model, muscle_data)
    else:
        raise ValueError("Experiment not in this file")
