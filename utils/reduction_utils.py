import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm
from sklearn.decomposition import PCA
from modules.jPCA import JPCA
from utils.exp_utils import env_dict
from utils.plot_utils import create_dir, save_fig, ax_3d_no_grid, standard_2d_ax
from modules.test import Test
from utils.exp_utils import interpolate_trial

################## Helper functions #################


def _direction_batches(batch_size, speed_cond=5, direction_cond=None):
    """Helper function to get options for batches"""
    reach_conds = (
        np.arange(0, 32, int(32 / batch_size))
        if direction_cond is None
        else direction_cond
    )
    return {
        "batch_size": batch_size,
        "reach_conds": reach_conds,
        "speed_cond": speed_cond,
        "delay_cond": 2,
    }


def _get_pcs_direction(model_name, epoch, env, system, options):
    """
    Get PCs and activity for network across different direction conditions
    at a particular speed

    Args:
        model_name (str): model to evaluate
        epoch (str): either delay or movement
        env (str): name of the environment in env_dict
        system (str): either neural or muscle
        options (dict): dictionary of batch options for env
    """

    model_path = f"checkpoints/{model_name}"

    test = Test(
        model_path,
        model_name,
    )

    trial_data = test.trial(options, env_dict[env])

    if system == "neural":
        mode = "h"
        size = test.hid_size
    elif system == "muscle":
        mode = "muscle_acts"
        size = 6
    else:
        raise ValueError

    if epoch == "delay":
        offset = 0
    elif epoch == "movement":
        offset = 15
    else:
        raise ValueError

    env_h = test.get_epoch(trial_data, epoch, mode, offset=offset)

    pca_3d = PCA()
    pca_3d.fit(env_h.reshape((-1, size)))

    return pca_3d, env_h


def _get_pcs_speeds(model_name, epoch, env, system, direction_cond=0):
    """
    Get PCs and activity for network across different speed conditions
    at a particular direction

    Args:
        model_name (str): model to evaluate
        epoch (str): either delay or movement
        env (str): name of the environment in env_dict
        system (str): either neural or muscle
        direction_cond (int): int value for one of 32 direction conditions
    """

    model_path = f"checkpoints/{model_name}"

    test = Test(
        model_path,
        model_name,
    )

    if system == "neural":
        mode = "h"
        size = test.hid_size
    elif system == "muscle":
        mode = "muscle_acts"
        size = 6
    else:
        raise ValueError

    speed_conds = []
    for s in range(10):
        options = _direction_batches(1, speed_cond=s, direction_cond=direction_cond)
        trial_data = test.trial(options, env_dict[env])

        env_h = test.get_epoch(trial_data, epoch, mode)
        env_h = env_h.squeeze(0)
        speed_conds.append(env_h)

    # Need to interpolate speed conditions to match timepoints
    interpolated_speed_conds = []
    for speed_cond in speed_conds:
        interpolated_speed_conds.append(
            interpolate_trial(speed_cond, speed_conds[-1].shape[1])
        )
    interpolated_speed_conds = torch.stack(interpolated_speed_conds)

    pca_3d = PCA()
    pca_3d.fit(interpolated_speed_conds.reshape((-1, size)))

    return pca_3d, interpolated_speed_conds


################## Primary utility functions #################


def plot_pca3d(model_name, epoch, system, condition):
    exp_path = f"results/{model_name}/pca"
    create_dir(exp_path)

    for env in env_dict:
        if epoch == "delay":
            options = _direction_batches(32, speed_cond=5)
        elif epoch == "movement":
            options = _direction_batches(16, speed_cond=5)
        else:
            raise ValueError

        if condition == "direction":
            pca_3d, env_h = _get_pcs_direction(model_name, epoch, env, system, options)
        elif condition == "speed":
            pca_3d, env_h = _get_pcs_speeds(
                model_name, epoch, env, system, direction_cond=0
            )
        else:
            raise ValueError

        colors = plt.cm.inferno(np.linspace(0, 1, env_h.shape[0]))
        fig, ax = ax_3d_no_grid()

        all_proj = pca_3d.transform(env_h.reshape((-1, env_h.shape[-1])))
        min_z = np.min(all_proj[:, 2])

        for i, h in enumerate(env_h):
            # transform
            h_proj = pca_3d.transform(h)

            # Plot the 3D line
            ax.plot(
                h_proj[:, 0],
                h_proj[:, 1],
                h_proj[:, 2],
                color=colors[i],
                linewidth=4,
                alpha=0.75,
                zorder=10,
            )
            ax.plot(
                h_proj[:, 0], h_proj[:, 1], min_z, color="grey", linewidth=2, alpha=0.5
            )

            # Start and end points (start is triangle, end is x)
            ax.scatter(
                h_proj[0, 0],
                h_proj[0, 1],
                h_proj[0, 2],
                marker="^",
                color=colors[i],
                s=200,
                zorder=10,
            )
            ax.scatter(
                h_proj[-1, 0],
                h_proj[-1, 1],
                h_proj[-1, 2],
                marker="X",
                color=colors[i],
                s=200,
                zorder=10,
            )

        # Set background to white
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        save_fig(
            os.path.join(
                exp_path, "trajs", system, epoch, condition, f"{env}_trajectory"
            ),
            eps=True,
        )

        plt.rc("figure", figsize=(3, 6))
        # Create a figure
        fig = plt.figure()
        # Add a 3D subplot
        ax = fig.add_subplot(111)

        if system == "neural":
            comps = 20
        elif system == "muscle":
            comps = 6
        else:
            raise ValueError

        for c, cond in enumerate(env_h):
            # transform
            cond_proj = pca_3d.transform(cond)
            total_var = cond.var(dim=0).sum()
            ax.plot(
                np.arange(1, comps + 1),
                [cond_proj[:, :i].var(axis=0).sum() / total_var for i in range(comps)],
                marker="o",
                color=colors[c],
                alpha=0.5,
                linewidth=4,
                markersize=20,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        save_fig(
            os.path.join(exp_path, "var", system, epoch, condition, f"{env}_var"),
            eps=True,
        )


def plot_mean_trial_activity(model_name, speed_cond=5, delay_cond=1):
    """Plot mean hidden activity across units over the full trial for each direction."""
    exp_path = f"results/{model_name}/psth"
    model_path = f"checkpoints/{model_name}"
    test = Test(model_path, model_name)

    for env in env_dict:
        options = _direction_batches(32, speed_cond=speed_cond)
        options["delay_cond"] = delay_cond
        trial_data = test.trial(options, env_dict[env])
        mean_activity = trial_data["h"].mean(dim=-1).detach().cpu().numpy()
        colors = plt.cm.inferno(np.linspace(0, 1, mean_activity.shape[0]))

        fig, ax = plt.subplots(figsize=(8, 4))
        timesteps = np.arange(mean_activity.shape[1])
        for direction_idx, activity in enumerate(mean_activity):
            ax.plot(
                timesteps,
                activity,
                color=colors[direction_idx],
                linewidth=4,
                alpha=0.75,
            )

        epoch_bounds = trial_data["epoch_bounds"]
        for epoch in ("stable", "delay", "movement", "hold"):
            epoch_end = epoch_bounds[epoch][1]
            if 0 < epoch_end < mean_activity.shape[1]:
                ax.axvline(epoch_end, color="grey", linestyle=":", linewidth=2)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mean hidden activity")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        save_fig(
            os.path.join(
                exp_path,
                f"speed_{speed_cond}",
                f"delay_{delay_cond}",
                f"{env}_mean_activity",
            ),
            eps=True,
        )


def plot_jpcs(model_name, epoch):
    exp_path = f"results/{model_name}/pca"

    for env in env_dict:
        _, env_h = _get_pcs_speeds(model_name, epoch, env, "neural", direction_cond=4)

        jpca = JPCA()
        formatted_data = [np.array(condition) for condition in env_h]
        jpca.fit(formatted_data)

        projected, _, _, _ = jpca.fit(formatted_data)

        colors = plt.cm.plasma(np.linspace(0, 1, env_h.shape[0]))
        data_list = [data[:, [0, 1]] for data in projected]

        start_x_list = [data[0, 0] for data in data_list]
        color_indices = np.argsort(start_x_list)

        _, ax = standard_2d_ax()

        for i, data in enumerate(np.array(data_list)[color_indices]):
            ax.plot(data[:, 0], data[:, 1], color=colors[i], linewidth=4)

            # Start and end positions
            ax.scatter(
                data[0, 0], data[0, 1], marker="^", color=colors[i], s=250, zorder=10
            )
            ax.scatter(
                data[-1, 0], data[-1, 1], marker="X", color=colors[i], s=250, zorder=10
            )

        save_fig(os.path.join(exp_path, "jpcs", f"{env}_{epoch}"), eps=True)
