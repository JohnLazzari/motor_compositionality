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
from utils.jpca_util import plot_projections
from utils.exp_utils import env_dict
from utils.plot_utils import create_dir, save_fig
from modules.test import Test
from utils.exp_utils import interpolate_trial

""" The functions here are currently doing pca on each environment then plotting, this may change
"""

# TODO Work on PCs across directions rn, make it generalizable enough to speeds soon
# Figure out how the batches work again to see
# potentially make three functions, pc across direction, speed, and both, make pcs_3d and 2d choose from these


# TODO make these batches more flexible
def _direction_batches(batch_size, speed_cond=5, direction_cond=None):
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
    this function gathers PCs for a particular model, on either the neurons or muscles, for a
    particular epoch and environment. Batches are across directions, speed is fixed.
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

    env_h = test.get_epoch(trial_data, epoch, mode)

    pca_3d = PCA()
    pca_3d.fit(env_h.reshape((-1, size)))

    return pca_3d, env_h


def _get_pcs_speeds(model_name, epoch, env, system, direction_cond=0):
    """
    this function gathers PCs for a particular model, on either the neurons or muscles, for a
    particular epoch and environment. Batches are across directions, speed is fixed.
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

    # cat speeds, should be [speed, direction, T, N]
    speed_conds = torch.stack(speed_conds)

    interpolated_speed_conds = []
    for speed_cond in speed_conds:
        interpolated_speed_conds.append(
            interpolate_trial(speed_cond, speed_conds[-1].shape[1])
        )
    interpolated_speed_conds = torch.stack(interpolated_speed_conds)

    pca_3d = PCA()
    pca_3d.fit(interpolated_speed_conds.reshape((-1, size)))

    return pca_3d, interpolated_speed_conds


def plot_pca3d(model_name, epoch, system, condition):
    exp_path = f"results/{model_name}/pca"
    create_dir(exp_path)

    for env in env_dict:
        options = _direction_batches(32, speed_cond=5)
        if condition == "direction":
            pca_3d, env_h = _get_pcs_direction(model_name, epoch, env, system, options)
        elif condition == "speed":
            pca_3d, env_h = _get_pcs_speeds(
                model_name, epoch, env, system, direction_cond=0
            )
        else:
            raise ValueError

        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, env_h.shape[0]))
        # Create a figure
        fig = plt.figure(figsize=(4, 4))
        # Add a 3D subplot
        ax = fig.add_subplot(111, projection="3d")

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

        # No grid
        ax.grid(False)
        # Set background to white
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        save_fig(
            os.path.join(exp_path, "3d", system, f"{env}_{epoch}_trajectory"), eps=True
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
            os.path.join(exp_path, "3d", system, f"{env}_{condition}_{epoch}_var"),
            eps=True,
        )


# TODO fix this function
def _plot_jpcs(model_name, epoch):
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pca"
    hp = load_hp(model_path)

    options = {
        "batch_size": 8,
        "reach_conds": np.arange(0, 32, int(32 / 8)),
        "speed_cond": 5,
        "delay_cond": 2,
    }

    for env in env_dict:
        trial_data = _test(model_path, model_file, options, env=env_dict[env])

        mode = "h"
        size = hp["hid_size"]

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
                trial_data["epoch_bounds"]["movement"][0] + 10 : trial_data[
                    "epoch_bounds"
                ]["movement"][1],
            ]
        else:
            raise ValueError("not valid epoch")

        jpca = JPCA()
        formatted_data = [np.array(condition) for condition in env_h]
        jpca.fit(formatted_data)

        (projected, full_data_var, pca_var_capt, jpca_var_capt) = jpca.fit(
            formatted_data
        )

        plot_projections(projected)
        save_fig(os.path.join(exp_path, "jpcs", f"{env}_{epoch}"), eps=True)
