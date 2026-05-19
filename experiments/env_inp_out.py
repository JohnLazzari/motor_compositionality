import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import motornet as mn
import torch
import os
from utils.exp_utils import env_dict
from utils.plot_utils import create_dir, save_fig
import matplotlib.pyplot as plt
import numpy as np
import config
import tqdm as tqdm
from modules.test import Test
from itertools import product

plt.rcParams.update({"font.size": 18})  # Sets default font size for all text


def _invisible_spines_for_input(ax, col):
    ax[col].spines["top"].set_visible(False)
    ax[col].spines["right"].set_visible(False)
    ax[col].spines["bottom"].set_visible(False)
    return ax


def _v_lines_for_input(ax, col, epoch_bounds):
    ax[col].axvline(epoch_bounds["delay"][0], color="grey", linestyle="dashed")
    ax[col].axvline(epoch_bounds["movement"][0], color="grey", linestyle="dashed")
    ax[col].axvline(epoch_bounds["hold"][0], color="grey", linestyle="dashed")
    return ax


def plot_task_trajectories():
    """This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    exp_path = "results/trajectories"

    create_dir(exp_path)

    for env in env_dict:
        for speed in range(10):
            options = {
                "batch_size": 8,
                "reach_conds": torch.arange(0, 32, 4),
                "speed_cond": speed,
                "delay_cond": 0,
            }
            effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
            cur_env = env_dict[env](effector=effector)
            _, _ = cur_env.reset(testing=True, options=options)

            # Get kinematics and activity in a center out setting
            # On random and delay
            colors = plt.cm.inferno(np.linspace(0, 1, cur_env.traj.shape[1]))

            for tg in cur_env.traj:
                plt.scatter(tg[:, 0], tg[:, 1], s=10, color=colors)
                plt.scatter(tg[0, 0], tg[0, 1], s=150, marker="x", color="black")
                plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker="^", color="black")
            save_fig(os.path.join(exp_path, f"{env}_speed{speed}_tg_trajectory.png"))


def plot_task_feedback(model_name):
    """This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/input"

    create_dir(exp_path)

    options = {
        "batch_size": 8,
        "reach_conds": torch.arange(0, 32, 4),
        "speed_cond": 5,
        "delay_cond": 1,
    }

    test = Test(model_path, model_name)

    for env in env_dict:
        trial_data = test.trial(options, env_dict[env])

        for i, inp in enumerate(trial_data["obs"]):
            fig, ax = plt.subplots(
                9,
                1,
                gridspec_kw={
                    "height_ratios": [1, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 1, 1]
                },
            )
            fig.set_size_inches(4, 8)
            plt.rc("font", size=10)

            colors = plt.cm.Set2(np.linspace(0, 1, 8))

            # Generate Gaussian noise with mean=0 and std=1, shaped [t, n]
            noise = np.random.normal(loc=0.0, scale=0.05, size=(inp.shape[0], 28))

            # --------------------- Rule Input --------------------------

            ax[0].imshow(inp[:, :10].T + noise[:, :10].T, cmap="Purples", aspect="auto")
            _invisible_spines_for_input(ax, 0)
            _v_lines_for_input(ax, 0, trial_data["epoch_bounds"])
            ax[0].set_xticks([])
            ax[0].set_title("Rule Input")

            # --------------------- Speed Scalar --------------------------

            ax[1].plot(inp[:, 10:11], linewidth=4, color=colors[0])
            ax[1].plot(
                inp[:, 10:11] + noise[:, 10:11],
                linewidth=4,
                color=colors[0],
                alpha=0.25,
            )
            _invisible_spines_for_input(ax, 1)
            _v_lines_for_input(ax, 1, trial_data["epoch_bounds"])
            ax[1].set_xticks([])
            ax[1].set_title("Speed Scalar")

            # --------------------- Go Cue --------------------------

            ax[2].plot(inp[:, 11:12], linewidth=4, color=colors[1])
            ax[2].plot(
                inp[:, 11:12] + noise[:, 11:12],
                linewidth=4,
                color=colors[1],
                alpha=0.25,
            )
            _invisible_spines_for_input(ax, 2)
            _v_lines_for_input(ax, 2, trial_data["epoch_bounds"])
            ax[2].set_xticks([])
            ax[2].set_title("Go Cue")

            # --------------------- Target x --------------------------

            ax[3].plot(inp[:, 12:13], linewidth=4, color=colors[2])
            ax[3].plot(
                inp[:, 12:13] + noise[:, 12:13],
                linewidth=4,
                color=colors[2],
                alpha=0.25,
            )
            _invisible_spines_for_input(ax, 3)
            _v_lines_for_input(ax, 3, trial_data["epoch_bounds"])
            ax[3].set_xticks([])
            ax[3].set_title("Target x Position")

            # --------------------- Target y --------------------------

            ax[4].plot(inp[:, 13:14], linewidth=4, color=colors[3])
            ax[4].plot(
                inp[:, 13:14] + noise[:, 13:14],
                linewidth=4,
                color=colors[3],
                alpha=0.25,
            )
            _invisible_spines_for_input(ax, 4)
            _v_lines_for_input(ax, 4, trial_data["epoch_bounds"])
            ax[4].set_xticks([])
            ax[4].set_title("Target y Position")

            # --------------------- Fingertip x --------------------------

            ax[5].plot(inp[:, 14:15], linewidth=4, color=colors[4])
            ax[5].plot(
                inp[:, 14:15] + noise[:, 14:15],
                linewidth=4,
                color=colors[4],
                alpha=0.25,
            )
            _invisible_spines_for_input(ax, 5)
            _v_lines_for_input(ax, 5, trial_data["epoch_bounds"])
            ax[5].set_xticks([])
            ax[5].set_title("Fingertip x Position")

            # --------------------- Fingertip y --------------------------

            ax[6].plot(inp[:, 15:16], linewidth=4, color=colors[5])
            ax[6].plot(
                inp[:, 15:16] + noise[:, 15:16],
                linewidth=4,
                color=colors[5],
                alpha=0.25,
            )
            _invisible_spines_for_input(ax, 6)
            _v_lines_for_input(ax, 6, trial_data["epoch_bounds"])
            ax[6].set_xticks([])
            ax[6].set_title("Fingertip y Position")

            # --------------------- Muscle Length --------------------------

            ax[7].plot(inp[:, 16:22], linewidth=4, color=colors[6])
            ax[7].plot(
                inp[:, 16:22] + noise[:, 16:22],
                linewidth=4,
                color=colors[6],
                alpha=0.25,
            )
            _invisible_spines_for_input(ax, 7)
            _v_lines_for_input(ax, 7, trial_data["epoch_bounds"])
            ax[7].set_xticks([])
            ax[7].set_title("Muscle Length")

            # --------------------- Muscle Velocity --------------------------

            ax[8].plot(inp[:, 22:28], linewidth=4, color=colors[7])
            ax[8].plot(
                inp[:, 22:28] + noise[:, 22:28],
                linewidth=4,
                color=colors[7],
                alpha=0.25,
            )
            _invisible_spines_for_input(ax, 8)
            _v_lines_for_input(ax, 8, trial_data["epoch_bounds"])
            ax[8].set_title("Muscle Velocity")
            ax[8].set_xlabel("Timesteps")

            save_fig(os.path.join(exp_path, f"{env}_input_orientation{i}"), eps=True)


if __name__ == "__main__":
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "plot_task_trajectories":
        plot_task_trajectories()
    elif args.experiment == "plot_task_feedback":
        plot_task_feedback(args.model_name)
    else:
        raise ValueError("Experiment not in this file")
