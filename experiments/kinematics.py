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
import config
import tqdm as tqdm
from utils.exp_utils import env_dict, extension_dict, retraction_dict
from modules.test import Test
from utils.plot_utils import save_fig, standard_2d_ax

# Keep all the envs in case transfer to other envs in future
from modules.envs.cclk_curved_reach import CClkCurvedReach
from modules.envs.cclk_cycle import CClkCycle

plt.rcParams.update({"font.size": 18})  # Sets default font size for all text


def plot_kin_2d(ax, tg, xy, color, linestyle="solid", alpha=0.75):
    ax.plot(
        xy[:, 0], xy[:, 1], linewidth=4, linestyle=linestyle, alpha=alpha, color=color
    )
    ax.scatter(xy[0, 0], xy[0, 1], s=150, marker="x", color=color)
    ax.scatter(tg[0], tg[1], s=150, marker="^", color=color)


def plot_1d(pos, env, color):
    plt.plot(pos, linewidth=4, color=color)
    if env in extension_dict:
        plt.xlim([0, 150])
    elif env in retraction_dict:
        plt.xlim([0, 300])
    else:
        raise Exception
    ax = plt.gca()
    # Remove everything
    ax.axis("off")


def plot_task_kinematics(model_name):
    """
    This function plots the kinematics for a model across
    all conditions

    Args:
        model_name (str): name of model
    """
    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/kinematics"

    plt.rc("figure", figsize=(4, 4))

    test = Test(model_path, model_name)

    for env in env_dict:
        for speed in range(10):
            _, ax = standard_2d_ax()

            options = {
                "batch_size": 8,
                "reach_conds": torch.arange(0, 32, 4),
                "speed_cond": speed,
                "delay_cond": 1,
            }

            trial_data = test.trial(options, env_dict[env])

            colors = plt.cm.inferno(np.linspace(0, 1, trial_data["xy"].shape[0]))

            for i, (obs, xy) in enumerate(zip(trial_data["obs"], trial_data["xy"])):
                plot_kin_2d(
                    ax,
                    obs[trial_data["epoch_bounds"]["movement"][0], 12:14],
                    xy,
                    colors[i],
                )

            save_fig(
                os.path.join(exp_path, "scatter", f"{env}_speed{speed}_kinematics"),
                eps=True,
            )


def plot_speed_kinematics(model_name):
    """
    Plot kinematics across speeds in the x and y plane separately

    Args:
        model_name (str): name of model to test on
    """

    model_path = f"checkpoints/{model_name}"
    exp_path = os.path.join(
        f"results/{model_name}/kinematics", "speeds", "speed_kinematics"
    )

    test = Test(model_path, model_name)

    plt.rc("figure", figsize=(8, 4))

    colors = plt.cm.Reds(np.linspace(0, 1, 10))

    for env in env_dict:
        for speed in range(10):
            options = {
                "batch_size": 16,
                "reach_conds": torch.arange(0, 32, 2),
                "speed_cond": speed,
                "delay_cond": 1,
            }
            trial_data = test.trial(options, env_dict[env], noise=True)

            start = trial_data["epoch_bounds"]["movement"][0]
            end = trial_data["epoch_bounds"]["movement"][1]

            x_pos = trial_data["xy"][0, start:end, 0]
            y_pos = trial_data["xy"][0, start:end, 1]

            # x pos
            plot_1d(x_pos, env, colors[speed])
            save_fig(
                os.path.join(
                    exp_path,
                    f"{env}_x_kinematics_speed_{speed}",
                ),
                eps=True,
            )

            # y pos
            plot_1d(y_pos, env, colors[speed])
            save_fig(
                os.path.join(
                    exp_path,
                    f"{env}_y_kinematics_speed_{speed}",
                ),
                eps=True,
            )


# TODO test this function
def plot_task_kinematics_held_out_transfer(model_name):
    """
    This function plots the kinematics on a network that performs transfer learning
    The tasks used for transfer are currently fixed to cclkcr and cclkcycle
    """
    transfer_model_name = f"{model_name}_heldout"
    base_model_path = f"checkpoints/{model_name}"
    transfer_model_path = f"checkpoints/{model_name}_heldout"

    exp_path = f"results/{model_name}/kinematics/held_out_transfer"

    transfer_env_dict = {
        "CClkCurvedReach": CClkCurvedReach,
        "CClkCycle": CClkCycle,
    }

    base_test = Test(base_model_path, model_name)
    transfer_test = Test(
        transfer_model_path, transfer_model_name, add_new_rule_inputs=True
    )

    plt.rc("figure", figsize=(4, 4))

    for env in env_dict:
        options = {
            "batch_size": 8,
            "reach_conds": torch.arange(0, 32, 4),
            "speed_cond": 5,
            "delay_cond": 1,
        }

        base_trial_data = base_test.trial(
            options,
            transfer_env_dict[env],
        )

        transfer_trial_data = transfer_test.trial(
            options,
            env=transfer_env_dict[env],
        )

        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, transfer_trial_data["xy"].shape[0]))

        for i, (tg, xy) in enumerate(zip(base_trial_data["tg"], base_trial_data["xy"])):
            plot_kin_2d(tg, xy, colors[i])

        # Access current axes and hide top/right spines
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        save_fig(
            os.path.join(exp_path, "scatter", f"{env}_before_kinematics"), eps=True
        )

        for i, (tg, xy) in enumerate(
            zip(transfer_trial_data["tg"], transfer_trial_data["xy"])
        ):
            plot_kin_2d(tg, xy, colors[i], linestyle="dashed", alpha=0.5)

        # Access current axes and hide top/right spines
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        save_fig(os.path.join(exp_path, "scatter", f"{env}_kinematics"), eps=True)


if __name__ == "__main__":
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "plot_task_kinematics":
        plot_task_kinematics(args.model_name)
    elif args.experiment == "plot_speed_kinematics":
        plot_speed_kinematics(args.model_name)
    elif args.experiment == "plot_task_kinematics_held_out_transfer":
        plot_task_kinematics_held_out_transfer(args.model_name)
    else:
        raise ValueError("Experiment not in this file")
