import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import os
from utils.plot_utils import create_dir, save_fig, standard_2d_ax
import matplotlib.pyplot as plt
import numpy as np
import config
import tqdm as tqdm
from utils.exp_utils import env_dict, load_pickle

plt.rcParams.update({"font.size": 18})  # Sets default font size for all text


def plot_test_performance(model_name):
    """
    This function will plot the validation loss of the model
    for each env during training

    Args:
        model_name (str): name of model that was saved
    """
    exp_path = f"results/{model_name}/performance"

    create_dir(exp_path)

    testing_performance = {}
    for env in env_dict.keys():
        testing_performance[env] = []

    env_test_losses = load_pickle(
        os.path.join("checkpoints", model_name, "val_env_losses.pkl")
    )
    plt.figure(figsize=(10, 6))
    for env in env_test_losses:
        plt.plot(env_test_losses[env], linewidth=4, label=env)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Iterations (500)")
    plt.ylabel("Validation Loss")

    # Get the current axes
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(os.path.join(exp_path, "testing_performance"), eps=True)


# TODO update this function
def plot_test_performance_held_out(model_name):
    """
    Plots the validation performance for tasks

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    model_path = f"checkpoints/{model_name}"
    model_path_baseline = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/performance"

    env_list = ["DlyHalfCircleCClk", "DlyFullCircleCClk"]

    create_dir(exp_path)

    testing_performance = {}
    testing_performance_baseline = {}
    for env in env_list:
        testing_performance[env] = []
        testing_performance_baseline[env] = []

    colors = ["purple", "cyan"]

    with open(os.path.join(model_path, "performance.txt"), "r") as f:
        for line in f:
            for env in env_list:
                if f"{env}|" in line:
                    # Split at the string and get the part after
                    after = line.split(f"{env}|", 1)[1].strip()
                    testing_performance[env].append(float(after))

    with open(os.path.join(model_path_baseline, "performance.txt"), "r") as f:
        for line in f:
            for env in env_list:
                if f"{env}|" in line:
                    # Split at the string and get the part after
                    after = line.split(f"{env}|", 1)[1].strip()
                    testing_performance_baseline[env].append(float(after))

    plt.figure(figsize=(4, 4))
    for i, env in enumerate(testing_performance):
        plt.plot(testing_performance[env], linewidth=4, color=colors[i])
        (baseline,) = plt.plot(
            testing_performance_baseline[env][: len(testing_performance[env])],
            linewidth=4,
            color=colors[i],
            alpha=0.5,
        )
        # Set custom dash pattern: [dash_length, gap_length]
        baseline.set_dashes([1, 1])  # Short dashes and short gaps

    # Get the current axes
    ax = plt.gca()
    ax.set_ylim([0, 0.2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(os.path.join(exp_path, "testing_performance"), eps=True)


# TODO update this function
def plot_transfer_losses():
    model_names = [
        "rnn256_softplus_heldout_transfer",
        "rnn256_softplus_heldout_sin_transfer",
        "rnn256_softplus_heldout_reach_transfer",
        "rnn256_softplus_heldout_nosin_transfer",
        "rnn256_softplus_heldout_nocr_transfer",
        "rnn256_softplus_heldout_cr_transfer",
    ]

    labels = ["all", "sin", "reach", "nosin", "nocr", "cr"]

    fig, ax = standard_2d_ax()
    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    for c, model in enumerate(model_names):
        model_path = f"checkpoints/{model}"
        losses = np.loadtxt(os.path.join(model_path, "losses.txt"))
        avg_losses = []
        for i in range(0, len(losses) - 100, 100):
            avg_losses.append(sum(losses[i : i + 100]) / 100)
        ax.plot(avg_losses, color=colors[c], linewidth=4, label=labels[c])
    # ax.legend()
    save_fig("results/across_model_performance/transfer_losses", eps=True)


if __name__ == "__main__":
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "plot_test_performance":
        plot_test_performance(args.model_name)
    elif args.experiment == "plot_test_performance_held_out":
        plot_test_performance_held_out(args.model_name)
    elif args.experiment == "plot_transfer_losses":
        plot_transfer_losses()
    else:
        raise ValueError("Experiment not in this file")
