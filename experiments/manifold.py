import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import os
import matplotlib.pyplot as plt
import numpy as np
import config
import tqdm as tqdm
from utils.manifold_utils import (
    gather_principal_angles,
    gather_vaf_ratio,
    plot_principal_angles,
    plot_vaf_ratio,
    task_ccs,
    network_muscle_mode_similarity,
)
from utils.plot_utils import save_fig, standard_2d_ax

plt.rcParams.update({"font.size": 18})  # Sets default font size for all text


def neural_principal_angles_task_cond(model_name):
    exp_path = f"results/{model_name}/pc_angles"
    x, angles_dict, control_array = gather_principal_angles(
        model_name, "neural", "task"
    )
    x, angles_dict_cond, control_array_cond = gather_principal_angles(
        model_name, "neural", "condition"
    )
    plt.figure(figsize=(4, 4))
    plot_principal_angles(
        angles_dict_cond, control_array_cond, x, color="skyblue", alpha=0.5
    )
    plot_principal_angles(angles_dict, control_array, x, color="blue", alpha=0.25)
    save_fig(os.path.join(exp_path, "neural_task_cond_principal_angles"), eps=True)


def neural_principal_angles_epoch(model_name):
    exp_path = f"results/{model_name}/pc_angles"
    x, angles_dict, control_array = gather_principal_angles(
        model_name, "neural", "epoch"
    )
    plt.figure(figsize=(4, 4))
    plot_principal_angles(angles_dict, control_array, x, color="blue", alpha=0.75)
    save_fig(os.path.join(exp_path, "neural_epoch_principal_angles"), eps=True)


def muscle_principal_angles_task_cond(model_name):
    exp_path = f"results/{model_name}/pc_angles"
    x, angles_dict, control_array = gather_principal_angles(
        model_name, "muscle", "task"
    )
    x, angles_dict_cond, control_array_cond = gather_principal_angles(
        model_name, "muscle", "condition"
    )
    plt.figure(figsize=(4, 4))
    plot_principal_angles(
        angles_dict_cond, control_array_cond, x, color="skyblue", alpha=0.5
    )
    plot_principal_angles(angles_dict, control_array, x, color="blue", alpha=0.25)
    save_fig(os.path.join(exp_path, "muscle_task_cond_principal_angles"), eps=True)


def neural_vaf_ratio_task_cond(model_name):
    exp_path = f"results/{model_name}/pc_angles"
    vaf_ratio_list, vaf_ratio_control = gather_vaf_ratio(model_name, "neural", "task")
    vaf_ratio_list_cond, vaf_ratio_control_cond = gather_vaf_ratio(
        model_name, "neural", "condition"
    )
    plt.figure(figsize=(4, 4))
    plot_vaf_ratio(vaf_ratio_list, vaf_ratio_control, color="purple")
    plot_vaf_ratio(vaf_ratio_list_cond, vaf_ratio_control_cond, color="violet")
    save_fig(os.path.join(exp_path, "neural_task_cond_vaf_ratio"), eps=True)


def neural_vaf_ratio_epoch(model_name):
    exp_path = f"results/{model_name}/pc_angles"
    vaf_ratio_list, vaf_ratio_control = gather_vaf_ratio(model_name, "neural", "epoch")
    plt.figure(figsize=(4, 4))
    plot_vaf_ratio(vaf_ratio_list, vaf_ratio_control, color="purple")
    save_fig(os.path.join(exp_path, "neural_epoch_vaf_ratio"), eps=True)


def muscle_vaf_ratio_task_cond(model_name):
    exp_path = f"results/{model_name}/pc_angles"
    vaf_ratio_list, vaf_ratio_control = gather_vaf_ratio(model_name, "muscle", "task")
    vaf_ratio_list_cond, vaf_ratio_control_cond = gather_vaf_ratio(
        model_name, "muscle", "condition"
    )
    plt.figure(figsize=(4, 4))
    plot_vaf_ratio(vaf_ratio_list, vaf_ratio_control, color="purple")
    plot_vaf_ratio(vaf_ratio_list_cond, vaf_ratio_control_cond, color="violet")
    save_fig(os.path.join(exp_path, "muscle_task_cond_vaf_ratio"), eps=True)


def plot_network_vs_muscle_ccs(model_name):
    exp_path = f"results/{model_name}/pc_angles"

    network_ccs = task_ccs(model_name, "h")
    muscle_ccs = task_ccs(model_name, "muscle_acts")

    x_muscle = np.arange(1, 5)
    x_network = np.arange(1, 11)
    _, ax = standard_2d_ax()
    for cc in network_ccs:
        ax.plot(x_network, cc, linewidth=4, color="blue", alpha=0.5)
    for cc in muscle_ccs:
        ax.plot(x_muscle, cc, linewidth=4, color="purple", alpha=0.5)
    ax.set_xticks([10])

    save_fig(os.path.join(exp_path, "network_muscle_ccs"), eps=True)


if __name__ == "__main__":
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    # Principle Angles
    if args.experiment == "neural_principal_angles_task_cond":
        neural_principal_angles_task_cond(args.model_name)
    elif args.experiment == "muscle_principal_angles_task_cond":
        muscle_principal_angles_task_cond(args.model_name)
    elif args.experiment == "neural_principal_angles_epoch":
        neural_principal_angles_epoch(args.model_name)

    # VAF
    elif args.experiment == "neural_vaf_ratio_task_cond":
        neural_vaf_ratio_task_cond(args.model_name)
    elif args.experiment == "muscle_vaf_ratio_task_cond":
        muscle_vaf_ratio_task_cond(args.model_name)
    elif args.experiment == "neural_vaf_ratio_epoch":
        neural_vaf_ratio_epoch(args.model_name)

    elif args.experiment == "plot_network_vs_muscle_ccs":
        plot_network_vs_muscle_ccs(args.model_name)

    elif args.experiment == "network_muscle_mode_similarity":
        network_muscle_mode_similarity(args.model_name)

    else:
        raise ValueError("Experiment not in this file")
