import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings("ignore")

import torch
import os
from utils import create_dir, save_fig
import matplotlib.pyplot as plt
import numpy as np
import config
import tqdm as tqdm
from exp_utils import _test, env_dict
from plt_utils import standard_2d_ax

plt.rcParams.update({'font.size': 18})  # Sets default font size for all text

def plot_test_performance(model_name):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/performance"

    env_list = [
        "DlyHalfReach",
        "DlyHalfCircleClk", 
        "DlyHalfCircleCClk", 
        "DlySinusoid", 
        "DlySinusoidInv",
        "DlyFullReach",
        "DlyFullCircleClk",
        "DlyFullCircleCClk",
        "DlyFigure8",
        "DlyFigure8Inv"
    ]

    create_dir(exp_path)

    testing_performance = {}
    for env in env_list:
        testing_performance[env] = []

    with open(os.path.join(model_path, "performance.txt"), "r") as f:
        for line in f:
            for env in env_list:
                if f"{env}|" in line:
                    # Split at the string and get the part after
                    after = line.split(f"{env}|", 1)[1].strip()
                    testing_performance[env].append(float(after))
    
    plt.figure(figsize=(6, 4))
    for env in testing_performance:
        plt.plot(testing_performance[env], linewidth=4, label=env)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Iterations (500)")
    plt.ylabel("Validation Loss")
    # Get the current axes
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, "testing_performance"), eps=True)





def plot_test_performance_held_out(model_name):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    model_path = f"checkpoints/{model_name}"
    model_path_baseline = f"checkpoints/rnn256_softplus_sd1e-3_ma1e-2"
    exp_path = f"results/{model_name}/performance"

    env_list = [
        "DlyHalfCircleCClk",
        "DlyFullCircleCClk"
    ]

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
        baseline, = plt.plot(testing_performance_baseline[env][:len(testing_performance[env])], linewidth=4, color=colors[i], alpha=0.5)
        # Set custom dash pattern: [dash_length, gap_length]
        baseline.set_dashes([1, 1])  # Short dashes and short gaps
    
    # Get the current axes
    ax = plt.gca()
    ax.set_ylim([0, 0.2])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, "testing_performance"), eps=True)





def plot_training_loss(model_name):

    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/psth"

    create_dir(exp_path)
    # TODO this is in the incorrect order for plotting

    for env in env_dict:
        for speed in range(10):

            options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": speed, "delay_cond": 1}

            trial_data = _test(model_path, model_file, options, env=env_dict[env])
        
            # Get kinematics and activity in a center out setting
            # On random and delay
            colors = plt.cm.rainbow(np.linspace(0, 1, trial_data["h"].shape[0])) 

            delay = trial_data["epoch_bounds"]["delay"][0]
            mov = trial_data["epoch_bounds"]["movement"][0]
            hold = trial_data["epoch_bounds"]["hold"][0]

            for i, h in enumerate(trial_data["h"]):
                for unit in range(h.shape[-1]):
                    plt.plot(h[delay:, unit], color=colors[i], linewidth=4, alpha=0.5)
                    plt.axvline(mov-delay, linestyle="dashed", color="grey")
                    plt.axvline(hold-delay, linestyle="dashed", color="grey")
                # Access current axes and hide top/right spines
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                save_fig(os.path.join(exp_path, f"unit_{unit}", f"{env}_speed{speed}_tg_trajectory.png"))




def plot_transfer_losses():
    
    model_names = [
        "rnn256_softplus_heldout_transfer",
        "rnn256_softplus_heldout_sin_transfer",
        "rnn256_softplus_heldout_reach_transfer",
        "rnn256_softplus_heldout_nosin_transfer",
        "rnn256_softplus_heldout_nocr_transfer",
        "rnn256_softplus_heldout_cr_transfer"
    ]

    labels = [
        "all",
        "sin",
        "reach",
        "nosin",
        "nocr",
        "cr"
    ]

    fig, ax = standard_2d_ax()
    colors = plt.cm.Set2(np.linspace(0, 1, 6)) 
    for c, model in enumerate(model_names):
        model_path = f"checkpoints/{model}"
        losses = np.loadtxt(os.path.join(model_path, "losses.txt"))
        avg_losses = []
        for i in range(0, len(losses)-100, 100):
            avg_losses.append(sum(losses[i:i+100])/100)
        ax.plot(avg_losses, color=colors[c], linewidth=4, label=labels[c])
    #ax.legend()
    save_fig("results/across_model_performance/transfer_losses", eps=True)





if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "plot_test_performance":
        plot_test_performance(args.model_name) 
    elif args.experiment == "plot_test_performance_held_out":
        plot_test_performance_held_out(args.model_name) 
    elif args.experiment == "plot_training_loss":
        plot_single_unit_psth(args.model_name) 
    elif args.experiment == "plot_transfer_losses":
        plot_transfer_losses() 
    else:
        raise ValueError("Experiment not in this file")