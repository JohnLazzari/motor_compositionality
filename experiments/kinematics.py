import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp, interpolate_trial

import warnings
warnings.filterwarnings("ignore")

from train import train_2link
import motornet as mn
from model import RNNPolicy, GRUPolicy, OrthogonalNet
import torch
import os
from utils import load_hp, create_dir, save_fig, load_pickle, interpolate_trial, random_orthonormal_basis
from envs import DlyHalfReach, DlyHalfCircleClk, DlyHalfCircleCClk, DlySinusoid, DlySinusoidInv
from envs import DlyFullReach, DlyFullCircleClk, DlyFullCircleCClk, DlyFigure8, DlyFigure8Inv, ComposableEnv
import matplotlib.pyplot as plt
import numpy as np
import config
from analysis.clustering import Analysis
import pickle
from analysis.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
import analysis.plot_utils as plot_utils
from analysis.manifold import principal_angles, vaf_ratio
import dPCA
from dPCA import dPCA
import tqdm as tqdm
import itertools
from sklearn.decomposition import PCA
from losses import l1_dist
import scipy
from mRNNTorch.analysis import flow_field
import matplotlib.patches as mpatches
from exp_utils import _test, env_dict
from itertools import product

plt.rcParams.update({'font.size': 18})  # Sets default font size for all text

def plot_task_kinematics(model_name):
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
    exp_path = f"results/{model_name}/kinematics"

    plt.rc('figure', figsize=(4, 4))

    for env in env_dict:
        for speed in range(10):

            options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": speed, "delay_cond": 1}

            trial_data = _test(model_path, model_file, options, env=env_dict[env])
        
            # Get kinematics and activity in a center out setting
            # On random and delay
            colors = plt.cm.inferno(np.linspace(0, 1, trial_data["xy"].shape[0])) 

            for i, (tg, xy) in enumerate(zip(trial_data["tg"], trial_data["xy"])):
                plt.plot(xy[:, 0], xy[:, 1], linewidth=4, color=colors[i], alpha=0.75)
                plt.scatter(xy[0, 0], xy[0, 1], s=150, marker='x', color=colors[i])
                plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker='^', color=colors[i])

            # Access current axes and hide top/right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            save_fig(os.path.join(exp_path, "scatter", f"{env}_speed{speed}_kinematics"), eps=True)

            # Plot x coordinate only 
            for i, xy in enumerate(trial_data["xy"]):
                plt.plot(xy[:, 0], color=colors[i])

            # Access current axes and hide top/right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            save_fig(os.path.join(exp_path, "xpos", f"{env}_speed{speed}_xpos"))

            # Plot y coordinate only 
            for i, xy in enumerate(trial_data["xy"]):
                plt.plot(xy[:, 1], color=colors[i])

            # Access current axes and hide top/right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            save_fig(os.path.join(exp_path, "ypos", f"{env}_speed{speed}_ypos"))






def plot_task_kinematics_held_out_transfer(model_name):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    transfer_model_path = f"checkpoints/{model_name}"
    transfer_model_file = f"{model_name}.pth"

    held_out_model_path = f"checkpoints/rnn256_softplus_heldout"
    held_out_model_file = f"rnn256_softplus_heldout.pth"

    exp_path = f"results/{model_name}/kinematics/held_out_transfer"

    env_dict = {
        "DlyHalfCircleCClk": DlyHalfCircleCClk,
        "DlyFullCircleCClk": DlyFullCircleCClk
    }

    plt.rc('figure', figsize=(4, 4))

    for env in env_dict:

        options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": 5, "delay_cond": 1}

        transfer_trial_data = _test(transfer_model_path, transfer_model_file, options, env=env_dict[env], add_new_rule_inputs=True)
        held_out_trial_data = _test(held_out_model_path, held_out_model_file, options, env=env_dict[env])
    
        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, transfer_trial_data["xy"].shape[0])) 

        for i, (tg, xy) in enumerate(zip(transfer_trial_data["tg"], transfer_trial_data["xy"])):
            plt.plot(xy[:, 0], xy[:, 1], linewidth=4, color=colors[i], alpha=0.75)
            plt.scatter(xy[0, 0], xy[0, 1], s=150, marker='x', color=colors[i])
            plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker='^', color=colors[i])

        # Access current axes and hide top/right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        save_fig(os.path.join(exp_path, "scatter", f"{env}_kinematics"), eps=True)

        for i, (tg, xy) in enumerate(zip(held_out_trial_data["tg"], held_out_trial_data["xy"])):
            plt.plot(xy[:, 0], xy[:, 1], linewidth=4, color=colors[i], linestyle="dashed", alpha=0.5)

        # Access current axes and hide top/right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        save_fig(os.path.join(exp_path, "scatter", f"{env}_before_kinematics"), eps=True)



def plot_task_kinematics_compositional_env(model_name):
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
    exp_path = f"results/{model_name}/kinematics"

    plt.rc('figure', figsize=(4, 4))

    forward_motifs = [
        "forward_halfreach",
        "forward_halfcircleclk",
        "forward_halfcirclecclk",
        "forward_sinusoid",
        "forward_sinusoidinv"
    ]

    backward_motifs = [
        "backward_fullreach",
        "backward_fullcircleclk",
        "backward_fullcirclecclk",
        "backward_figure8",
        "backward_figure8inv"
    ]

    combination_idx = list(product(forward_motifs, backward_motifs))
    combination_idx.remove(("forward_halfreach", "backward_fullreach"))
    combination_idx.remove(("forward_halfcircleclk", "backward_fullcircleclk"))
    combination_idx.remove(("forward_halfcirclecclk", "backward_fullcirclecclk"))
    combination_idx.remove(("forward_sinusoid", "backward_figure8"))
    combination_idx.remove(("forward_sinusoidinv", "backward_figure8inv"))

    for combination in combination_idx:
        for speed in range(10):

            options = {
                "batch_size": 8, 
                "reach_conds": torch.arange(0, 32, 4), 
                "speed_cond": 5, 
                "forward_key": combination[0], 
                "backward_key": combination[1], 
            }

            trial_data = _test(model_path, model_file, options, env=ComposableEnv, add_new_rule_inputs=True)
        
            # Get kinematics and activity in a center out setting
            # On random and delay
            colors = plt.cm.inferno(np.linspace(0, 1, trial_data["xy"].shape[0])) 

            for i, (tg, xy) in enumerate(zip(trial_data["tg"], trial_data["xy"])):
                plt.plot(xy[:, 0], xy[:, 1], linewidth=4, color=colors[i], alpha=0.75)
                plt.scatter(xy[0, 0], xy[0, 1], s=150, marker='x', color=colors[i])
                plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker='^', color=colors[i])

            # Access current axes and hide top/right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            save_fig(os.path.join(exp_path, "scatter", f"{combination}_speed{speed}_kinematics"), eps=True)

            # Plot x coordinate only 
            for i, xy in enumerate(trial_data["xy"]):
                plt.plot(xy[:, 0], color=colors[i])

            # Access current axes and hide top/right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            save_fig(os.path.join(exp_path, "xpos", f"{combination}_speed{speed}_xpos"))

            # Plot y coordinate only 
            for i, xy in enumerate(trial_data["xy"]):
                plt.plot(xy[:, 1], color=colors[i])

            # Access current axes and hide top/right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            save_fig(os.path.join(exp_path, "ypos", f"{combination}_speed{speed}_ypos"))






def plot_speed_kinematics(model_name):
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
    exp_path = f"results/{model_name}/kinematics"

    plt.rc('figure', figsize=(8, 4))

    colors = plt.cm.Reds(np.linspace(0, 1, 10)) 

    for e, env in enumerate(env_dict):

        for speed in range(10):

            options = {"batch_size": 16, "reach_conds": torch.arange(0, 32, 2), "speed_cond": speed, "delay_cond": 1}
            trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=True)

            start = trial_data["epoch_bounds"]["movement"][0]
            end = trial_data["epoch_bounds"]["movement"][1]

            x_pos = trial_data["xy"][0, start:end, 0]
            y_pos = trial_data["xy"][0, start:end, 1]
    
            # x pos
            plt.plot(x_pos, linewidth=4, color=colors[speed])
            if e < 5:
                plt.xlim([0, 150])
            else:
                plt.xlim([0, 300])
            ax = plt.gca()
            lines = ax.get_lines()
            # Remove everything
            ax.axis('off')
            save_fig(os.path.join(exp_path, "speeds", "speed_kinematics", f"{env}_x_kinematics_speed_{speed}"), eps=True)

            # y pos
            plt.plot(y_pos, linewidth=4, color=colors[speed])
            if e < 5:
                plt.xlim([0, 150])
            else:
                plt.xlim([0, 300])
            ax = plt.gca()
            lines = ax.get_lines()
            # Remove everything
            ax.axis('off')
            save_fig(os.path.join(exp_path, "speeds", "speed_kinematics", f"{env}_y_kinematics_speed_{speed}"), eps=True)






def plot_speed_modulation(model_name):
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
    exp_path = f"results/{model_name}/kinematics"

    plt.rc('figure', figsize=(4, 4))

    speed_vels_mean_dict_x = {}
    speed_vels_mean_dict_y = {}

    speed_vels_std_dict_x = {}
    speed_vels_std_dict_y = {}

    speed_scalars_dict = {}

    for env in env_dict:

        speed_vels_x = []
        speed_vels_y = []

        speed_stds_x = []
        speed_stds_y = []

        speed_scalars = []
        for speed in range(10):

            options = {"batch_size": 16, "reach_conds": torch.arange(0, 32, 2), "speed_cond": speed, "delay_cond": 1}
            trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=True)

            start = trial_data["epoch_bounds"]["movement"][0]
            end = trial_data["epoch_bounds"]["movement"][1]

            speed_vels_x.append(np.abs((trial_data["xy"][:, start+1:end, 0] - trial_data["xy"][:, start:end-1, 0]) / 0.01).mean())
            speed_vels_y.append(np.abs((trial_data["xy"][:, start+1:end, 1] - trial_data["xy"][:, start:end-1, 1]) / 0.01).mean())

            speed_stds_x.append(np.abs((trial_data["xy"][:, start+1:end, 0] - trial_data["xy"][:, start:end-1, 0]) / 0.01).std())
            speed_stds_y.append(np.abs((trial_data["xy"][:, start+1:end, 1] - trial_data["xy"][:, start:end-1, 1]) / 0.01).std())

            # Getting it from one timepoint of one condition, should be the same across conditions and epochs (except stability)
            speed_scalars.append(trial_data["obs"][0, 100, 10:11].item())
        
        speed_vels_mean_dict_x[env] = np.array(speed_vels_x)
        speed_vels_mean_dict_y[env] = np.array(speed_vels_y)

        speed_vels_std_dict_x[env] = np.array(speed_stds_x)
        speed_vels_std_dict_y[env] = np.array(speed_stds_y)

        speed_scalars_dict[env] = speed_scalars
        
    for (vel, std, scalars) in zip(speed_vels_mean_dict_x, speed_vels_std_dict_x, speed_scalars_dict):
        plt.plot(speed_scalars_dict[scalars], speed_vels_mean_dict_x[vel], linewidth=4)
        ax = plt.gca()
        lines = ax.get_lines()
        plt.fill_between(speed_scalars_dict[scalars], 
            speed_vels_mean_dict_x[vel] - speed_vels_std_dict_x[std], 
            speed_vels_mean_dict_x[vel] + speed_vels_std_dict_x[std], 
            alpha=0.1, color=lines[-1].get_color()
        )

    plt.xlabel("Speed Input")
    plt.ylabel("Kinematic Speed")

    # Access current axes and hide top/right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, "speeds", f"speed_modulation_x"), eps=True)

    for (vel, std, scalars) in zip(speed_vels_mean_dict_y, speed_vels_std_dict_y, speed_scalars_dict):
        plt.plot(speed_scalars_dict[scalars], speed_vels_mean_dict_y[vel], linewidth=4)
        ax = plt.gca()
        lines = ax.get_lines()
        plt.fill_between(speed_scalars_dict[scalars], 
            speed_vels_mean_dict_y[vel] - speed_vels_std_dict_y[std], 
            speed_vels_mean_dict_y[vel] + speed_vels_std_dict_y[std], 
            alpha=0.1, color=lines[-1].get_color()
        )

    plt.xlabel("Speed Input Value")
    plt.ylabel("Kinematic Velocity")

    # Access current axes and hide top/right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, "speeds", f"speed_modulation_y"), eps=True)




if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "plot_task_kinematics":
        plot_task_kinematics(args.model_name) 
    elif args.experiment == "plot_speed_kinematics":
        plot_speed_kinematics(args.model_name) 
    elif args.experiment == "plot_task_kinematics_compositional_env":
        plot_task_kinematics_compositional_env(args.model_name) 
    elif args.experiment == "plot_task_kinematics_held_out_transfer":
        plot_task_kinematics_held_out_transfer(args.model_name) 
    elif args.experiment == "plot_speed_modulation":
        plot_speed_modulation(args.model_name) 
    else:
        raise ValueError("Experiment not in this file")