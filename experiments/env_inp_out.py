import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp, interpolate_trial

import warnings
warnings.filterwarnings("ignore")

from train import train_2link
import motornet as mn
from model import RNNPolicy, GRUPolicy
import torch
import os
from utils import load_hp, create_dir, save_fig, load_pickle, interpolate_trial, random_orthonormal_basis
from envs import DlyHalfReach, DlyHalfCircleClk, DlyHalfCircleCClk, DlySinusoid, DlySinusoidInv
from envs import DlyFullReach, DlyFullCircleClk, DlyFullCircleCClk, DlyFigure8, DlyFigure8Inv
from envs import ComposableEnv
from cog_envs import Go
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
import matplotlib as mpl
from exp_utils import _test, env_dict, cog_env_dict 
from itertools import product

plt.rcParams.update({'font.size': 18})  # Sets default font size for all text

def plot_task_trajectories(model_name):
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
    exp_path = f"results/trajectories"

    create_dir(exp_path)

    for env in env_dict:
        for speed in range(10):

            options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": speed}

            effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
            cur_env = env_dict[env](effector=effector)
            
            obs, info = cur_env.reset(testing=True, options=options)

            # Get kinematics and activity in a center out setting
            # On random and delay
            colors = plt.cm.inferno(np.linspace(0, 1, cur_env.traj.shape[1])) 

            for i, tg in enumerate(cur_env.traj):
                plt.scatter(tg[:, 0], tg[:, 1], s=10, color=colors)
                plt.scatter(tg[0, 0], tg[0, 1], s=150, marker='x', color="black")
                plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker='^', color="black")
            save_fig(os.path.join(exp_path, f"{env}_speed{speed}_tg_trajectory.png"))




def plot_task_trajectories_compositional_env(model_name):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    exp_path = f"results/trajectories"

    create_dir(exp_path)

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

        options = {
            "batch_size": 8, 
            "reach_conds": torch.arange(0, 32, 4), 
            "speed_cond": 5, 
            "forward_key": combination[0], 
            "backward_key": combination[1], 
        }

        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        cur_env = ComposableEnv(effector=effector)
        
        obs, info = cur_env.reset(testing=True, options=options)
        print(cur_env.traj.shape)

        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, cur_env.traj.shape[1])) 

        for i, tg in enumerate(cur_env.traj):
            plt.scatter(tg[:, 0], tg[:, 1], s=10, color=colors)
            plt.scatter(tg[0, 0], tg[0, 1], s=150, marker='x', color="black")
            plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker='^', color="black")
        save_fig(os.path.join(exp_path, "compositional_env", f"{combination[0]}_{combination[1]}_tg_trajectory"))



def plot_task_input_output(model_name):
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
    exp_path = f"results/input"

    create_dir(exp_path)

    for env in env_dict:

        options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": 5, "delay_cond": 0}

        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        cur_env = env_dict[env](effector=effector)
        
        obs, info = cur_env.reset(testing=True, options=options)

        for batch in range(options["batch_size"]):

            fig, ax = plt.subplots(5, 1)
            fig.set_size_inches(3, 6)
            plt.rc('font', size=6)

            ax[0].imshow(cur_env.rule_input[batch].unsqueeze(0).repeat(cur_env.max_ep_duration, 1).T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            
            # Remove top and right only (common for minimalist style)
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['bottom'].set_visible(False)
            ax[0].set_xticks([])
            ax[0].set_title("Rule Input")
            ax[0].axvline(cur_env.epoch_bounds["delay"][0], color="grey", linestyle="dashed")
            ax[0].axvline(cur_env.epoch_bounds["movement"][0], color="grey", linestyle="dashed")
            ax[0].axvline(cur_env.epoch_bounds["hold"][0], color="grey", linestyle="dashed")

            ax[1].plot(cur_env.speed_scalar[batch], color="blue")
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[1].set_xticks([])
            ax[1].set_title("Speed Scalar")
            ax[1].axvline(cur_env.epoch_bounds["delay"][0], color="grey", linestyle="dashed")
            ax[1].axvline(cur_env.epoch_bounds["movement"][0], color="grey", linestyle="dashed")
            ax[1].axvline(cur_env.epoch_bounds["hold"][0], color="grey", linestyle="dashed")

            ax[2].plot(cur_env.go_cue[batch], color="blue")
            ax[2].spines['top'].set_visible(False)
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['bottom'].set_visible(False)
            ax[2].set_xticks([])
            ax[2].set_title("Go Cue")
            ax[2].axvline(cur_env.epoch_bounds["delay"][0], color="grey", linestyle="dashed")
            ax[2].axvline(cur_env.epoch_bounds["movement"][0], color="grey", linestyle="dashed")
            ax[2].axvline(cur_env.epoch_bounds["hold"][0], color="grey", linestyle="dashed")

            ax[3].imshow(cur_env.vis_inp[batch].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[3].spines['top'].set_visible(False)
            ax[3].spines['right'].set_visible(False)
            ax[3].spines['bottom'].set_visible(False)
            ax[3].set_xticks([])
            ax[3].set_title("Visual Input")
            ax[3].axvline(cur_env.epoch_bounds["delay"][0], color="grey", linestyle="dashed")
            ax[3].axvline(cur_env.epoch_bounds["movement"][0], color="grey", linestyle="dashed")
            ax[3].axvline(cur_env.epoch_bounds["hold"][0], color="grey", linestyle="dashed")

            ax[4].imshow(cur_env.traj[batch].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[4].spines['top'].set_visible(False)
            ax[4].spines['right'].set_visible(False)
            ax[4].spines['bottom'].set_visible(False)
            ax[4].set_xticks([])
            ax[4].set_title("Tg Output (Only Movement Epoch)")

            save_fig(os.path.join(exp_path, f"{env}_input_orientation{batch}"), eps=True)


def plot_task_input_output_cog(model_name):
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
    exp_path = f"results/input"

    create_dir(exp_path)

    for env in cog_env_dict:

        options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": 5, "delay_cond": 0}

        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        cur_env = cog_env_dict[env](effector=effector)
        
        obs, info = cur_env.reset(testing=True, options=options)

        for batch in range(options["batch_size"]):

            fig, ax = plt.subplots(5, 1)
            fig.set_size_inches(3, 6)
            plt.rc('font', size=6)

            ax[0].imshow(cur_env.rule_input[batch].unsqueeze(0).repeat(cur_env.max_ep_duration, 1).T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            
            # Remove top and right only (common for minimalist style)
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['bottom'].set_visible(False)
            ax[0].set_xticks([])
            ax[0].set_title("Rule Input")
            ax[0].axvline(cur_env.epoch_bounds["delay"][0], color="grey", linestyle="dashed")
            ax[0].axvline(cur_env.epoch_bounds["movement"][0], color="grey", linestyle="dashed")
            ax[0].axvline(cur_env.epoch_bounds["hold"][0], color="grey", linestyle="dashed")

            ax[1].plot(cur_env.go_cue[batch], color="blue")
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[1].set_xticks([])
            ax[1].set_title("Go Cue")
            ax[1].axvline(cur_env.epoch_bounds["delay"][0], color="grey", linestyle="dashed")
            ax[1].axvline(cur_env.epoch_bounds["movement"][0], color="grey", linestyle="dashed")
            ax[1].axvline(cur_env.epoch_bounds["hold"][0], color="grey", linestyle="dashed")

            ax[2].imshow(cur_env.stim_1[batch].T, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax[2].spines['top'].set_visible(False)
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['bottom'].set_visible(False)
            ax[2].set_xticks([])
            ax[2].set_title("Stimulus 1")
            ax[2].axvline(cur_env.epoch_bounds["delay"][0], color="grey", linestyle="dashed")
            ax[2].axvline(cur_env.epoch_bounds["movement"][0], color="grey", linestyle="dashed")
            ax[2].axvline(cur_env.epoch_bounds["hold"][0], color="grey", linestyle="dashed")

            ax[3].imshow(cur_env.stim_2[batch].T, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax[3].spines['top'].set_visible(False)
            ax[3].spines['right'].set_visible(False)
            ax[3].spines['bottom'].set_visible(False)
            ax[3].set_xticks([])
            ax[3].set_title("Stimulus 2")
            ax[3].axvline(cur_env.epoch_bounds["delay"][0], color="grey", linestyle="dashed")
            ax[3].axvline(cur_env.epoch_bounds["movement"][0], color="grey", linestyle="dashed")
            ax[3].axvline(cur_env.epoch_bounds["hold"][0], color="grey", linestyle="dashed")

            ax[4].imshow(cur_env.traj[batch].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[4].spines['top'].set_visible(False)
            ax[4].spines['right'].set_visible(False)
            ax[4].spines['bottom'].set_visible(False)
            ax[4].set_xticks([])
            ax[4].set_title("Tg Output (Only Movement Epoch)")

            save_fig(os.path.join(exp_path, f"{env}_input_orientation{batch}"), eps=True)


def plot_task_feedback(model_name):
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
    exp_path = f"results/{model_name}/input"

    create_dir(exp_path)

    options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": 5, "delay_cond": 1}

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env_dict[env])

        for i, inp in enumerate(trial_data["obs"]):

            fig, ax = plt.subplots(9, 1,
                                    gridspec_kw={'height_ratios': [1, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 1, 1]})
            fig.set_size_inches(4, 8)
            plt.rc('font', size=10)

            colors = plt.cm.Set2(np.linspace(0, 1, 8)) 

            # Generate Gaussian noise with mean=0 and std=1, shaped [t, n]
            noise = np.random.normal(loc=0.0, scale=0.05, size=(inp.shape[0], 28))

            ax[0].imshow(inp[:, :10].T + noise[:, :10].T, cmap="Purples", aspect="auto")
            ax[0].axvline(25, linewidth=1, color="grey", linestyle="dashed")
            ax[0].axvline(75, linewidth=1, color="grey", linestyle="dashed")
            ax[0].axvline(175, linewidth=1, color="grey", linestyle="dashed")
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['bottom'].set_visible(False)
            ax[0].set_xticks([])
            ax[0].set_title("Rule Input")

            ax[1].plot(inp[:, 10:11], linewidth=4, color=colors[0])
            ax[1].plot(inp[:, 10:11]+noise[:, 10:11], linewidth=4, color=colors[0], alpha=0.25)
            ax[1].axvline(25, linewidth=1, color="grey", linestyle="dashed")
            ax[1].axvline(75, linewidth=1, color="grey", linestyle="dashed")
            ax[1].axvline(175, linewidth=1, color="grey", linestyle="dashed")
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[1].set_xticks([])
            ax[1].set_title("Speed Scalar")

            ax[2].plot(inp[:, 11:12], linewidth=4, color=colors[1])
            ax[2].plot(inp[:, 11:12]+noise[:, 11:12], linewidth=4, color=colors[1], alpha=0.25)
            ax[2].axvline(25, linewidth=1, color="grey", linestyle="dashed")
            ax[2].axvline(75, linewidth=1, color="grey", linestyle="dashed")
            ax[2].axvline(175, linewidth=1, color="grey", linestyle="dashed")
            ax[2].spines['top'].set_visible(False)
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['bottom'].set_visible(False)
            ax[2].set_xticks([])
            ax[2].set_title("Go Cue")

            ax[3].plot(inp[:, 12:13], linewidth=4, color=colors[2])
            ax[3].plot(inp[:, 12:13]+noise[:, 12:13], linewidth=4, color=colors[2], alpha=0.25)
            ax[3].axvline(25, linewidth=1, color="grey", linestyle="dashed")
            ax[3].axvline(75, linewidth=1, color="grey", linestyle="dashed")
            ax[3].axvline(175, linewidth=1, color="grey", linestyle="dashed")
            ax[3].spines['top'].set_visible(False)
            ax[3].spines['right'].set_visible(False)
            ax[3].spines['bottom'].set_visible(False)
            ax[3].set_xticks([])
            ax[3].set_title("Target x Position")

            ax[4].plot(inp[:, 13:14], linewidth=4, color=colors[3])
            ax[4].plot(inp[:, 13:14]+noise[:, 13:14], linewidth=4, color=colors[3], alpha=0.25)
            ax[4].axvline(25, linewidth=1, color="grey", linestyle="dashed")
            ax[4].axvline(75, linewidth=1, color="grey", linestyle="dashed")
            ax[4].axvline(175, linewidth=1, color="grey", linestyle="dashed")
            ax[4].spines['top'].set_visible(False)
            ax[4].spines['right'].set_visible(False)
            ax[4].spines['bottom'].set_visible(False)
            ax[4].set_xticks([])
            ax[4].set_title("Target y Position")

            ax[5].plot(inp[:, 14:15], linewidth=4, color=colors[4])
            ax[5].plot(inp[:, 14:15]+noise[:, 14:15], linewidth=4, color=colors[4], alpha=0.25)
            ax[5].axvline(25, linewidth=1, color="grey", linestyle="dashed")
            ax[5].axvline(75, linewidth=1, color="grey", linestyle="dashed")
            ax[5].axvline(175, linewidth=1, color="grey", linestyle="dashed")
            ax[5].spines['top'].set_visible(False)
            ax[5].spines['right'].set_visible(False)
            ax[5].spines['bottom'].set_visible(False)
            ax[5].set_xticks([])
            ax[5].set_title("Fingertip x Position")

            ax[6].plot(inp[:, 15:16], linewidth=4, color=colors[5])
            ax[6].plot(inp[:, 15:16]+noise[:, 15:16], linewidth=4, color=colors[5], alpha=0.25)
            ax[6].axvline(25, linewidth=1, color="grey", linestyle="dashed")
            ax[6].axvline(75, linewidth=1, color="grey", linestyle="dashed")
            ax[6].axvline(175, linewidth=1, color="grey", linestyle="dashed")
            ax[6].spines['top'].set_visible(False)
            ax[6].spines['right'].set_visible(False)
            ax[6].spines['bottom'].set_visible(False)
            ax[6].set_xticks([])
            ax[6].set_title("Fingertip y Position")

            ax[7].plot(inp[:, 16:22], linewidth=4, color=colors[6])
            ax[7].plot(inp[:, 16:22]+noise[:, 16:22], linewidth=4, color=colors[6], alpha=0.25)
            ax[7].axvline(25, linewidth=1, color="grey", linestyle="dashed")
            ax[7].axvline(75, linewidth=1, color="grey", linestyle="dashed")
            ax[7].axvline(175, linewidth=1, color="grey", linestyle="dashed")
            ax[7].spines['top'].set_visible(False)
            ax[7].spines['right'].set_visible(False)
            ax[7].spines['bottom'].set_visible(False)
            ax[7].set_xticks([])
            ax[7].set_title("Muscle Length")
            
            ax[8].plot(inp[:, 22:28], linewidth=4, color=colors[7])
            ax[8].plot(inp[:, 22:28]+noise[:, 22:28], linewidth=4, color=colors[7], alpha=0.25)
            ax[8].axvline(25, linewidth=1, color="grey", linestyle="dashed")
            ax[8].axvline(75, linewidth=1, color="grey", linestyle="dashed")
            ax[8].axvline(175, linewidth=1, color="grey", linestyle="dashed")
            ax[8].spines['top'].set_visible(False)
            ax[8].spines['right'].set_visible(False)
            ax[8].spines['bottom'].set_visible(False)
            ax[8].set_title("Muscle Velocity")
            ax[8].set_xlabel("Timesteps")

            save_fig(os.path.join(exp_path, f"{env}_input_orientation{i}"), eps=True)
    




if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "plot_task_trajectories":
        plot_task_trajectories(args.model_name) 
    elif args.experiment == "plot_task_trajectories_compositional_env":
        plot_task_trajectories_compositional_env(args.model_name) 
    elif args.experiment == "plot_task_input_output":
        plot_task_input_output(args.model_name) 
    elif args.experiment == "plot_task_feedback":
        plot_task_feedback(args.model_name) 
    else:
        raise ValueError("Experiment not in this file")