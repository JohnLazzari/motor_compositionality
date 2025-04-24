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

            save_fig(os.path.join(exp_path, f"{env}_input_orientation{batch}"))




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

    options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4)}

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env])
    
        for i, inp in enumerate(trial_data["obs"]):

            fig, ax = plt.subplots(7, 1)
            fig.set_size_inches(3, 6)
            plt.rc('font', size=6)

            ax[0].imshow(inp[:, :10].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            # Remove top and right only (common for minimalist style)
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['bottom'].set_visible(False)
            ax[0].set_xticks([])
            ax[0].set_title("Rule Input")

            ax[1].plot(inp[:, 10:11], color="blue")
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[1].set_xticks([])
            ax[1].set_title("Speed Scalar")

            ax[2].plot(inp[:, 11:12], color="blue")
            ax[2].spines['top'].set_visible(False)
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['bottom'].set_visible(False)
            ax[2].set_xticks([])
            ax[2].set_title("Go Cue")

            ax[3].imshow(inp[:, 12:14].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[3].spines['top'].set_visible(False)
            ax[3].spines['right'].set_visible(False)
            ax[3].spines['bottom'].set_visible(False)
            ax[3].set_xticks([])
            ax[3].set_title("Target Position")

            ax[4].imshow(inp[:, 14:16].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[4].spines['top'].set_visible(False)
            ax[4].spines['right'].set_visible(False)
            ax[4].spines['bottom'].set_visible(False)
            ax[4].set_xticks([])
            ax[4].set_title("Fingertip")

            ax[5].imshow(inp[:, 16:22].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[5].spines['top'].set_visible(False)
            ax[5].spines['right'].set_visible(False)
            ax[5].spines['bottom'].set_visible(False)
            ax[5].set_xticks([])
            ax[5].set_title("Muscle Length")
            
            ax[6].imshow(inp[:, 22:28].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[6].spines['top'].set_visible(False)
            ax[6].spines['right'].set_visible(False)
            ax[6].spines['bottom'].set_visible(False)
            ax[6].set_title("Muscle Velocity")

            save_fig(os.path.join(exp_path, f"{env}_input_orientation{i}"))




if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "plot_task_trajectories":
        plot_task_trajectories(args.model_name) 
    elif args.experiment == "plot_task_input_output":
        plot_task_input_output(args.model_name) 
    elif args.experiment == "plot_task_feedback":
        plot_task_feedback(args.model_name) 
    else:
        raise ValueError("Experiment not in this file")