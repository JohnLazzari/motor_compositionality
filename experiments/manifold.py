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

plt.rcParams.update({'font.size': 18})  # Sets default font size for all text

def _principal_angles(model_name, system, comparison):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

    if system == "neural":
        mode = "h"
        x = np.arange(1, 13)
    elif system == "muscle":
        mode = "muscle_acts"
        x = np.arange(1, 4)
    else:
        raise ValueError("Not a valid system")

    if comparison == "task":

        trial_data_mode = {}
        combinations = []
        for env in env_dict:
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            trial_data_mode[env] = trial_data[mode][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
        # Get all unique pairs of unit activity across tasks
        combination_labels = list(itertools.combinations(trial_data_mode, 2))
        for combination_label in combination_labels:
            combinations.append((
                trial_data_mode[combination_label[0]],
                trial_data_mode[combination_label[1]]
            ))

    elif comparison == "epoch":

        combinations = []
        combination_labels = []
        for env in env_dict:
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            combination_labels.append(env)
            combinations.append((
                trial_data[mode][:, trial_data["epoch_bounds"]["delay"][0]:trial_data["epoch_bounds"]["delay"][1]], 
                trial_data[mode][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            ))

    elif comparison == "condition":

        options = {"batch_size": 4, "reach_conds": torch.arange(0, 32, 8), "delay_cond": 0, "speed_cond": 5}

        combinations = []
        combination_labels = list(itertools.combinations([0, 1, 2, 3], 2))
        for env in env_dict:
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            movement_act = trial_data[mode][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            movement_act = [act for act in movement_act]
            # Get all unique pairs of unit activity across tasks
            for combination_label in combination_labels:
                combinations.append((
                    movement_act[combination_label[0]],
                    movement_act[combination_label[1]]
                ))

    angles_list, control_array = principal_angles(combinations, combination_labels, mode)

    return x, angles_list, control_array




def _plot_principal_angles(angles_dict, control_array, x, color="blue", alpha=0.75, control_color="grey"):


    # Take mean of each angle in control
    mean_control = np.mean(control_array, axis=0)
    
    for angles in angles_dict:
        plt.plot(x, angles, linewidth=4, alpha=alpha, color=color)
    plt.plot(x, mean_control, linewidth=2, linestyle="dashed", color=control_color)

    # Access current axes and hide top/right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)




def _vaf_ratio(model_name, system, comparison):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    hp = load_hp(model_path)

    if system == "neural":
        mode = "h"
    elif system == "muscle":
        mode = "muscle_acts"
    else:
        raise ValueError("Not a valid system")

    if comparison == "task":

        options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

        trial_data_mode = {}
        combinations = []
        for env in env_dict:
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            trial_data_mode[env] = trial_data[mode][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
        # Get all unique pairs of unit activity across tasks
        combination_labels = list(itertools.combinations(trial_data_mode, 2))
        for combination_label in combination_labels:
            combinations.append((
                trial_data_mode[combination_label[0]],
                trial_data_mode[combination_label[1]]
            ))

    elif comparison == "epoch":

        options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

        combinations = []
        for env in env_dict:
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            combinations.append((
                trial_data["h"][:, trial_data["epoch_bounds"]["delay"][0]:trial_data["epoch_bounds"]["delay"][1]], 
                trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            ))
    
    elif comparison == "condition":

        options = {"batch_size": 4, "reach_conds": torch.arange(0, 32, 8), "delay_cond": 0, "speed_cond": 5}

        combinations = []
        combination_labels = list(itertools.combinations([0, 1, 2, 3], 2))
        for env in env_dict:
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            movement_act = trial_data[mode][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            movement_act = [act for act in movement_act]
            # Get all unique pairs of unit activity across tasks
            for combination_label in combination_labels:
                combinations.append((
                    movement_act[combination_label[0]],
                    movement_act[combination_label[1]]
                ))
    
    vaf_ratio_list, vaf_ratio_control = vaf_ratio(combinations, mode)

    return vaf_ratio_list, vaf_ratio_control




def _plot_vaf_ratio(vaf_ratio_list, vaf_ratio_control, color="purple", control_color="grey"):
    

    bins = np.linspace(0, 1, 15)
    weights_data = np.ones_like(vaf_ratio_list) / len(vaf_ratio_list)
    control_mean = sum(vaf_ratio_control) / len(vaf_ratio_control)
    plt.hist(vaf_ratio_list, bins=bins, weights=weights_data, color=color, alpha=0.75)
    plt.axvline(control_mean, color=control_color, linestyle="dashed")
    plt.xlim(0, 1)

    # Access current axes and hide top/right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)





def neural_principal_angles_task_cond(model_name):
    exp_path = f"results/{model_name}/pc_angles"
    x, angles_dict, control_array = _principal_angles(model_name, "neural", "task")
    x, angles_dict_cond, control_array_cond = _principal_angles(model_name, "neural", "condition")
    plt.figure(figsize=(4, 4))
    _plot_principal_angles(angles_dict_cond, control_array_cond, x, color="skyblue", alpha=0.5)
    _plot_principal_angles(angles_dict, control_array, x, color="blue", alpha=0.25)
    save_fig(os.path.join(exp_path, f"neural_task_cond_principal_angles"), eps=True)

def neural_principal_angles_epoch(model_name):
    exp_path = f"results/{model_name}/pc_angles"
    x, angles_dict, control_array = _principal_angles(model_name, "neural", "epoch")
    plt.figure(figsize=(4, 4))
    _plot_principal_angles(angles_dict, control_array, x, color="blue", alpha=0.75)
    save_fig(os.path.join(exp_path, f"neural_epoch_principal_angles"), eps=True)

def muscle_principal_angles_task_cond(model_name):
    exp_path = f"results/{model_name}/pc_angles"
    x, angles_dict, control_array = _principal_angles(model_name, "muscle", "task")
    x, angles_dict_cond, control_array_cond = _principal_angles(model_name, "muscle", "condition")
    plt.figure(figsize=(4, 4))
    _plot_principal_angles(angles_dict_cond, control_array_cond, x, color="skyblue", alpha=0.5)
    _plot_principal_angles(angles_dict, control_array, x, color="blue", alpha=0.25)
    save_fig(os.path.join(exp_path, f"muscle_task_cond_principal_angles"), eps=True)




def neural_vaf_ratio_task_cond(model_name):
    exp_path = f"results/{model_name}/pc_angles"
    vaf_ratio_list, vaf_ratio_control = _vaf_ratio(model_name, "neural", "task")
    vaf_ratio_list_cond, vaf_ratio_control_cond = _vaf_ratio(model_name, "neural", "condition")
    plt.figure(figsize=(4, 4))
    _plot_vaf_ratio(vaf_ratio_list, vaf_ratio_control, color="purple")
    _plot_vaf_ratio(vaf_ratio_list_cond, vaf_ratio_control_cond, color="violet")
    save_fig(os.path.join(exp_path, f"neural_task_cond_vaf_ratio"), eps=True)

def neural_vaf_ratio_epoch(model_name):
    exp_path = f"results/{model_name}/pc_angles"
    vaf_ratio_list, vaf_ratio_control = _vaf_ratio(model_name, "neural", "epoch")
    plt.figure(figsize=(4, 4))
    _plot_vaf_ratio(vaf_ratio_list, vaf_ratio_control, color="purple")
    save_fig(os.path.join(exp_path, f"neural_epoch_vaf_ratio"), eps=True)

def muscle_vaf_ratio_task_cond(model_name):
    exp_path = f"results/{model_name}/pc_angles"
    vaf_ratio_list, vaf_ratio_control = _vaf_ratio(model_name, "muscle", "task")
    vaf_ratio_list_cond, vaf_ratio_control_cond = _vaf_ratio(model_name, "muscle", "condition")
    plt.figure(figsize=(4, 4))
    _plot_vaf_ratio(vaf_ratio_list, vaf_ratio_control, color="purple")
    _plot_vaf_ratio(vaf_ratio_list_cond, vaf_ratio_control_cond, color="violet")
    save_fig(os.path.join(exp_path, f"muscle_task_cond_vaf_ratio"), eps=True)






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
    
    # dPCA
    elif args.experiment == "plot_dpca":
        plot_dpca(args.model_name) 

    elif args.experiment == "muscle_synergies":
        muscle_synergies(args.model_name) 

    else:
        raise ValueError("Experiment not in this file")