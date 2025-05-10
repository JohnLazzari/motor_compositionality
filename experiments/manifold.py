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
    exp_path = f"results/{model_name}/pc_angles"

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

    plt.figure(figsize=(4, 4))

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

    angles_dict, control_array = principal_angles(combinations, combination_labels, mode)

    # Take mean of each angle in control
    mean_control = np.mean(control_array, axis=0)
    print(mean_control)
    
    for angles in angles_dict:
        plt.plot(x, angles_dict[angles], linewidth=4, marker='o', markersize=15, alpha=0.5)
    plt.plot(x, mean_control, linewidth=4, linestyle="dashed", color="grey")

    # Access current axes and hide top/right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, f"{system}_{comparison}_principal_angles"), eps=True)




def _vaf_ratio(model_name, system, comparison):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pc_angles"
    hp = load_hp(model_path)

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

    plt.figure(figsize=(4, 4))

    if system == "neural":
        mode = "h"
    elif system == "muscle":
        mode = "muscle_acts"
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
        for env in env_dict:
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            combinations.append((
                trial_data["h"][:, trial_data["epoch_bounds"]["delay"][0]:trial_data["epoch_bounds"]["delay"][1]], 
                trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            ))

    vaf_ratio_list, vaf_ratio_control = vaf_ratio(combinations, mode)
    
    bins = np.linspace(0, 1, 15)
    weights_data = np.ones_like(vaf_ratio_list) / len(vaf_ratio_list)
    control_mean = sum(vaf_ratio_control) / len(vaf_ratio_control)
    plt.hist(vaf_ratio_list, bins=bins, weights=weights_data, color="purple")
    plt.axvline(control_mean, color="grey", linestyle="dashed")
    plt.xlim(0, 1)

    # Access current axes and hide top/right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, f"{system}_{comparison}_vaf_ratio"), eps=True)




def muscle_synergies(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pc_angles"
    hp = load_hp(model_path)

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

    plt.figure(figsize=(4, 4))

    trial_data_envs = []
    for env in env_dict:
        trial_data = _test(model_path, model_file, options, env=env_dict[env])
        trial_data_envs.append(trial_data["muscle_acts"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]])
    trial_data_envs_cat = torch.concatenate(trial_data_envs, dim=1).reshape((-1, hp["hid_size"]))

    all_task_pca = PCA()
    all_task_pca.fit(trial_data_envs_cat)

    colors = plt.cm.inferno(np.linspace(0, 1, len(trial_data_envs)))
    comps = 6

    plt.rc('figure', figsize=(3, 6))
    # Create a figure
    fig = plt.figure()
    # Add a 3D subplot
    ax = fig.add_subplot(111)

    for e, env_data in enumerate(trial_data_envs):
        # transform
        cond_proj = all_task_pca.transform(env_data.reshape((-1, hp["hid_size"])))
        total_var = env_data.reshape((-1, hp["hid_size"])).var(dim=0).sum()
        ax.plot(
            np.arange(1, comps+1),
            [cond_proj[:, :i].var(axis=0).sum() / total_var for i in range(comps)], 
            marker='o', color=colors[e], alpha=0.5, linewidth=4, markersize=15)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, f"muscles_var_explained"), eps=True)




def neural_principal_angles_task(model_name):
    _principal_angles(model_name, "neural", "task")
def neural_principal_angles_epoch(model_name):
    _principal_angles(model_name, "neural", "epoch")
def muscle_principal_angles_task(model_name):
    _principal_angles(model_name, "muscle", "task")
def muscle_principal_angles_epoch(model_name):
    _principal_angles(model_name, "muscle", "epoch")




def neural_vaf_ratio_task(model_name):
    _vaf_ratio(model_name, "neural", "task")
def neural_vaf_ratio_epoch(model_name):
    _vaf_ratio(model_name, "neural", "epoch")
def muscle_vaf_ratio_task(model_name):
    _vaf_ratio(model_name, "muscle", "task")
def muscle_vaf_ratio_epoch(model_name):
    _vaf_ratio(model_name, "muscle", "epoch")






if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    # Principle Angles
    if args.experiment == "neural_principal_angles_task":
        neural_principal_angles_task(args.model_name) 
    elif args.experiment == "muscle_principal_angles_task":
        muscle_principal_angles_task(args.model_name) 
    elif args.experiment == "neural_principal_angles_epoch":
        neural_principal_angles_epoch(args.model_name) 
    elif args.experiment == "muscle_principal_angles_epoch":
        muscle_principal_angles_epoch(args.model_name) 

    # VAF
    elif args.experiment == "neural_vaf_ratio_task":
        neural_vaf_ratio_task(args.model_name) 
    elif args.experiment == "muscle_vaf_ratio_task":
        muscle_vaf_ratio_task(args.model_name) 
    elif args.experiment == "neural_vaf_ratio_epoch":
        neural_vaf_ratio_epoch(args.model_name) 
    elif args.experiment == "muscle_vaf_ratio_epoch":
        muscle_vaf_ratio_epoch(args.model_name) 
    
    # dPCA
    elif args.experiment == "plot_dpca":
        plot_dpca(args.model_name) 

    elif args.experiment == "muscle_synergies":
        muscle_synergies(args.model_name) 

    else:
        raise ValueError("Experiment not in this file")