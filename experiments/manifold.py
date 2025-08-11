import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp, interpolate_trial

import warnings
warnings.filterwarnings("ignore")

from train import train_2link
import motornet as mn
from exp_utils import get_middle_movement
from model import RNNPolicy, GRUPolicy
import torch
import os
from utils import load_hp, create_dir, save_fig, load_pickle, interpolate_trial, random_orthonormal_basis
from plt_utils import standard_2d_ax
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
from sklearn.cross_decomposition import CCA
from losses import l1_dist
import scipy
from mRNNTorch.analysis import flow_field
import matplotlib.patches as mpatches
from exp_utils import _test, env_dict

plt.rcParams.update({'font.size': 18})  # Sets default font size for all text


def _principal_angles(model_name, system, comparison):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    hp = load_hp(model_path)

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only

    if system == "neural":
        mode = "h"
        x = np.arange(1, 13)
        num_units = hp["hid_size"]
    elif system == "muscle":
        mode = "muscle_acts"
        x = np.arange(1, 4)
        num_units = 6
    else:
        raise ValueError("Not a valid system")

    if comparison == "task":
        options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

        trial_data_mode = {}
        combinations = []
        for env in env_dict:
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            mov_beg = trial_data["epoch_bounds"]["movement"][0]
            mov_end = trial_data["epoch_bounds"]["movement"][1]
            movement_data = trial_data[mode][:, mov_beg:mov_end]
            trial_data_mode[env] = movement_data

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
    
    # Keeping this here for now in case I need it later, remove otherwise
    """
    options = {"batch_size": 4, "reach_conds": torch.arange(0, 32, 8), "delay_cond": 0, "speed_cond": 5}
    # Gather data for baseline
    for env in env_dict:
        trial_data = _test(model_path, model_file, options, env=env_dict[env])
        mov_beg = trial_data["epoch_bounds"]["movement"][0]
        mov_end = trial_data["epoch_bounds"]["movement"][1]
        movement_data = trial_data[mode][:, mov_beg:mov_end]
        if env == "DlyFullReach" or env == "DlyFullCircleClk" or env == "DlyFullCircleCClk" or env == "DlyFigure8" or env == "DlyFigure8Inv":
            baseline_data.append(movement_data)
        else:
            conds = []
            for cond in range(options["batch_size"]):
                conds.append(interpolate_trial(movement_data[cond], 200))
            movement_data = np.stack(conds, axis=0)
            baseline_data.append(movement_data)
        
    baseline_data = np.concatenate(baseline_data, axis=0)
    """
    baseline_data = []

    angles_list, control_array = principal_angles(combinations, baseline_data, mode, num_units)

    return x, angles_list, control_array




def _plot_principal_angles(angles_dict, control_array, x, color="blue", alpha=0.75, control_color="grey"):

    # Take mean of each angle in control
    mean_control = np.percentile(control_array, 0.1, axis=0, keepdims=False)
    
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
    num_units = hp["hid_size"]

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
    
    vaf_ratio_list, vaf_ratio_control = vaf_ratio(combinations, mode, num_units)

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




def _cca(task1, task2, system):

    if system == "h":
        n_components = 10
    elif system == "muscle_acts":
        n_components = 4

    task1_pca = PCA(n_components=n_components)
    task1_h = task1_pca.fit_transform(task1.reshape((-1, task1.shape[-1])))

    task2_pca = PCA(n_components=n_components)
    task2_h = task2_pca.fit_transform(task2.reshape((-1, task2.shape[-1])))

    cca = CCA(n_components=n_components)
    cca.fit(task1_h, task2_h)
    X_c, Y_c = cca.transform(task1_h, task2_h)

    return X_c, Y_c, cca




def _task_ccs(model_name, system):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 2, "speed_cond": 5}

    trial_data_mode = {}
    combinations = []
    for env in env_dict:
        trial_data = _test(model_path, model_file, options, env=env_dict[env])
        trial_data_mode[env] = trial_data[system][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
    
    extensions = [
        "DlyHalfReach",
        "DlyHalfCircleClk",
        "DlyHalfCircleCClk",
        "DlySinusoid",
        "DlySinusoidInv"
    ]

    retractions = [
        "DlyFullReach",
        "DlyFullCircleClk",
        "DlyFullCircleCClk",
        "DlyFigure8",
        "DlyFigure8Inv"
    ]

    extension_combinations = list(itertools.combinations(extensions, 2))
    retraction_combinations = list(itertools.combinations(retractions, 2))
    combination_labels = [*extension_combinations, *retraction_combinations]

    # Get all unique pairs of unit activity across tasks
    for combination_label in combination_labels:
        combinations.append((
            trial_data_mode[combination_label[0]],
            trial_data_mode[combination_label[1]]
        ))
    
    all_ccs = []
    for combination in combinations:
        X_c, Y_c, _ = _cca(combination[0], combination[1], system)
        corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[-1])]
        all_ccs.append(corrs)
    
    return all_ccs




def plot_network_vs_muscle_ccs(model_name):

    exp_path = f"results/{model_name}/pc_angles"

    network_ccs = _task_ccs(model_name, "h")
    muscle_ccs = _task_ccs(model_name, "muscle_acts")

    x_muscle = np.arange(1, 5)
    x_network = np.arange(1, 11)
    fig, ax = standard_2d_ax()
    for cc in network_ccs:
        ax.plot(x_network, cc, linewidth=4, color="blue", alpha=0.5)
    for cc in muscle_ccs:
        ax.plot(x_muscle, cc, linewidth=4, color="purple", alpha=0.5)
    ax.set_xticks([10])
    
    save_fig(os.path.join(exp_path, "network_muscle_ccs"), eps=True)
    



def network_muscle_mode_similarity(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pc_angles"
    hp = load_hp(model_path)
    baseline_data = []

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only

    options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "delay_cond": 0, "speed_cond": 5}

    trial_data_hs = []
    trial_data_mas = []
    for env in env_dict:
        trial_data = _test(model_path, model_file, options, env=env_dict[env])
        mov_beg = trial_data["epoch_bounds"]["movement"][0]
        mov_end = trial_data["epoch_bounds"]["movement"][1]
        movement_h = trial_data["h"][:, mov_beg:mov_end]
        movement_muscle = trial_data["muscle_acts"][:, mov_beg:mov_end]
        for cond in range(options["batch_size"]):
            trial_data_hs.append(movement_h[cond])
            trial_data_mas.append(movement_muscle[cond])

    # Get all unique pairs of unit activity across tasks
    combinations_h = list(itertools.combinations(trial_data_hs, 2))
    combinations_mas = list(itertools.combinations(trial_data_mas, 2))

    angles_list_h, control_array_h = principal_angles(combinations_h, baseline_data, "h", hp["hid_size"])
    angles_list_m, control_array_m = principal_angles(combinations_mas, baseline_data, "muscle_acts", 6)

    angles_list_h = np.stack(angles_list_h)
    angles_list_m = np.stack(angles_list_m)

    mean_control_h = np.percentile(control_array_h, 0.1, axis=0, keepdims=False)
    mean_control_m = np.percentile(control_array_m, 0.1, axis=0, keepdims=False)

    comparison_h = angles_list_h < mean_control_h
    comparison_m = angles_list_m < mean_control_m

    percent_below_h = comparison_h.sum(axis=1) / 12 * 100  # shape: (5000,)
    percent_below_m = comparison_m.sum(axis=1) / 3 * 100  # shape: (5000,)

    bin_edges = np.arange(0, 110, 10)

    fig, ax = standard_2d_ax(w=8, h=5)
    ax.hist(percent_below_h, bins=10, range=(0, 100), weights=np.ones_like(percent_below_h) / len(percent_below_h) * 100, color="blue", width=5)
    ax.hist(percent_below_m, bins=10, range=(0, 100), weights=np.ones_like(percent_below_m) / len(percent_below_m) * 100, color="purple", width=5)
    ax.set_xticks(bin_edges)

    save_fig(os.path.join(exp_path, "network_muscle_mode_hist"), eps=True)





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