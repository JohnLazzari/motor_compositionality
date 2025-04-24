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


def _principal_angles(model_name, system, comparison):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pc_angles"

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

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
            trial_data_mode[env] = trial_data[mode][:, trial_data["epoch_bounds"]["delay"][0]:]
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

    angles_dict = principal_angles(combinations, combination_labels)
    
    for angles in angles_dict:
        plt.plot(angles_dict[angles], label=angles)
    # Access current axes and hide top/right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, f"{system}_{comparison}_principal_angles.png"))




def _vaf_ratio(model_name, system, comparison):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pc_angles"
    hp = load_hp(model_path)

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

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
            trial_data_mode[env] = trial_data[mode][:, trial_data["epoch_bounds"]["delay"][0]:]
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
                trial_data["h"][:, :trial_data["epoch_bounds"]["delay"][0]], 
                trial_data["h"][:, trial_data["epoch_bounds"]["delay"][0]:]
            ))

    vaf_ratio_list = vaf_ratio(combinations, hp["hid_size"])
    
    bins = np.linspace(0, 1, 25)
    weights = np.ones_like(vaf_ratio_list) * 100 / len(vaf_ratio_list)
    plt.hist(vaf_ratio_list, bins=bins, weights=weights, color="purple")
    #plt.hist(vaf_ratio_list_control, bins=bins, weights=weights, color="grey")
    plt.xlim(0, 1)

    # Access current axes and hide top/right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, f"{system}_{comparison}_vaf_ratio.png"))




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




def plot_dpca(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/dpca"
    hp = load_hp(model_path)

    max_timesteps = 300

    env_trials = torch.empty(size=(hp["hid_size"], 10, 10, 32, max_timesteps))
    for i, env in enumerate(env_dict):
        speed_trials = torch.empty(size=(hp["hid_size"], 10, 32, max_timesteps))
        for speed in range(10):
            options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": speed}
            trial_data = _test(model_path, model_file, options, env=env_dict[env])

            interpolated_trials = torch.stack([interpolate_trial(h[trial_data["epoch_bounds"]["movement"][0]:], max_timesteps) for h in trial_data["h"]])
            speed_trials[:, speed, ...] = interpolated_trials.permute(2, 0, 1)
        env_trials[:, i, ...] = speed_trials
    
    # mean center
    dpca = dPCA.dPCA(labels='esdt')
    Z = dpca.fit_transform(env_trials.numpy())    

    fig, ax = plt.subplots(1, 4)

    print(Z['t'].shape)
    print(Z['e'].shape)
    print(Z['s'].shape)
    print(Z['d'].shape)

    for s in range(S):
        plot(time,Z['t'][0,s])

    title('1st time component')
        
    for s in range(S):
        plot(time,Z['s'][0,s])
        
    title('1st stimulus component')
        
    for s in range(S):
        plot(time,Z['st'][0,s])
        
    title('1st mixing component')
    save_fig(os.path.join(exp_path, "dpcas.png"))





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
    else:
        raise ValueError("Experiment not in this file")