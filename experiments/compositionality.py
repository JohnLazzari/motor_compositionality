import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp, interpolate_trial

import warnings
warnings.filterwarnings("ignore")

import math
import motornet as mn
from model import RNNPolicy, GRUPolicy
import torch
import os
from utils import load_hp, create_dir, save_fig, load_pickle
import matplotlib.pyplot as plt
import numpy as np
import config
import pickle
from analysis.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
import analysis.plot_utils as plot_utils
from analysis.manifold import principal_angles, vaf_ratio
import tqdm as tqdm
from sklearn.decomposition import PCA, NMF
import sklearn 
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.patches as mpatches
from exp_utils import _test, env_dict, split_movement_epoch, get_interpolation_input, pvalues, get_middle_movement, composite_input_optimization
from exp_utils import distances_from_combinations, angles_from_combinations, shapes_from_combinations, convert_motif_dict_to_list, test_sequential_inputs
from exp_utils import convert_motif_dict_to_list
from envs import DlyHalfReach, DlyHalfCircleClk, DlyHalfCircleCClk, DlySinusoid, DlySinusoidInv
from envs import DlyFullReach, DlyFullCircleClk, DlyFullCircleCClk, DlyFigure8, DlyFigure8Inv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import seaborn as sns
from DSA import DSA
import scipy
from utils import interpolate_trial
import pandas as pd
from sklearn.model_selection import train_test_split
from plt_utils import standard_2d_ax, ax_3d_no_grid, empty_3d, no_ticks_2d_ax, empty_2d_ax

plt.rcParams.update({'font.size': 18})  # Sets default font size for all text

# This is not included in a task pairing
full_movements = [
    "DlyFullReach",
    "DlyFullCircleClk",
    "DlyFullCircleCClk",
    "DlyFigure8",
    "DlyFigure8Inv"
]

env_dict_ext = {
    "DlyHalfReach": DlyHalfReach, 
    "DlyHalfCircleClk": DlyHalfCircleClk, 
    "DlyHalfCircleCClk": DlyHalfCircleCClk, 
    "DlySinusoid": DlySinusoid, 
    "DlySinusoidInv": DlySinusoidInv,
}

env_dict_ret = {
    "DlyFullReach": DlyFullReach, 
    "DlyFullCircleClk": DlyFullCircleClk, 
    "DlyFullCircleCClk": DlyFullCircleCClk, 
    "DlyFigure8": DlyFigure8, 
    "DlyFigure8Inv": DlyFigure8Inv,
}

# These are included in task pairings
extension_movements_half = [
    "DlyHalfReach",
    "DlyHalfCircleClk",
    "DlyHalfCircleCClk",
    "DlySinusoid",
    "DlySinusoidInv"
]

extension_movements_full = [
    "DlyFullReach1",
    "DlyFullCircleClk1",
    "DlyFullCircleCClk1",
    "DlyFigure81",
    "DlyFigure8Inv1"
]

retraction_movements_full = [
    "DlyFullReach2",
    "DlyFullCircleClk2",
    "DlyFullCircleCClk2",
    "DlyFigure82",
    "DlyFigure8Inv2"
]

extension_half_combinations = list(itertools.combinations(extension_movements_half, 2))
extension_full_combinations = list(itertools.combinations(extension_movements_full, 2))
extension_tasks = [*extension_half_combinations, *extension_full_combinations]

#all_extensions = [*extension_movements_half, *extension_movements_full]
#extension_tasks = list(itertools.combinations(all_extensions, 2))

retraction_full_combinations = list(itertools.combinations(retraction_movements_full, 2))
retraction_tasks = retraction_full_combinations

subset_tasks = [
    ("DlyHalfCircleClk", "DlyFullCircleClk1"),
    ("DlySinusoid", "DlyFigure81"),
    ("DlyHalfReach", "DlyFullReach1"),
    ("DlyHalfCircleCClk", "DlyFullCircleCClk1"),
    ("DlySinusoidInv", "DlyFigure8Inv1"),
]

rotated_tasks = [
    ("DlyHalfCircleClk", "DlyHalfCircleCClk"),
    ("DlySinusoid", "DlySinusoidInv"),
    ("DlyFullCircleClk1", "DlyFullCircleCClk1"),
    ("DlyFullCircleClk2", "DlyFullCircleCClk2"),
    ("DlyFigure81", "DlyFigure8Inv1"),
    ("DlyFigure82", "DlyFigure8Inv2"),
]

extension_retraction_tasks = list(itertools.product(extension_movements_half, retraction_movements_full))


def _get_mean_act(model_name, epoch, movement_type, speed_cond=5, delay_cond=1, batch_size=32, add_new_rule_inputs=False):

    """
    Extracts trial-aligned activity vectors at a specified epoch across environments and fits a 2D PCA.

    This function runs a trained model across a set of movement environments, extracts neural or motor
    activity at a specific epoch, and fits a PCA across all collected activity vectors.

    Parameters
    ----------
    model_name : str
        Name of the trained model (used to locate checkpoints and hyperparameters).
    epoch : str
        Epoch from which to extract activity (e.g., 'delay', 'stable', 'extension', 'retraction', 'hold').
    system : str
        System to analyze: either "neural" (hidden state `h`) or "motor" (muscle activations).
    movement_type : str
        Movement type: "extension" (single movement) or "extension_retraction" (compound movement).
    speed_cond : int, optional
        Speed condition for the trials (default: 5).
    delay_cond : int, optional
        Delay condition for the trials (default: 1).
    batch_size : int, optional
        Number of trials to simulate per environment (default: 256).

    Returns
    -------
    epoch_pca : sklearn.decomposition.PCA
        PCA object fitted on activity vectors at the specified epoch across all environments.
    env_hs : list of Tensor
        List containing `[batch_size, 1, size]` tensors of activity for each environment.

    Raises
    ------
    ValueError
        If an unknown `system`, `movement_type`, or `epoch` is provided.
    """

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    hp = load_hp(model_path)

    options = {
        "batch_size": batch_size, 
        "reach_conds": np.arange(0, 32, int(32/batch_size)),
        "speed_cond": speed_cond, 
        "delay_cond": delay_cond
    }

    mode = "h"
    size = hp["hid_size"]
    env_hs = []

    if movement_type == "extension":
        envs_to_use = extension_movements_half.copy()
    elif movement_type == "extension_retraction":
        envs_to_use = full_movements.copy()
    else:
        raise ValueError("not a valid movement type")

    for env in envs_to_use:

        trial_data = _test(model_path, model_file, options, env=env_dict[env], add_new_rule_inputs=add_new_rule_inputs)

        end_delay = trial_data["epoch_bounds"]["delay"][1]-1
        end_stable = trial_data["epoch_bounds"]["stable"][1]-1
        end_movement = trial_data["epoch_bounds"]["movement"][1]-1
        end_hold = trial_data["epoch_bounds"]["hold"][1]-1

        if movement_type == "extension_retraction":

            if epoch == "stable":
                env_hs.append(trial_data[mode][:, end_stable].mean(dim=0).unsqueeze(0))
            elif epoch == "delay":
                env_hs.append(trial_data[mode][:, end_delay].mean(dim=0).unsqueeze(0))
            elif epoch == "extension":
                middle_movement = get_middle_movement(trial_data)
                env_hs.append(trial_data[mode][:, middle_movement].mean(dim=0).unsqueeze(0))
            elif epoch == "retraction":
                env_hs.append(trial_data[mode][:, end_movement].mean(dim=0).unsqueeze(0))
            elif epoch == "hold":
                env_hs.append(trial_data[mode][:, end_hold].mean(dim=0).unsqueeze(0))
            else:
                raise ValueError("not valid epoch for extension-retraction task")
        
        elif movement_type == "extension":

            if epoch == "stable":
                env_hs.append(trial_data[mode][:, end_stable].mean(dim=0).unsqueeze(0))
            elif epoch == "delay":
                env_hs.append(trial_data[mode][:, end_delay].mean(dim=0).unsqueeze(0))
            elif epoch == "extension":
                env_hs.append(trial_data[mode][:, end_movement].mean(dim=0).unsqueeze(0))
            elif epoch == "hold":
                env_hs.append(trial_data[mode][:, end_hold].mean(dim=0).unsqueeze(0))
            else:
                raise ValueError("not valid epoch for extension task")


    return env_hs, envs_to_use




def _epoch_pcs(model_name, epoch, movement_type, add_new_rule_inputs=False, plot_3d=False):

    exp_path = f"results/{model_name}/compositionality/pcs"
    create_dir(exp_path)

    env_hs, envs_to_use = _get_mean_act(model_name, epoch, movement_type, add_new_rule_inputs=add_new_rule_inputs)

    epoch_pca = PCA(n_components=3)
    epoch_pca.fit(torch.cat(env_hs, dim=0))

    colors = plt.cm.tab10(np.linspace(0, 1, len(env_dict))) 
    env_color_dict = {}

    # Adding colors for easier indexing
    for (env, color) in zip(env_dict, colors):
        env_color_dict[env] = color

    handles = []

    if plot_3d:
        fig, ax = ax_3d_no_grid()
    else:
        fig, ax = no_ticks_2d_ax()
    

    for i, (env_data, env) in enumerate(zip(env_hs, envs_to_use)):

        # Create patches with no border
        handles.append(mpatches.Patch(color=env_color_dict[env], label=env, edgecolor='none'))

        # transform
        h_proj = epoch_pca.transform(env_data)

        # Plot the 3D line
        if plot_3d:
            ax.scatter(h_proj[-1, 0], h_proj[-1, 1], h_proj[-1, 2], color=env_color_dict[env], s=250, alpha=0.75)
        else:
            ax.scatter(h_proj[-1, 0], h_proj[-1, 1], color=env_color_dict[env], s=250, alpha=0.75)
    
    # Set labels for axes
    save_fig(os.path.join(exp_path, f"{epoch}_{movement_type}_pcs"), eps=True)




def _epoch_lda(model_name, epoch, movement_type, add_new_rule_inputs=False):

    exp_path = f"results/{model_name}/compositionality/lda"
    create_dir(exp_path)

    env_hs, envs_to_use = _get_mean_act(model_name, epoch, movement_type, add_new_rule_inputs=add_new_rule_inputs)
    envs_to_use = envs_to_use[1:]
    env_hs = torch.cat(env_hs[1:], dim=0).numpy()

    """
    # Perform LDA across kinematics
    kinematic_labels = np.array([0, 0, 1, 1])
    kin_lda = LinearDiscriminantAnalysis()
    kin_lda.fit(env_hs, kinematic_labels)

    # Perform LDA across rotations
    rotation_labels = np.array([0, 1, 0, 1])
    rot_lda = LinearDiscriminantAnalysis()
    rot_lda.fit(env_hs, rotation_labels)

    kin_comp = kin_lda.coef_.squeeze()
    rot_comp = rot_lda.coef_.squeeze()
    """

    kin_comp = env_hs[:2].mean(axis=0) - env_hs[2:].mean(axis=0)
    rot_comp = env_hs[[0, 2]].mean(axis=0) - env_hs[[1, 3]].mean(axis=0)
    
    # Normalize v1 to get the first orthogonal vector
    u1 = kin_comp

    # Subtract projection of v2 onto v1 from v2
    proj_v2_on_u1 = np.dot(rot_comp, u1) / np.dot(u1, u1) * u1
    u2 = rot_comp - proj_v2_on_u1

    colors = plt.cm.tab10(np.linspace(0, 1, len(env_dict))) 
    env_color_dict = {}

    # Adding colors for easier indexing
    for (env, color) in zip(env_dict, colors):
        env_color_dict[env] = color

    handles = []
    fig, ax = no_ticks_2d_ax()

    for i, (env_data, env) in enumerate(zip(env_hs, envs_to_use)):

        # Create patches with no border
        handles.append(mpatches.Patch(color=env_color_dict[env], label=env, edgecolor='none'))

        # transform
        kin_trans = (u1 / np.linalg.norm(u1)) @ env_data
        rot_trans = (u2 / np.linalg.norm(u2)) @ env_data

        ax.scatter(kin_trans, rot_trans, color=env_color_dict[env], s=250, alpha=0.75)
    
    # Set labels for axes
    save_fig(os.path.join(exp_path, f"{epoch}_{movement_type}_lda"), eps=True)




def _two_task_pcs(model_name, task1, task2, task1_period, task2_period, system):

    # File names
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/pcs"

    # testing options
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 1, "speed_cond": 5}
    colors = plt.cm.inferno(np.linspace(0, 1, 10)) 

    # Gather data from test runs
    trial_data1 = _test(model_path, model_file, options, env=env_dict[task1], noise=False)
    trial_data2 = _test(model_path, model_file, options, env=env_dict[task2], noise=False)

    # Get hidden activity during the first or second half of movement (or all)
    trial_data1_h_epoch = split_movement_epoch(trial_data1, task1_period, system)
    C_1, T_1, N_1 = trial_data1_h_epoch.shape
    trial_data2_h_epoch = split_movement_epoch(trial_data2, task2_period, system)
    C_2, T_2, N_2 = trial_data2_h_epoch.shape

    # Combine all trials and timepoints then do PCA
    combined_tasks = torch.cat([trial_data1_h_epoch, trial_data2_h_epoch], dim=1)
    pca_3d = PCA(n_components=3)
    pca_3d.fit(combined_tasks.reshape((-1, combined_tasks.shape[-1])))

    # Transform to get low dimensional trajectories for both tasks
    task1_projected = pca_3d.transform(trial_data1_h_epoch.reshape((-1, trial_data1_h_epoch.shape[-1]))).reshape((C_1, T_1, 3))
    task2_projected = pca_3d.transform(trial_data2_h_epoch.reshape((-1, trial_data2_h_epoch.shape[-1]))).reshape((C_2, T_2, 3))
    
    for c in range(options["batch_size"]):

        fig, ax = ax_3d_no_grid()
        shadow_traj = np.concatenate([task1_projected[c, :], task2_projected[c, :]])
        min_z = np.min(shadow_traj[:, 2])

        ax.plot(task1_projected[c, :, 0], task1_projected[c, :, 1], task1_projected[c, :, 2], linewidth=4, alpha=0.75, color=colors[7], zorder=10)
        ax.plot(task1_projected[c, :, 0], task1_projected[c, :, 1], min_z, linewidth=4, alpha=0.25, color="grey")
        ax.scatter(task1_projected[c, 0, 0], task1_projected[c, 0, 1], task1_projected[c, 0, 2], s=150, marker="^", color=colors[7])
        ax.scatter(task1_projected[c, -1, 0], task1_projected[c, -1, 1], task1_projected[c, -1, 2], s=150, marker="X", color=colors[7])

        ax.plot(task2_projected[c, :, 0], task2_projected[c, :, 1], task2_projected[c, :, 2], linewidth=4, alpha=0.75, color="black", zorder=10)
        ax.plot(task2_projected[c, :, 0], task2_projected[c, :, 1], min_z, linewidth=4, alpha=0.25, color="grey")
        ax.scatter(task2_projected[c, 0, 0], task2_projected[c, 0, 1], task2_projected[c, 0, 2], s=150, marker="^", color="black")
        ax.scatter(task2_projected[c, -1, 0], task2_projected[c, -1, 1], task2_projected[c, -1, 2], s=150, marker="X", color="black")

        save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{system}_{task1}_{task2}_cond_{c}_pcs"), eps=True)

        fig, ax = ax_3d_no_grid()
        mtx1, mtx2, _ = scipy.spatial.procrustes(task1_projected[c, :], task2_projected[c, :])
        shadow_traj = np.concatenate([mtx1, mtx2])
        min_zp = np.min(shadow_traj[:, 2])

        ax.plot(mtx1[:, 0], mtx1[:, 1], mtx1[:, 2], linewidth=4, alpha=0.75, color=colors[7], zorder=10)
        ax.plot(mtx1[:, 0], mtx1[:, 1], min_zp, linewidth=4, alpha=0.25, color="grey")
        ax.scatter(mtx1[0, 0], mtx1[0, 1], mtx1[0, 2], s=150, marker="^", color=colors[7])
        ax.scatter(mtx1[-1, 0], mtx1[-1, 1], mtx1[-1, 2], s=150, marker="X", color=colors[7])

        ax.plot(mtx2[:, 0], mtx2[:, 1], mtx2[:, 2], linewidth=4, alpha=0.75, color="black", zorder=10)
        ax.plot(mtx2[:, 0], mtx2[:, 1], min_zp, linewidth=4, alpha=0.25, color="grey")
        ax.scatter(mtx2[0, 0], mtx2[0, 1], mtx2[0, 2], s=150, marker="^", color="black")
        ax.scatter(mtx2[-1, 0], mtx2[-1, 1], mtx2[-1, 2], s=150, marker="X", color="black")

        save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{system}_{task1}_{task2}_cond_{c}_procrustes_pcs"), eps=True)






def stable_pcs_extension(model_name, add_new_rule_inputs=False, plot_3d=False):
    _epoch_pcs(model_name, "stable", "extension", add_new_rule_inputs=add_new_rule_inputs, plot_3d=plot_3d)
def delay_pcs_extension(model_name, add_new_rule_inputs=False, plot_3d=False):
    _epoch_pcs(model_name, "delay", "extension", add_new_rule_inputs=add_new_rule_inputs, plot_3d=plot_3d)
def movement_pcs_extension(model_name, add_new_rule_inputs=False, plot_3d=False):
    _epoch_pcs(model_name, "extension", "extension", add_new_rule_inputs=add_new_rule_inputs, plot_3d=plot_3d)
def hold_pcs_extension(model_name, add_new_rule_inputs=False, plot_3d=False):
    _epoch_pcs(model_name, "hold", "extension", add_new_rule_inputs=add_new_rule_inputs, plot_3d=plot_3d)

def stable_pcs_extension_retraction(model_name, add_new_rule_inputs=False, plot_3d=False):
    _epoch_pcs(model_name, "stable", "extension_retraction", add_new_rule_inputs=add_new_rule_inputs, plot_3d=plot_3d)
def delay_pcs_extension_retraction(model_name, add_new_rule_inputs=False, plot_3d=False):
    _epoch_pcs(model_name, "delay", "extension_retraction", add_new_rule_inputs=add_new_rule_inputs, plot_3d=plot_3d)
def extension_pcs_extension_retraction(model_name, add_new_rule_inputs=False, plot_3d=False):
    _epoch_pcs(model_name, "extension", "extension_retraction", add_new_rule_inputs=add_new_rule_inputs, plot_3d=plot_3d)
def retraction_pcs_extension_retraction(model_name, add_new_rule_inputs=False, plot_3d=False):
    _epoch_pcs(model_name, "retraction", "extension_retraction", add_new_rule_inputs=add_new_rule_inputs, plot_3d=plot_3d)
def hold_pcs_extension_retraction(model_name, add_new_rule_inputs=False, plot_3d=False):
    _epoch_pcs(model_name, "hold", "extension_retraction", add_new_rule_inputs=add_new_rule_inputs, plot_3d=plot_3d)

def run_all_epoch_pcs(model_name):
    stable_pcs_extension(model_name)
    delay_pcs_extension(model_name)
    movement_pcs_extension(model_name)
    hold_pcs_extension(model_name)
    stable_pcs_extension_retraction(model_name)
    delay_pcs_extension_retraction(model_name)
    extension_pcs_extension_retraction(model_name)
    retraction_pcs_extension_retraction(model_name)
    hold_pcs_extension_retraction(model_name)

def run_all_epoch_pcs_transfer(model_name):
    stable_pcs_extension(model_name, add_new_rule_inputs=True)
    delay_pcs_extension(model_name, add_new_rule_inputs=True)
    movement_pcs_extension(model_name, add_new_rule_inputs=True)
    hold_pcs_extension(model_name, add_new_rule_inputs=True)
    stable_pcs_extension_retraction(model_name, add_new_rule_inputs=True)
    delay_pcs_extension_retraction(model_name, add_new_rule_inputs=True)
    extension_pcs_extension_retraction(model_name, add_new_rule_inputs=True)
    retraction_pcs_extension_retraction(model_name, add_new_rule_inputs=True)
    hold_pcs_extension_retraction(model_name, add_new_rule_inputs=True)





def stable_lda_extension(model_name, add_new_rule_inputs=False):
    _epoch_lda(model_name, "stable", "extension", add_new_rule_inputs=add_new_rule_inputs)
def delay_lda_extension(model_name, add_new_rule_inputs=False):
    _epoch_lda(model_name, "delay", "extension", add_new_rule_inputs=add_new_rule_inputs)
def movement_lda_extension(model_name, add_new_rule_inputs=False):
    _epoch_lda(model_name, "extension", "extension", add_new_rule_inputs=add_new_rule_inputs)
def hold_lda_extension(model_name, add_new_rule_inputs=False):
    _epoch_lda(model_name, "hold", "extension", add_new_rule_inputs=add_new_rule_inputs)

def stable_lda_extension_retraction(model_name, add_new_rule_inputs=False):
    _epoch_lda(model_name, "stable", "extension_retraction", add_new_rule_inputs=add_new_rule_inputs)
def delay_lda_extension_retraction(model_name, add_new_rule_inputs=False):
    _epoch_lda(model_name, "delay", "extension_retraction", add_new_rule_inputs=add_new_rule_inputs)
def extension_lda_extension_retraction(model_name, add_new_rule_inputs=False):
    _epoch_lda(model_name, "extension", "extension_retraction", add_new_rule_inputs=add_new_rule_inputs)
def retraction_lda_extension_retraction(model_name, add_new_rule_inputs=False):
    _epoch_lda(model_name, "retraction", "extension_retraction", add_new_rule_inputs=add_new_rule_inputs)
def hold_lda_extension_retraction(model_name, add_new_rule_inputs=False):
    _epoch_lda(model_name, "hold", "extension_retraction", add_new_rule_inputs=add_new_rule_inputs)

def run_all_epoch_lda(model_name):
    stable_lda_extension(model_name)
    delay_lda_extension(model_name)
    movement_lda_extension(model_name)
    hold_lda_extension(model_name)
    stable_lda_extension_retraction(model_name)
    delay_lda_extension_retraction(model_name)
    extension_lda_extension_retraction(model_name)
    retraction_lda_extension_retraction(model_name)
    hold_lda_extension_retraction(model_name)

def run_all_epoch_lda_transfer(model_name):
    stable_lda_extension(model_name, add_new_rule_inputs=True)
    delay_lda_extension(model_name, add_new_rule_inputs=True)
    movement_lda_extension(model_name, add_new_rule_inputs=True)
    hold_lda_extension(model_name, add_new_rule_inputs=True)
    stable_lda_extension_retraction(model_name, add_new_rule_inputs=True)
    delay_lda_extension_retraction(model_name, add_new_rule_inputs=True)
    extension_lda_extension_retraction(model_name, add_new_rule_inputs=True)
    retraction_lda_extension_retraction(model_name, add_new_rule_inputs=True)
    hold_lda_extension_retraction(model_name, add_new_rule_inputs=True)




def neural_two_task_pcs_sinusoid_fullcircleclk(model_name):
    _two_task_pcs(model_name, "DlySinusoid", "DlyFullCircleClk", "all", "second", "h")
def muscle_two_task_pcs_sinusoid_fullcircleclk(model_name):
    _two_task_pcs(model_name, "DlySinusoid", "DlyFullCircleClk", "all", "second", "muscle_acts")

def neural_two_task_pcs_halfcircleclk_halfcirclecclk(model_name):
    _two_task_pcs(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "all", "all", "h")
def muscle_two_task_pcs_halfcircleclk_halfcirclecclk(model_name):
    _two_task_pcs(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "all", "all", "muscle_acts")




def _interpolated_fps(model_name, task1, task2,  epoch, task1_period="all", task2_period="all", input_component=None, 
    add_new_rule_inputs=False, num_new_inputs=10):

    # Task 1 and task 2 period represent either collecting only half of the movement, or all, in specific periods
    # These values can be first, second, or all

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"

    NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
    N_INITS = 1024 # The number of initial states to provide

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 16, "reach_conds": torch.arange(0, 32, 2), "delay_cond": 1, "speed_cond": 5}

    hp = load_hp(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

    # Loading in model
    if hp["network"] == "rnn":
        policy = RNNPolicy(
            hp["inp_size"],
            hp["hid_size"],
            effector.n_muscles, 
            activation_name=hp["activation_name"],
            noise_level_act=hp["noise_level_act"], 
            noise_level_inp=hp["noise_level_inp"], 
            constrained=hp["constrained"], 
            dt=hp["dt"],
            t_const=hp["t_const"],
            device=device,
            add_new_rule_inputs=add_new_rule_inputs,
            num_new_inputs=num_new_inputs
        )
    elif hp["network"] == "gru":
        policy = GRUPolicy(hp["inp_size"], hp["hid_size"], effector.n_muscles, batch_first=True)
    else:
        raise ValueError("Not a valid architecture")

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=torch.device('cpu'))
    policy.load_state_dict(checkpoint['agent_state_dict'])

    trial_data1 = _test(model_path, model_file, options, env=env_dict[task1], add_new_rule_inputs=add_new_rule_inputs, num_new_inputs=num_new_inputs)
    trial_data2 = _test(model_path, model_file, options, env=env_dict[task2], add_new_rule_inputs=add_new_rule_inputs, num_new_inputs=num_new_inputs)

    if epoch == "delay":

        # Get inputs and x and h from desired timepoint
        inp1 = trial_data1["obs"][:, trial_data1["epoch_bounds"]["delay"][1]-1]
        inp2 = trial_data2["obs"][:, trial_data2["epoch_bounds"]["delay"][1]-1]

    elif epoch == "movement":

        inp1 = get_interpolation_input(trial_data1, task1_period)
        inp2 = get_interpolation_input(trial_data2, task2_period)

    '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
    descriptions of available hyperparameters.'''

    fpf_hps = {
        'max_iters': 250,
        'lr_init': 1.,
        'outlier_distance_scale': 10.0,
        'verbose': True, 
        'super_verbose': False,
        'tol_unique': 1,
        'do_compute_jacobians': True}
        
    cond_fps_list = []
    for c, (cond1, cond2) in enumerate(zip(inp1, inp2)):

        # Setup environment and initialize it
        env = env_dict[task1](effector=mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle()))
        obs, info = env.reset(testing=True, options=options)

        # May need to change in the future if I do middle of movement
        x = torch.zeros(size=(1, hp["hid_size"]))
        h = torch.zeros(size=(1, hp["hid_size"]))

        if epoch == "delay":
            timesteps = env.epoch_bounds["delay"][1]-1
        elif epoch == "movement":
            timesteps = middle_movement1
            
        # Draw a line from fingertip to goal 
        if input_component == None:
            interpolated_input = cond1.unsqueeze(0) + \
                torch.linspace(0, 1, steps=20).unsqueeze(1) * (cond2 - cond1).unsqueeze(0)

        elif input_component == "rule":
            interpolated_input_rule = cond1[:10].unsqueeze(0) + \
                torch.linspace(0, 1, steps=20).unsqueeze(1) * (cond2[:10] - cond1[:10]).unsqueeze(0)
            fixed_inp = cond1[10:].repeat(20, 1)
            interpolated_input = torch.cat([interpolated_input_rule, fixed_inp], dim=1)

        elif input_component == "proprioception":
            fixed_inp_pre = cond1[:16].repeat(20, 1)
            interpolated_input_proprioception = cond1[16:].unsqueeze(0) + \
                torch.linspace(0, 1, steps=20).unsqueeze(1) * (cond2[16:] - cond1[16:]).unsqueeze(0)
            interpolated_input = torch.cat([fixed_inp_pre, interpolated_input_proprioception], dim=1)

        fps_list = []
        # Going thorugh each interpolated input
        for i, inp in enumerate(interpolated_input):

            # Setup the fixed point finder
            fpf = FixedPointFinder(policy.mrnn, **fpf_hps)

            '''Draw random, noise corrupted samples of those state trajectories
            to use as initial states for the fixed point optimizations.'''

            # Currently using original h for initial states
            initial_states = fpf.sample_states(trial_data1["h"][c:c+1, timesteps:timesteps+1],
                n_inits=N_INITS,
                noise_scale=NOISE_SCALE)

            # Run the fixed point finder
            unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs=inp[None, :])

            # Add fixed points and their info to dict
            fps_list.append(
                {"fps": unique_fps, 
                "interp_point": i 
                }
            )

        cond_fps_list.append(fps_list)

    # Save all information of fps across tasks to pickle file
    save_name = f'interpolated_fps_{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}'
    fname = os.path.join(model_path, save_name + '.pkl')
    print('interpolated fps saved at {:s}'.format(fname))
    with open(fname, 'wb') as f:
        pickle.dump(cond_fps_list, f)





#---------------------------------------------------------------- Subset Pair

# Delay period with different input interpolations
def compute_interpolated_fps_halfreach_fullreach_delay(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay")
# Movement period with different input interpolations
def compute_interpolated_fps_halfreach_fullreach_movement(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", task1_period="all", task2_period="first")

#---------------------------------------------------------------- Extension Pair

# Delay period with different input interpolations
def compute_interpolated_fps_halfcircleclk_sinusoidinv_delay(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "delay")
# Movement period with different input interpolations
def compute_interpolated_fps_halfcircleclk_sinusoidinv_movement(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "movement", task1_period="all", task2_period="all")

#---------------------------------------------------------------- Retraction Pair

# Delay period with different input interpolations
def compute_interpolated_fps_fullcircleclk_figure8_delay(model_name):
    _interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "delay")
# Movement period with different input interpolations
def compute_interpolated_fps_fullcircleclk_figure8_movement(model_name):
    _interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "movement", task1_period="second", task2_period="second")

#---------------------------------------------------------------- Extension-Retraction Pair

# Delay period with different input interpolations
def compute_interpolated_fps_sinusoid_fullreach_delay(model_name):
    _interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "delay")
# Movement period with different input interpolations
def compute_interpolated_fps_sinusoid_fullreach_movement(model_name):
    _interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "movement", task1_period="all", task2_period="second")




def run_all_compute_interpolated_fps(model_name):
    compute_interpolated_fps_halfreach_fullreach_delay(model_name)
    compute_interpolated_fps_halfreach_fullreach_movement(model_name)
    compute_interpolated_fps_halfcircleclk_sinusoidinv_delay(model_name)
    compute_interpolated_fps_halfcircleclk_sinusoidinv_movement(model_name)
    compute_interpolated_fps_fullcircleclk_figure8_delay(model_name)
    compute_interpolated_fps_fullcircleclk_figure8_movement(model_name)
    compute_interpolated_fps_sinusoid_fullreach_delay(model_name)
    compute_interpolated_fps_sinusoid_fullreach_movement(model_name)





def _plot_interpolated_fps(model_name, task1, task2, epoch, task1_period="all", task2_period="all", input_component=None,
        add_new_rule_inputs=False, num_new_inputs=10, save_metrics=False, y_dist=1):
    """
    Generate and save visualizations of fixed points and their dynamics for interpolated inputs between two tasks.

    This function performs the following:
        1. Loads fixed points obtained from interpolated rule inputs between two tasks.
        2. Projects fixed points and trial trajectories into PCA space.
        3. Plots fixed points color-coded by stability over interpolation.
        4. Plots the maximum eigenvalue of the Jacobian at each fixed point across interpolation steps.
        5. Plots the Euclidean distance between selected fixed points across interpolation steps.
        6. Projects and plots the neural population trajectories for both tasks in 3D PCA space.

    Parameters
    ----------
    model_name : str
        Name of the trained model to load fixed points and generate test data.
    task1 : str
        Name of the first task for interpolation and testing.
    task2 : str
        Name of the second task for interpolation and testing.
    epoch : str
        Epoch name or key (e.g., "go", "delay") to extract activity from trial data.
    task1_period : str, optional
        Task period to extract for PCA projection from task1 trials. Default is "all".
    task2_period : str, optional
        Task period to extract for PCA projection from task2 trials. Default is "all".
    input_component : str or None, optional
        Name of the input component used to generate interpolations. Used in figure file names.
    add_new_rule_inputs : bool, optional
        Whether to include new rule inputs when generating test trials. Default is False.
    num_new_inputs : int, optional
        Number of new rule inputs to include if `add_new_rule_inputs` is True. Default is 10.
    save_metrics : bool, optional
        Whether to return eigenvalue and distance metrics computed during plotting. Default is False.
    y_dist : float, optional
        Y-axis limit for the Euclidean distance plot. Default is 1.

    Returns
    -------
    tuple of list[list[float]], optional
        Only returned if `save_metrics` is True. Contains:
        - max_eigs_conds: Per-condition list of differences in max eigenvalues across interpolation.
        - euc_dists_conds: Per-condition list of differences in Euclidean distances between successive fixed points.

    Notes
    -----
    - Requires fixed point data to be saved as a pickle file in `checkpoints/{model_name}`.
    - Saves plots to `results/{model_name}/compositionality/interpolated_fps/...`.
    - Assumes access to auxiliary plotting utilities such as `plot_utils.plot_fixed_point`, `ax_3d_no_grid`, `standard_2d_ax`, `save_fig`, etc.
    - Uses PCA projections computed only from task1 trial data for FP projection.
    """
    def fp_plot():
        """
        Plot fixed points projected into 2D PCA space, color-coded by interpolation step and stability.

        Fixed points with a leading eigenvalue greater than 1 are considered unstable and are marked with white faces.
        Saves the resulting plot to disk.
        """
        colors_alpha = plt.cm.magma(np.linspace(0, 1, 20)) 
        two_task_pca = PCA(n_components=2)
        two_task_pca.fit(trial_data1_h_epoch.reshape((-1, trial_data1_h_epoch.shape[-1])))

        fig, ax = ax_3d_no_grid()
        # cond is a list containing the fps for each interpolated rule input for a given condition
        for c, cond in enumerate(fps):
            
            interpolated_fps = [unique_fps["fps"] for unique_fps in cond]

            for i, fps_step in enumerate(interpolated_fps):
                n_inits = fps_step.n
                for init_idx in range(n_inits):
                    zstar = plot_utils.plot_fixed_point(
                                fps_step[init_idx],
                                two_task_pca,
                                make_plot=False
                            )
                    # Stability of top eigenvalue
                    stability = np.abs(fps_step[init_idx].eigval_J_xstar[0, 0])
                    if stability > 1:
                        ax.scatter(i/20, zstar[:, 0], zstar[:, 1], marker='.', alpha=0.75, edgecolors=colors_alpha[i], facecolors="w", s=250)
                    else:
                        ax.scatter(i/20, zstar[:, 0], zstar[:, 1], marker='.', alpha=0.75, color=colors_alpha[i], s=250)

        save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}"), eps=True)

    def max_eigenvalues():
        """
        Plot maximum eigenvalues of selected fixed points across interpolation steps.

        At each step, the fixed point closest to the previously chosen one (or a trajectory point for the first step)
        is selected. Plots per-condition eigenvalue traces and saves the figure.
        """
        fig, ax = standard_2d_ax()
        colors_conds = plt.cm.inferno(np.linspace(0, 1, 16)) 
        # Generate a plot of max eigenvalues
        max_eigs_conds = []
        for c, cond in enumerate(fps):
            interpolated_fps = [unique_fps["fps"] for unique_fps in cond]
            max_eigs = []
            chosen_fps = []
            for i, fps_step in enumerate(interpolated_fps):
                n_inits = fps_step.n
                max_eig = 0
                dist = np.inf

                if i == 0:
                    for init_idx in range(n_inits):
                        cur_dist = np.linalg.norm(trajectory_point[c] - fps_step[init_idx].xstar)
                        if cur_dist < dist:
                            dist = cur_dist
                            chosen_fp = fps_step[init_idx]
                else:
                    for init_idx in range(n_inits):
                        cur_dist = np.linalg.norm(chosen_fps[i-1].xstar - fps_step[init_idx].xstar)
                        if cur_dist < dist:
                            dist = cur_dist
                            chosen_fp = fps_step[init_idx]
                chosen_fps.append(chosen_fp)

                max_eig = chosen_fp.eigval_J_xstar[0, 0].real
                max_eigs.append(max_eig)

            x = np.arange(1, 21)
            eig_mean = np.mean(max_eigs)
            ax.scatter(x, max_eigs, marker="o", color=colors_conds[c], s=200, alpha=0.75)
            ax.set_ylim([0.5, 1.2])
            max_eigs_conds.append([np.abs(max_eigs[i+1] - max_eigs[i]) for i in range(len(max_eigs)-1)])
        save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}_max_eigs"), eps=True)
        return max_eigs_conds

    def euc_dists():
        """
        Plot Euclidean distances between matched fixed points across interpolation steps.

        Uses nearest-neighbor matching across steps to construct a smooth path through fixed points.
        Saves the resulting distance plot.
        """
        fig, ax = standard_2d_ax()
        # Generate a plot of fp distances
        euc_dists_conds = []
        for c, cond in enumerate(fps):
            interpolated_fps = [unique_fps["fps"] for unique_fps in cond]
            chosen_fps = []
            for i, fps_step in enumerate(interpolated_fps):
                n_inits = fps_step.n
                dist = np.inf
                if i == 0:
                    for init_idx in range(n_inits):
                        cur_dist = np.linalg.norm(trajectory_point[c] - fps_step[init_idx].xstar)
                        if cur_dist < dist:
                            dist = cur_dist
                            chosen_fp = fps_step[init_idx].xstar
                else:
                    for init_idx in range(n_inits):
                        cur_dist = np.linalg.norm(chosen_fps[i-1] - fps_step[init_idx].xstar)
                        if cur_dist < dist:
                            dist = cur_dist
                            chosen_fp = fps_step[init_idx].xstar
                chosen_fps.append(chosen_fp)
            
            dist_list = [np.linalg.norm(chosen_fps[i+1] - chosen_fps[i]) for i in range(len(chosen_fps)-1)]
            x = np.arange(1, 20)
            ax.scatter(x, dist_list, marker="o", color=colors_conds[c], s=200, alpha=0.75)
            ax.set_ylim([0, y_dist])
            euc_dists_conds.append([np.abs(dist_list[i+1] - dist_list[i]) for i in range(len(dist_list)-1)])
        save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}_dists"), eps=True)
        return euc_dist_conds
        

    def pc_plots():
        """
        Plot 3D PCA projections of trial trajectories for both tasks.

        Projects trial trajectories into a shared 3D PCA space computed from concatenated task1 and task2 data.
        Saves the resulting 3D trajectory plot.
        """
        fig, ax = empty_3d() 
        colors_traj = plt.cm.inferno(np.linspace(0, 1, 10)) 

        trial_data1_h_epoch = split_movement_epoch(trial_data1, task1_period, "h")
        trial_data2_h_epoch = split_movement_epoch(trial_data2, task2_period, "h")

        # Get trajectories during task period
        combined_tasks = torch.cat([trial_data1_h_epoch, trial_data2_h_epoch], dim=1)

        two_task_pca_3d = PCA(n_components=3)
        two_task_pca_3d.fit(combined_tasks.reshape((-1, combined_tasks.shape[-1])))

        trial_data_1_projected = two_task_pca_3d.transform(trial_data1_h_epoch.reshape((-1, trial_data1_h_epoch.shape[-1])))
        trial_data_2_projected = two_task_pca_3d.transform(trial_data2_h_epoch.reshape((-1, trial_data2_h_epoch.shape[-1])))

        trial_data_1_projected = trial_data_1_projected.reshape((
            trial_data1_h_epoch.shape[0], 
            trial_data1_h_epoch.shape[1],
            3
        ))
        
        trial_data_2_projected = trial_data_2_projected.reshape((
            trial_data2_h_epoch.shape[0], 
            trial_data2_h_epoch.shape[1],
            3
        ))

        for c, condition in enumerate(trial_data_1_projected):
            ax.plot(condition[:, 0], condition[:, 1], condition[:, 2], linewidth=4, color=colors_traj[7], alpha=0.75, zorder=10)
            ax.scatter(condition[0, 0], condition[0, 1], condition[0, 2], s=150, marker="^", color=colors_traj[7])
            ax.scatter(condition[-1, 0], condition[-1, 1], condition[-1, 2], s=150, marker="X", color=colors_traj[7])
        for c, condition in enumerate(trial_data_2_projected):
            ax.plot(condition[:, 0], condition[:, 1], condition[:, 2], linewidth=4, color="black", alpha=0.75, zorder=10)
            ax.scatter(condition[0, 0], condition[0, 1], condition[0, 2], s=150, marker="^", color="black", zorder=20)
            ax.scatter(condition[-1, 0], condition[-1, 1], condition[-1, 2], s=150, marker="X", color="black", zorder=20)

        save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}_pca"), eps=True)

    # Setup paths and load fixed points
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    load_name = os.path.join(model_path, f"interpolated_fps_{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}.pkl")
    exp_path = f"results/{model_name}/compositionality/interpolated_fps"

    fps = load_pickle(load_name)

    options = {"batch_size": 16, "reach_conds": torch.arange(0, 32, 2), "delay_cond": 1, "speed_cond": 5}

    # Get trial data from model
    trial_data1 = _test(model_path, model_file, options, env=env_dict[task1], add_new_rule_inputs=add_new_rule_inputs, num_new_inputs=num_new_inputs)
    trial_data2 = _test(model_path, model_file, options, env=env_dict[task2], add_new_rule_inputs=add_new_rule_inputs, num_new_inputs=num_new_inputs)

    trial_data1_h_epoch = trial_data1["h"][:, trial_data1["epoch_bounds"][epoch][0]:trial_data1["epoch_bounds"][epoch][1]] 
    trial_data2_h_epoch = trial_data2["h"][:, trial_data2["epoch_bounds"][epoch][0]:trial_data2["epoch_bounds"][epoch][1]] 
    halfway_task1 = int(trial_data1_h_epoch.shape[1] / 2)

    trajectory_point = trial_data1_h_epoch[:, halfway_task1]

    # Make all plots
    fp_plot()
    max_eigs_conds = max_eigenvalues()
    euc_dists_conds = euc_dists()
    pc_plots()

    if save_metrics:
        return max_eigs_conds, euc_dists_conds
    





#---------------------------------------------------------------- Subset Pair
def plot_interpolated_fps_halfreach_fullreach_movement(model_name, save_metrics=False):
    if save_metrics:
        max_eigs, euc_dist = _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", 
            task1_period="all", task2_period="first", save_metrics=save_metrics)
        return max_eigs, euc_dist
    else:
        _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", 
            task1_period="all", task2_period="first", save_metrics=save_metrics)
#---------------------------------------------------------------- Extension Pair
def plot_interpolated_fps_halfcircleclk_sinusoidinv_movement(model_name, save_metrics=False):
    if save_metrics:
        max_eigs, euc_dist = _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "movement", 
            task1_period="all", task2_period="all", save_metrics=save_metrics)
        return max_eigs, euc_dist
    else:
        _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "movement", 
            task1_period="all", task2_period="all", save_metrics=save_metrics)
#---------------------------------------------------------------- Retraction Pair
def plot_interpolated_fps_fullcircleclk_figure8_movement(model_name, save_metrics=False):
    if save_metrics:
        max_eigs, euc_dist = _plot_interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "movement", 
            task1_period="second", task2_period="second", save_metrics=save_metrics)
        return max_eigs, euc_dist
    else:
        _plot_interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "movement", 
            task1_period="second", task2_period="second", save_metrics=save_metrics)
#---------------------------------------------------------------- Extension-Retraction Pair
def plot_interpolated_fps_sinusoid_fullreach_movement(model_name, save_metrics=False):
    if save_metrics:
        max_eigs, euc_dist = _plot_interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "movement", 
            task1_period="all", task2_period="second", save_metrics=save_metrics)
        return max_eigs, euc_dist
    else:
        _plot_interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "movement", 
            task1_period="all", task2_period="second", save_metrics=save_metrics)



def run_all_plot_interpolated_fps(model_name):
    # Add data from each task
    plot_interpolated_fps_halfreach_fullreach_movement(model_name, save_metrics=False)
    plot_interpolated_fps_halfcircleclk_sinusoidinv_movement(model_name, save_metrics=False)
    plot_interpolated_fps_fullcircleclk_figure8_movement(model_name, save_metrics=False)
    plot_interpolated_fps_sinusoid_fullreach_movement(model_name, save_metrics=False)




def _plot_bar(combinations, metric, exp_path, metric_name, combination_labels, combination_colors):

    fig, ax = standard_2d_ax()

    combination_means = []
    combination_stds = []
    combination_data = {}
    for l, combination in enumerate(combinations):
        task_metric = convert_motif_dict_to_list(combination, metric)

        """
            This is for finding examples for two_task_pcs. Find task pairs with large disparities, etc.
            Delete this once done for sure, change as needed to find different examples
        """

        if combination_labels[l] == "extension_tasks" and metric_name == "muscle_shapes":
            min_val = np.argmax(task_metric)
            condition_val = min_val % 32
            min_val /= 32
            min_val = math.floor(min_val)
            print(f"ext pair with highest muscle shape is: {combination[min_val]}, {condition_val}")

        combination_data[combination_labels[l]] = task_metric
        combination_means.append(sum(task_metric) / len(task_metric))
        combination_stds.append(np.std(task_metric, ddof=1))

    # Convert values to list
    data_values = list(combination_data.values())
    labels = list(combination_data.keys())
    
    ax.axhline(combination_means[-1], color="dimgrey", linestyle="dashed")
    parts = ax.violinplot(data_values[:-1], showmeans=True)

    # Custom colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(combination_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(1.2)
    parts['cbars'].set_edgecolor('black')
    parts['cmins'].set_edgecolor('black')
    parts['cmaxes'].set_edgecolor('black')
    parts['cmeans'].set_color('black')

    if "angles" in metric_name:
        plt.yticks([0, 1.5])
    elif "shapes" in metric_name:
        plt.yticks([0, 1])
    plt.xticks([])
    save_fig(os.path.join(exp_path, "movement", metric_name), eps=True)

    combination_list = list(combination_data.keys())
    pvalues(combination_list, combination_data, metric_name)

    
def _plot_scatter(all_combinations, combinations, combination_colors, metric1, metric2, exp_path, metric1_name, metric2_name):

    fig, ax = standard_2d_ax()

    task_metric1 = convert_motif_dict_to_list(all_combinations, metric1)
    task_metric2 = convert_motif_dict_to_list(all_combinations, metric2)

    metric1_list = np.array(task_metric1).reshape((-1, 1))
    metric2_list = np.array(task_metric2).reshape((-1, 1))

    regression = LinearRegression()
    regression.fit(metric1_list, metric2_list)
    print(f"R^2 {metric1_name} to {metric2_name}: ", regression.score(metric1_list, metric2_list))
    x = np.linspace(0, max(metric1_list))
    ax.plot(x, regression.coef_ * x + regression.intercept_, color="black")

    for c, combination in enumerate(combinations[:-1]):
        task_metric1_comb = convert_motif_dict_to_list(combination, metric1)
        task_metric2_comb = convert_motif_dict_to_list(combination, metric2)
        ax.scatter(task_metric1_comb, task_metric2_comb, s=100, alpha=0.25, color=combination_colors[c])
    save_fig(os.path.join(exp_path, "movement", f"{metric1_name} vs {metric2_name}"), eps=True)


def _trajectory_alignment(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/alignment"

    options = {"batch_size": 32, "reach_conds": np.tile(np.arange(0, 32, 1), int(32/32)), "speed_cond": 5}

    trial_data_h = {}
    trial_data_muscle = {}
    combinations_h = {}
    combinations_muscle = {}
    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=False)

        if env == "DlyFullReach" or env == "DlyFullCircleClk" or env == "DlyFullCircleCClk" or env == "DlyFigure8" or env == "DlyFigure8Inv":

            halfway = int((trial_data["epoch_bounds"]["movement"][0] + trial_data["epoch_bounds"]["movement"][1]) / 2)

            trial_data_h[env+"1"] = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]
            trial_data_muscle[env+"1"] = trial_data["muscle_acts"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]

            trial_data_h[env+"2"] = trial_data["h"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]
            trial_data_muscle[env+"2"] = trial_data["muscle_acts"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]
        
        else:

            trial_data_h[env] = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            trial_data_muscle[env] = trial_data["muscle_acts"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]

    # Get all unique pairs of unit activity across tasks
    combination_labels = list(itertools.combinations(trial_data_h, 2))
    for combination_label in combination_labels:
        combinations_h[combination_label] = (
            trial_data_h[combination_label[0]],
            trial_data_h[combination_label[1]]
        )
        combinations_muscle[combination_label] = (
            trial_data_muscle[combination_label[0]],
            trial_data_muscle[combination_label[1]]
        )
    
    print("Computing Distances...")
    distances_h = distances_from_combinations(combinations_h, options["batch_size"])
    distances_muscle = distances_from_combinations(combinations_muscle, options["batch_size"])

    print("Computing Angles...")
    angles_h = angles_from_combinations(combinations_h, options["batch_size"])
    angles_muscle = shapes_from_combinations(combinations_muscle, options["batch_size"])

    print("Computing Shapes...")
    shapes_h = shapes_from_combinations(combinations_h, options["batch_size"])
    shapes_muscle = shapes_from_combinations(combinations_muscle, options["batch_size"])

    all_subsets = [
        subset_tasks,
        retraction_tasks,
        extension_tasks, 
        extension_retraction_tasks,
        combination_labels
    ]

    all_subset_labels = [
        "subset_tasks",
        "retraction_tasks",
        "extension_tasks",
        "extension_retraction_tasks",
        "all_tasks"
    ]

    all_subset_colors = ["purple", "pink", "blue", "orange", "grey"]

    all_metrics = {
        "neural_distances": distances_h,
        "muscle_distances": distances_muscle,
        "neural_angles": angles_h,
        "muscle_angles": angles_muscle,
        "neural_shapes": shapes_h,
        "muscle_shapes": shapes_muscle,
    }

    # Make each bar plot
    for metric in all_metrics:
        _plot_bar(all_subsets, all_metrics[metric], exp_path, metric, all_subset_labels, all_subset_colors)

    # -------------------------------------- NEURAL AND MUSCLE SHAPE DISTRIBUTIONS

    fig, ax = standard_2d_ax()
    all_shapes_h = convert_motif_dict_to_list(combination_labels, shapes_h)
    all_shapes_muscle = convert_motif_dict_to_list(combination_labels, shapes_muscle)

    bins = np.linspace(0, 1, 15)
    weights_data_h = np.ones_like(all_shapes_h) / len(all_shapes_h)
    weights_data_muscle = np.ones_like(all_shapes_muscle) / len(all_shapes_muscle)
    plt.hist(all_shapes_h, color="blue", alpha=0.5, bins=bins, weights=weights_data_h)
    plt.hist(all_shapes_muscle, color="purple", alpha=0.5, bins=bins, weights=weights_data_muscle)
    plt.axvline(sum(all_shapes_h)/len(all_shapes_h), color="blue", linestyle="dashed", linewidth=2)
    plt.axvline(sum(all_shapes_muscle)/len(all_shapes_muscle), color="purple", linestyle="dashed", linewidth=2)
    plt.xlim([0, 1])
    save_fig(os.path.join(exp_path, "movement", "neural_muscle_shape_dists"), eps=True)

    # ------------------------------------------------------------- ANGLE DISTRIBUTIONS

    fig, ax = standard_2d_ax()
    angle_h_dist = convert_motif_dict_to_list(combination_labels, angles_h)
    angle_muscle_dist = convert_motif_dict_to_list(combination_labels, angles_muscle)

    bins = np.linspace(0, 1.5, 15)
    weights_data_h = np.ones_like(angle_h_dist) / len(angle_h_dist)
    weights_data_muscle = np.ones_like(angle_muscle_dist) / len(angle_muscle_dist)
    plt.hist(angle_h_dist, color="blue", alpha=0.5, bins=bins, weights=weights_data_h)
    plt.hist(angle_muscle_dist, color="purple", alpha=0.5, bins=bins, weights=weights_data_muscle)
    plt.axvline(sum(angle_h_dist)/len(angle_h_dist), color="blue", linestyle="dashed", linewidth=2)
    plt.axvline(sum(angle_muscle_dist)/len(angle_muscle_dist), color="purple", linestyle="dashed", linewidth=2)
    plt.xlim([0, 1.5])
    save_fig(os.path.join(exp_path, "movement", "neural_muscle_angle_dists"), eps=True)

    # ------------------------------------------------------------- SVM DECODING

    metric_scores = {}
    for metric_label, metric in all_metrics.items():
        # Get the data for subsets into list
        subset_vals = np.array(convert_motif_dict_to_list(subset_tasks, metric))
        extension_vals = np.array(convert_motif_dict_to_list(extension_tasks, metric))
        retraction_vals = np.array(convert_motif_dict_to_list(retraction_tasks, metric))
        extension_retraction_vals = np.array(convert_motif_dict_to_list(extension_retraction_tasks, metric))

        # labels for each category
        subset_labels = 0*np.ones((subset_vals.shape[0], 1))
        extension_labels = 1*np.ones((extension_vals.shape[0], 1))
        retraction_labels = 2*np.ones((retraction_vals.shape[0], 1))
        extension_retraction_labels = 3*np.ones((extension_retraction_vals.shape[0], 1))

        X = np.concatenate([subset_vals, extension_vals, retraction_vals, extension_retraction_vals])
        X = np.expand_dims(X, axis=1)
        y = np.concatenate([subset_labels, extension_labels, retraction_labels, extension_retraction_labels])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        svm = sklearn.svm.SVC(kernel="linear")
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        metric_scores[metric_label] = score
    
    x_labels = [1, 2]
    fig, ax = standard_2d_ax()
    ax.plot(x_labels, [metric_scores["neural_angles"], metric_scores["neural_shapes"]], linewidth=4, color="black")
    ax.scatter(x_labels, [metric_scores["neural_angles"], metric_scores["neural_shapes"]], s=250, color="blue", zorder=10)
    ax.plot(x_labels, [metric_scores["muscle_angles"], metric_scores["muscle_shapes"]], linewidth=4, color="black")
    ax.scatter(x_labels, [metric_scores["muscle_angles"], metric_scores["muscle_shapes"]], s=250, color="purple", zorder=10)
    ax.set_xticks([0, 1, 2, 3], [" ", "Angular \n Distance", "Disparity", " "])
    save_fig(os.path.join(exp_path, "decoding"), eps=True)

    # -------------------------------------- SCATTER PLOTS

    for idx1, metric1 in enumerate(all_metrics):
        for idx2, metric2 in enumerate(all_metrics):
            if idx1 != idx2:
                _plot_scatter(combination_labels, all_subsets, all_subset_colors, all_metrics[metric1], all_metrics[metric2], exp_path, metric1, metric2)





def trajectory_alignment_movement(model_name):
    _trajectory_alignment(model_name)




def _get_vaf_combination(combination_labels, data, mode, comp_range):

    # Initialize the full pc dict
    all_vaf_list_means = []
    all_vaf_list_stds = []

    condition_tuple_dict = {}
    condition_label_dict = {}

    for combination in combination_labels:
        if combination not in data:
            combination = (combination[1], combination[0])
        for c, (task1_condition, task2_condition) in enumerate(zip(data[combination][0], data[combination][1])):
            if c not in condition_tuple_dict:
                condition_tuple_dict[c] = []
                condition_label_dict[c] = []
            # This should return a list of the variance explained for each pc
            condition_tuple_dict[c].append((task1_condition, task2_condition))
            condition_label_dict[c].append(combination)

    for pc in range(1, comp_range):
        condition_vaf_list = []
        for condition in condition_tuple_dict.values():
            # This should return a list of the variance explained for each pc
            ratio_list = vaf_ratio(condition, mode=mode, num_comps=pc, control=False)
            condition_vaf_list.extend(ratio_list)
        all_vaf_list_means.append(np.array(condition_vaf_list).mean())
        all_vaf_list_stds.append(np.array(condition_vaf_list).std())
    
    return np.array(all_vaf_list_means), np.array(all_vaf_list_stds)


def task_vaf_ratio(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/task_vaf_ratio"

    options = {"batch_size": 32, "reach_conds": np.tile(np.arange(0, 32, 1), int(32/32)), "speed_cond": 5}

    trial_data_h = {}
    trial_data_muscle = {}
    combinations_h = {}
    combinations_muscle = {}

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=False)

        if env == "DlyFullReach" or env == "DlyFullCircleClk" or env == "DlyFullCircleCClk" or env == "DlyFigure8" or env == "DlyFigure8Inv":

            halfway = int((trial_data["epoch_bounds"]["movement"][0] + trial_data["epoch_bounds"]["movement"][1]) / 2)

            trial_data_h[env+"1"] = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]
            trial_data_h[env+"2"] = trial_data["h"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]

            trial_data_muscle[env+"1"] = trial_data["muscle_acts"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]
            trial_data_muscle[env+"2"] = trial_data["muscle_acts"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]
        
        else:

            trial_data_h[env] = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            trial_data_muscle[env] = trial_data["muscle_acts"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
        

    # Get all unique pairs of unit activity across tasks
    combination_labels = list(itertools.combinations(trial_data_h, 2))
    for combination_label in combination_labels:
        combinations_h[combination_label] = (
            trial_data_h[combination_label[0]],
            trial_data_h[combination_label[1]]
        )
        combinations_muscle[combination_label] = (
            trial_data_muscle[combination_label[0]],
            trial_data_muscle[combination_label[1]]
        )

    all_subsets = {
        "extension_tasks": extension_tasks, 
        "retraction_tasks": retraction_tasks, 
        "subset_tasks": subset_tasks, 
        "extension_retraction_tasks": extension_retraction_tasks, 
    }

    all_subsets_colors = {
        "extension_tasks": "blue",
        "retraction_tasks": "pink",
        "subset_tasks": "purple",
        "extension_retraction_tasks": "orange"
    }

    comp_range = 11

    # Plotting vaf for different number of pc components neural

    subset_pc_dict_means = {}
    subset_pc_dict_stds = {}
    for subset in all_subsets:
        subset_pc_dict_means[subset], subset_pc_dict_stds[subset] = _get_vaf_combination(all_subsets[subset], combinations_h, "h", comp_range)
    all_task_pc_means, all_task_pc_stds = _get_vaf_combination(combination_labels, combinations_h, "h", comp_range)
    
    fig, ax = standard_2d_ax()
    x = np.arange(1, comp_range)
    for subset in subset_pc_dict_means:
        ax.plot(x, subset_pc_dict_means[subset], linewidth=4, alpha=0.75, color=all_subsets_colors[subset])
        ax.fill_between(
            x, 
            subset_pc_dict_means[subset] - subset_pc_dict_stds[subset], 
            subset_pc_dict_means[subset] + subset_pc_dict_stds[subset], 
            color=all_subsets_colors[subset], 
            alpha=0.25
        )
    ax.plot(x, all_task_pc_means, linewidth=4, alpha=0.75, color="grey")
    ax.fill_between(
        x, 
        all_task_pc_means - all_task_pc_stds, 
        all_task_pc_means + all_task_pc_stds, 
        color="grey", 
        alpha=0.25
    )

    comp_range = 7
    ax.set_ylim([0, 1.1])
    save_fig(os.path.join(exp_path, "vaf_ratio_neural"), eps=True)

    # Plotting vaf for different number of pc components muscle
    subset_pc_dict_means = {}
    subset_pc_dict_stds = {}
    for subset in all_subsets:
        subset_pc_dict_means[subset], subset_pc_dict_stds[subset] = _get_vaf_combination(all_subsets[subset], combinations_muscle, "muscle_acts", comp_range)
    all_task_pc_means, all_task_pc_stds = _get_vaf_combination(combination_labels, combinations_muscle, "muscle_acts", comp_range)
    
    fig, ax = standard_2d_ax()
    x = np.arange(1, comp_range)
    for subset in subset_pc_dict_means:
        ax.plot(x, subset_pc_dict_means[subset], linewidth=4, alpha=0.75, color=all_subsets_colors[subset])
        ax.fill_between(
            x, 
            subset_pc_dict_means[subset] - subset_pc_dict_stds[subset], 
            subset_pc_dict_means[subset] + subset_pc_dict_stds[subset], 
            color=all_subsets_colors[subset], 
            alpha=0.25
        )
    ax.plot(x, all_task_pc_means, linewidth=4, alpha=0.75, color="grey")
    ax.fill_between(
        x, 
        all_task_pc_means - all_task_pc_stds, 
        all_task_pc_means + all_task_pc_stds, 
        color="grey", 
        alpha=0.25
    )

    ax.set_ylim([0, 1.1])
    save_fig(os.path.join(exp_path, "vaf_ratio_muscle"), eps=True)




def dsa_similarity_matrix(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/dsa"

    options = {"batch_size": 32*4, "reach_conds": np.tile(np.arange(0, 32, 1), 4), "speed_cond": 5}

    trial_data_h = []
    trial_data_colors = []

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=True)

        if env == "DlyFullReach" or env == "DlyFullCircleClk" or env == "DlyFullCircleCClk" or env == "DlyFigure8" or env == "DlyFigure8Inv":

            halfway = int((trial_data["epoch_bounds"]["movement"][0] + trial_data["epoch_bounds"]["movement"][1]) / 2)
            extend = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]
            retract = trial_data["h"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]

            pca_extend = PCA(n_components=12)
            extend_reduced = pca_extend.fit_transform(extend.reshape((-1, extend.shape[-1])))
            extend_reduced = extend_reduced.reshape((extend.shape[0], extend.shape[1], 12))

            pca_retract = PCA(n_components=12)
            retract_reduced = pca_retract.fit_transform(retract.reshape((-1, retract.shape[-1])))
            retract_reduced = retract_reduced.reshape((retract.shape[0], retract.shape[1], 12))

            trial_data_h.append(extend_reduced)
            trial_data_colors.append("pink")

            trial_data_h.append(retract_reduced)
            trial_data_colors.append("purple")
        
        else:

            extend = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            pca_extend = PCA(n_components=12)
            extend_reduced = pca_extend.fit_transform(extend.reshape((-1, extend.shape[-1])))
            extend_reduced = extend_reduced.reshape((extend.shape[0], extend.shape[1], 12))

            trial_data_h.append(extend_reduced)
            trial_data_colors.append("blue")

    # TODO play around with hyperparameters
    dsa = DSA(trial_data_h, n_delays=90, rank=150, verbose=True, score_method="euclidean", device="cpu")
    similarities = dsa.fit_score()

    dsa_data = {"similarities": similarities, "colors": trial_data_colors}

    with open(os.path.join(model_path, "dsa_similarity.txt"), 'wb') as f:
        pickle.dump(dsa_data, f)
    
    dsa_scatter(model_name)


def dsa_scatter(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    fig, ax = standard_2d_ax()

    dsa_data = load_pickle(os.path.join(model_path, "dsa_similarity.txt"))
    similarities = dsa_data["similarities"]
    colors = dsa_data["colors"]

    reduced = PCA(n_components=2).fit_transform(similarities)
    ax.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.75, s=250)
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_dsa_scatter"), eps=True)


def dsa_heatmap(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'

    dsa_data = load_pickle(os.path.join(model_path, "dsa_similarity.txt"))
    similarities = dsa_data["similarities"]

    # Reorder indices
    indices_extension = [0, 1, 2, 3, 4]
    indices_extension_long = [5, 7, 9, 11, 13]
    indices_retraction = [6, 8, 10, 12, 14]

    # full reordering index
    new_order = indices_extension + indices_extension_long + indices_retraction

    # reorder rows and columns at once
    re_similarity = similarities[np.ix_(new_order, new_order)]

    sns.heatmap(re_similarity, cmap="Purples")
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_dsa_similarity_vis"), eps=True)




def procrustes_similarity_matrix(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/dsa"

    options = {"batch_size": 32*4, "reach_conds": np.tile(np.arange(0, 32, 1), 4), "speed_cond": 5}

    trial_data_h = []
    trial_data_colors = []

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=True)

        if env == "DlyFullReach" or env == "DlyFullCircleClk" or env == "DlyFullCircleCClk" or env == "DlyFigure8" or env == "DlyFigure8Inv":

            halfway = int((trial_data["epoch_bounds"]["movement"][0] + trial_data["epoch_bounds"]["movement"][1]) / 2)
            extend = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]
            retract = trial_data["h"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]

            pca_extend = PCA(n_components=10)
            extend_reduced = pca_extend.fit_transform(extend.reshape((-1, extend.shape[-1])))
            extend_reduced = extend_reduced.reshape((extend.shape[0], extend.shape[1], 10))

            pca_retract = PCA(n_components=10)
            retract_reduced = pca_retract.fit_transform(retract.reshape((-1, retract.shape[-1])))
            retract_reduced = retract_reduced.reshape((retract.shape[0], retract.shape[1], 10))

            trial_data_h.append(extend_reduced.mean(axis=0))
            trial_data_colors.append("pink")

            trial_data_h.append(retract_reduced.mean(axis=0))
            trial_data_colors.append("purple")
        
        else:

            extend = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            pca_extend = PCA(n_components=10)
            extend_reduced = pca_extend.fit_transform(extend.reshape((-1, extend.shape[-1])))
            extend_reduced = extend_reduced.reshape((extend.shape[0], extend.shape[1], 10))

            trial_data_h.append(extend_reduced.mean(axis=0))
            trial_data_colors.append("blue")

    rows = []
    for task1 in trial_data_h:
        cols = []
        for task2 in trial_data_h:
            _, _, disparity = scipy.spatial.procrustes(task1, task2)
            cols.append(np.sqrt(disparity))
        rows.append(cols)
    similarities = np.array(rows)

    procrustes_data = {"similarities": similarities, "colors": trial_data_colors}

    with open(os.path.join(model_path, "procrustes_similarity.txt"), 'wb') as f:
        pickle.dump(procrustes_data, f)
    
    procrustes_scatter(model_name)


def procrustes_scatter(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    fig, ax = standard_2d_ax()

    procrustes_data = load_pickle(os.path.join(model_path, "procrustes_similarity.txt"))
    similarities = procrustes_data["similarities"]
    colors = procrustes_data["colors"]

    reduced = PCA(n_components=2).fit_transform(similarities)
    ax.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.75, s=250)
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_procrustes_scatter"), eps=True)


def procrustes_heatmap(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'

    procrustes_data = load_pickle(os.path.join(model_path, "procrustes_similarity.txt"))
    similarities = procrustes_data["similarities"]

    # Reorder indices
    indices_extension = [0, 1, 2, 3, 4]
    indices_extension_long = [5, 7, 9, 11, 13]
    indices_retraction = [6, 8, 10, 12, 14]

    # full reordering index
    new_order = indices_extension + indices_extension_long + indices_retraction

    # reorder rows and columns at once
    re_similarity = similarities[np.ix_(new_order, new_order)]

    sns.heatmap(re_similarity, cmap="Purples")
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_procrustes_similarity_vis"), eps=True)




def task_similarity_classification(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    def silhouette_scores(data, labels, num_clusters=3):
        silhouette_values = sklearn.metrics.silhouette_samples(data, labels)
        means_lst = []
        for label in range(num_clusters):
            means_lst.append(silhouette_values[labels == label].mean())
        return means_lst

    dsa_data = load_pickle(os.path.join(model_path, "dsa_similarity.txt"))
    dsa_similarities = dsa_data["similarities"]
    pca = PCA(n_components=2)
    dsa_similarities = pca.fit_transform(dsa_similarities)
    labels_dsa = np.ones(shape=(15,))
    for i, color in enumerate(dsa_data["colors"]):
        if color == "blue":
            labels_dsa[i] = 0
        elif color == "pink":
            labels_dsa[i] = 1
        elif color == "purple":
            labels_dsa[i] = 2
        
    means_dsa = silhouette_scores(dsa_similarities, labels_dsa, 3)

    procrustes_data = load_pickle(os.path.join(model_path, "procrustes_similarity.txt"))
    procrustes_similarities = procrustes_data["similarities"]
    pca = PCA(n_components=2)
    procrustes_similarities = pca.fit_transform(procrustes_similarities)
    labels_procrustes = np.ones(shape=(15,))
    for i, color in enumerate(procrustes_data["colors"]):
        if color == "blue":
            labels_procrustes[i] = 0
        elif color == "pink":
            labels_procrustes[i] = 1
        elif color == "purple":
            labels_procrustes[i] = 2

    means_pro = silhouette_scores(procrustes_similarities, labels_dsa, 3)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    x = ["Ext.", "Ext. (long)", "Ret."]
    plt.bar(x, means_dsa, color=["blue", "pink", "purple"], capsize=10, edgecolor="black", alpha=0.75)
    plt.xticks([])
    save_fig(os.path.join(exp_path, f"dsa_silhouette_bar"), eps=True)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    x = ["Ext.", "Ext. (long)", "Ret."]
    plt.bar(x, means_pro, color=["blue", "pink", "purple"], capsize=10, edgecolor="black", alpha=0.75)
    plt.xticks([])
    save_fig(os.path.join(exp_path, f"procrustes_silhouette_bar"), eps=True)




# Get the loss from every composite input on each environment and get the heat map
def composite_rule_input_heat_map(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/composite_rule_inputs/heat_map"
    load_name = f"checkpoints/{model_name}/composite_rule_inputs.pkl"
    trial_data = load_pickle(load_name)

    for env in env_dict_ext:
        fig, ax = no_ticks_2d_ax()
        rule_input = trial_data[env]["rule_input"][:, :5].numpy()
        im = ax.imshow(rule_input, cmap="RdBu", vmin=2, vmax=-2)
        cbar = fig.colorbar(im, ax=ax, fraction=0.07, pad=0.04)
        fig.tight_layout()
        save_fig(os.path.join(exp_path, f"extension_heat_map_{env}"), eps=True)
    



# Get the loss from every composite input on each environment and get the heat map
def composite_rule_input_kinematics(model_name):

    def plot_env_kinematics(xy):
        fig, ax = empty_2d_ax()
        for i, batch in enumerate(xy):
            ax.plot(batch[:, 0], batch[:, 1], linewidth=4, color=colors[i])
            ax.scatter(batch[0, 0], batch[0, 1], s=250, marker="^", color=colors[i])
            ax.scatter(batch[-1, 0], batch[-1, 1], s=250, marker="X", color=colors[i])

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/composite_rule_inputs/kinematics"
    load_name = f"checkpoints/{model_name}/composite_rule_inputs.pkl"
    trial_data = load_pickle(load_name)
    colors = plt.cm.inferno(np.linspace(0, 1, 8)) 

    for env in env_dict_ext:
        plot_env_kinematics(trial_data[env]["xy"])
        save_fig(os.path.join(exp_path, f"extension_kinematics_{env}"), eps=True)





def composite_input_init(model_name):

    exp_path = f"results/{model_name}/compositionality/composite_rule_inputs/init_cond"
    load_name = f"checkpoints/{model_name}/composite_rule_inputs.pkl"
    trial_data = load_pickle(load_name)
    colors_envs = plt.cm.tab10(np.linspace(0, 1, len(env_dict))) 

    env_hs, _ = _get_mean_act(model_name, "delay", "extension", delay_cond=2, batch_size=32)
    env_hs = np.concatenate(env_hs)
    epoch_pca = PCA(n_components=3)
    epoch_pca.fit(env_hs)

    fig, ax = ax_3d_no_grid()
    for e, env in enumerate(env_dict_ext):

        composite_h = trial_data[env]["h"][:, trial_data[env]["epoch_bounds"]["delay"][0]:trial_data[env]["epoch_bounds"]["delay"][1]]
        composite_h = composite_h.mean(dim=0)

        all_data_for_min = np.concatenate([composite_h, env_hs])
        red_all_data_for_min = epoch_pca.transform(all_data_for_min)
        min_val = np.min(red_all_data_for_min)

        reduced_baseline = epoch_pca.transform(env_hs)

        ax.scatter(reduced_baseline[e, 0], reduced_baseline[e, 1], reduced_baseline[e, 2], s=200, marker="o", color=colors_envs[e])
        ax.scatter(reduced_baseline[e, 0], reduced_baseline[e, 1], min_val, s=200, marker="o", color=colors_envs[e], alpha=0.10)

        reduced_composite = epoch_pca.transform(composite_h)
        #ax.plot(reduced_composite[:, 0], reduced_composite[:, 1], reduced_composite[:, 2], linewidth=4, color=colors_envs[e], alpha=0.5)
        ax.scatter(reduced_composite[-1, 0], reduced_composite[-1, 1], min_val, s=200,  marker="X", color=colors_envs[e], alpha=0.10)
        #ax.scatter(reduced_composite[0, 0], reduced_composite[0, 1], reduced_composite[0, 2], s=100, marker="^", color=colors_envs[e])
        ax.scatter(reduced_composite[-1, 0], reduced_composite[-1, 1], reduced_composite[-1, 2], s=200, marker="X", color=colors_envs[e])

    save_fig(os.path.join(exp_path, f"extension_init_all"), eps=True)





def composite_input_loss(model_name):

    exp_path = f"results/{model_name}/compositionality/composite_rule_inputs/losses"
    load_name = f"checkpoints/{model_name}/composite_rule_inputs.pkl"
    trial_data = load_pickle(load_name)
    colors_envs = plt.cm.tab10(np.linspace(0, 1, len(env_dict))) 

    fig, ax = standard_2d_ax()
    for e, env in enumerate(trial_data):
        loss = trial_data[env]["test_loss"]
        ax.plot(loss, linewidth=4, color=colors_envs[e], alpha=0.75)
    save_fig(os.path.join(exp_path, f"optimization_losses"), eps=True)




def run_composite_input_optimization(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    options = {"batch_size": 8, "reach_conds": np.arange(0, 32, 4), "speed_cond": 9, "custom_delay": 150}

    all_trial_data = {}
    for env in env_dict_ext:
        trial_data = composite_input_optimization(
            model_path, 
            model_file, 
            options,
            env_dict_ext[env],
            env)
        all_trial_data[env] = trial_data

    # Save all information of inputs across envs
    save_name = f'composite_rule_inputs.pkl'
    fname = os.path.join(model_path, save_name)
    with open(fname, 'wb') as f:
        pickle.dump(all_trial_data, f)




def _sequential_rule_inputs(model_name, extension, retraction):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/sequential_rule_inputs/kinematics"
    colors_conds = plt.cm.inferno(np.linspace(0, 1, 8)) 
    options = {"batch_size": 4, "reach_conds": np.arange(0, 32, 8), "speed_cond": 5, "delay_cond": 2}

    extension_env = env_dict[extension]
    retraction_env = env_dict[retraction]

    trial_data = test_sequential_inputs(model_path, model_file, options, extension_env, retraction_env)
    kinematics = trial_data["xy"]
    middle_movement = get_middle_movement(trial_data)

    fig, ax = empty_2d_ax()
    for cond in range(kinematics.shape[0]):
        ax.plot(kinematics[cond, :middle_movement, 0], kinematics[cond, :middle_movement, 1], linewidth=4, color=colors_conds[cond], alpha=0.25, linestyle="dashed")
        ax.plot(kinematics[cond, middle_movement:, 0], kinematics[cond, middle_movement:, 1], linewidth=4, color=colors_conds[cond])
        ax.scatter(kinematics[cond, 0, 0], kinematics[cond, 0, 1], s=100, marker="^", color=colors_conds[cond])
        ax.scatter(kinematics[cond, -1, 0], kinematics[cond, -1, 1], s=100, marker="x", color=colors_conds[cond])
    save_fig(os.path.join(exp_path, f"{extension}_{retraction}"), eps=True)




def run_all_sequential_rule_inputs(model_name):
    for ext in env_dict_ext:
        for ret in env_dict_ret:
            _sequential_rule_inputs(model_name, ext, ret)





if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    # --------------------------------------------------------- COMPUTE INTERPOLATED FPS

    if args.experiment == "run_all_compute_interpolated_fps":
        run_all_compute_interpolated_fps(args.model_name)

    # --------------------------------------------------------- PLOT INTERPOLATED FPS

    elif args.experiment == "run_all_plot_interpolated_fps":
        run_all_plot_interpolated_fps(args.model_name)

    elif args.experiment == "trajectory_alignment_movement":
        trajectory_alignment_movement(args.model_name)

    # Epoch pcs
    elif args.experiment == "stable_pcs_extension":
        stable_pcs_extension(args.model_name)
    elif args.experiment == "delay_pcs_extension":
        delay_pcs_extension(args.model_name)
    elif args.experiment == "movement_pcs_extension":
        movement_pcs_extension(args.model_name)
    elif args.experiment == "hold_pcs_extension":
        hold_pcs_extension(args.model_name)

    elif args.experiment == "stable_pcs_extension_retraction":
        stable_pcs_extension_retraction(args.model_name)
    elif args.experiment == "delay_pcs_extension_retraction":
        delay_pcs_extension_retraction(args.model_name)
    elif args.experiment == "extension_pcs_extension_retraction":
        extension_pcs_extension_retraction(args.model_name)
    elif args.experiment == "retraction_pcs_extension_retraction":
        retraction_pcs_extension_retraction(args.model_name)
    elif args.experiment == "hold_pcs_extension_retraction":
        hold_pcs_extension_retraction(args.model_name)
    

    elif args.experiment == "neural_two_task_pcs_sinusoid_fullcircleclk":
        neural_two_task_pcs_sinusoid_fullcircleclk(args.model_name)
    elif args.experiment == "muscle_two_task_pcs_sinusoid_fullcircleclk":
        muscle_two_task_pcs_sinusoid_fullcircleclk(args.model_name)
    elif args.experiment == "neural_two_task_pcs_halfcircleclk_halfcirclecclk":
        neural_two_task_pcs_halfcircleclk_halfcirclecclk(args.model_name)
    elif args.experiment == "muscle_two_task_pcs_halfcircleclk_halfcirclecclk":
        muscle_two_task_pcs_halfcircleclk_halfcirclecclk(args.model_name)

    elif args.experiment == "run_all_epoch_pcs":
        run_all_epoch_pcs(args.model_name)
    elif args.experiment == "run_all_epoch_pcs_transfer":
        run_all_epoch_pcs_transfer(args.model_name)
    elif args.experiment == "run_all_epoch_lda":
        run_all_epoch_lda(args.model_name)
    elif args.experiment == "run_all_epoch_lda_transfer":
        run_all_epoch_lda_transfer(args.model_name)


    elif args.experiment == "task_vaf_ratio":
        task_vaf_ratio(args.model_name)

    elif args.experiment == "dsa_similarity_matrix":
        dsa_similarity_matrix(args.model_name)
    elif args.experiment == "dsa_scatter":
        dsa_scatter(args.model_name)
    elif args.experiment == "dsa_heatmap":
        dsa_heatmap(args.model_name)

    elif args.experiment == "procrustes_similarity_matrix":
        procrustes_similarity_matrix(args.model_name)
    elif args.experiment == "procrustes_scatter":
        procrustes_scatter(args.model_name)
    elif args.experiment == "procrustes_heatmap":
        procrustes_heatmap(args.model_name)
    
    elif args.experiment == "task_similarity_classification":
        task_similarity_classification(args.model_name)
    
    elif args.experiment == "composite_rule_input_heat_map":
        composite_rule_input_heat_map(args.model_name)
    elif args.experiment == "composite_rule_input_kinematics":
        composite_rule_input_kinematics(args.model_name)
    elif args.experiment == "composite_input_init":
        composite_input_init(args.model_name)
    elif args.experiment == "composite_input_loss":
        composite_input_loss(args.model_name)
    elif args.experiment == "run_composite_input_optimization":
        run_composite_input_optimization(args.model_name)
    
    elif args.experiment == "run_all_sequential_rule_inputs":
        run_all_sequential_rule_inputs(args.model_name)

    else:
        raise ValueError("Experiment not in this file")