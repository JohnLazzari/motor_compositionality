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
import matplotlib.patches as mpatches
from exp_utils import _test, env_dict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import seaborn as sns
from DSA import DSA
import scipy
from utils import interpolate_trial
import pandas as pd
from compositionality import _gather_angles, _gather_shapes, _convert_motif_dict_to_list, _interpolated_fps, _plot_interpolated_fps

plt.rcParams.update({'font.size': 18})  # Sets default font size for all text

cclkhalfcircle = ["DlyHalfCircleCClk"]
cclkfullcircle = ["DlyFullCircleCClk2"]

# These are included in task pairings
extension_movements_half_cclkhalfcircle = [
    "DlyHalfReach",
    "DlyHalfCircleClk",
    "DlySinusoid",
    "DlySinusoidInv"
]

extension_movements_full_cclkhalfcircle = [
    "DlyFullReach1",
    "DlyFullCircleClk1",
    "DlyFullCircleCClk1",
    "DlyFigure81",
    "DlyFigure8Inv1"
]

retraction_movements_full_cclkhalfcircle = [
    "DlyFullReach2",
    "DlyFullCircleClk2",
    "DlyFullCircleCClk2",
    "DlyFigure82",
    "DlyFigure8Inv2"
]

# InvSinusoid combinations
cclkhalfcircle_extension1 = list(itertools.product(extension_movements_half_cclkhalfcircle, cclkhalfcircle))
cclkhalfcircle_extension2 = list(itertools.product(extension_movements_full_cclkhalfcircle, cclkhalfcircle))
cclkhalfcircle_extension = [*cclkhalfcircle_extension1, *cclkhalfcircle_extension2]
cclkhalfcircle_retraction = list(itertools.product(retraction_movements_full_cclkhalfcircle, cclkhalfcircle))

# These are included in task pairings
extension_movements_half_cclkfullcircle = [
    "DlyHalfReach",
    "DlyHalfCircleClk",
    "DlyHalfCircleCClk",
    "DlySinusoid",
    "DlySinusoidInv"
]

extension_movements_full_cclkfullcircle = [
    "DlyFullReach1",
    "DlyFullCircleClk1",
    "DlyFullCircleCClk1",
    "DlyFigure81",
    "DlyFigure8Inv1"
]

retraction_movements_full_cclkfullcircle = [
    "DlyFullReach2",
    "DlyFullCircleClk2",
    "DlyFigure82",
    "DlyFigure8Inv2",
]


# InvFigure8 combinations
cclkfullcircle_extension1 = list(itertools.product(extension_movements_half_cclkfullcircle, cclkfullcircle))
cclkfullcircle_extension2 = list(itertools.product(extension_movements_full_cclkfullcircle, cclkfullcircle))
cclkfullcircle_extension = [*cclkfullcircle_extension1, *cclkfullcircle_extension2]
cclkfullcircle_retraction = list(itertools.product(retraction_movements_full_cclkfullcircle, cclkfullcircle))


# Movement period with different input interpolations
def compute_interpolated_fps_cclkhalfcircle_cclkfullcircle1_movement(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleCClk", "DlyFullCircleCClk", "movement", 
        task1_period="all", task2_period="first", input_component=None, add_new_rule_inputs=True, num_new_inputs=10)
def compute_interpolated_fps_cclkhalfcircle_cclkfullcircle2_movement(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleCClk", "DlyFullCircleCClk", "movement", 
        task1_period="all", task2_period="second", input_component=None, add_new_rule_inputs=True, num_new_inputs=10)

def run_all_compute_interpolated_fps(model_name):
    compute_interpolated_fps_cclkhalfcircle_cclkfullcircle1_movement(model_name)
    compute_interpolated_fps_cclkhalfcircle_cclkfullcircle2_movement(model_name)

def plot_interpolated_fps_cclkhalfcircle_cclkfullcircle1_movement(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleCClk", "DlyFullCircleCClk", "movement", 
        task1_period="all", task2_period="first", input_component=None, add_new_rule_inputs=True, num_new_inputs=10, y_dist=10)
def plot_interpolated_fps_cclkhalfcircle_cclkfullcircle2_movement(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleCClk", "DlyFullCircleCClk", "movement", 
        task1_period="all", task2_period="second", input_component=None, add_new_rule_inputs=True, num_new_inputs=10, y_dist=10)

def run_all_plot_interpolated_fps(model_name):
    plot_interpolated_fps_cclkhalfcircle_cclkfullcircle1_movement(model_name)
    plot_interpolated_fps_cclkhalfcircle_cclkfullcircle2_movement(model_name)

def _plot_bar(combinations, metric, exp_path, metric_name, combination_labels, combination_colors, heldout_task):

    # Create figure and 3D axes
    fig = plt.figure(figsize=(2, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    combination_means = []
    combination_stds = []
    combination_data = {}
    for l, combination in enumerate(combinations):
        task_metric = _convert_motif_dict_to_list(combination, metric)
        combination_data[combination_labels[l]] = task_metric
        combination_means.append(sum(task_metric) / len(task_metric))
        combination_stds.append(np.std(task_metric, ddof=1))
    plt.bar(combination_labels, combination_means, yerr=combination_stds, capsize=10, color=combination_colors, edgecolor='black')

    if "angles" in metric_name:
        plt.yticks([0, 1.5])
    elif "shapes" in metric_name:
        plt.yticks([0, 1])
    plt.xticks([])
    save_fig(os.path.join(exp_path, "movement", metric_name + f"_{heldout_task}"), eps=True)

    combination_labels = list(itertools.combinations(combination_data, 2))
    # Print out significance here
    for combination in combination_labels:
        result = scipy.stats.mannwhitneyu(combination_data[combination[0]], combination_data[combination[1]])
        pvalue = result[1]
        if pvalue < 0.001:
            pvalue_str = "***"
        elif pvalue < 0.01:
            pvalue_str = "**"
        elif pvalue < 0.05:
            pvalue_str = "*"
        else:
            pvalue_str = " "
        print(f"pvalue for {combination[0]} and {combination[1]} in metric {metric_name} is: {pvalue_str}")


def _trajectory_alignment(model_name, all_subsets, all_subset_labels, all_subset_colors, heldout_task):

    model_path_heldout = f"checkpoints/rnn256_softplus_heldout"
    model_file_heldout = f"rnn256_softplus_heldout.pth"

    model_path_heldout_transfer = f"checkpoints/rnn256_softplus_heldout_transfer"
    model_file_heldout_transfer = f"rnn256_softplus_heldout_transfer.pth"

    exp_path = f"results/{model_name}/transfer_analysis"

    options = {"batch_size": 32, "reach_conds": np.tile(np.arange(0, 32, 1), int(32/32)), "speed_cond": 5}

    trial_data_h = {}
    combinations_h = {}
    for env in env_dict:

        if env == "DlyHalfCircleCClk" or env == "DlyFullCircleCClk":
            trial_data = _test(model_path_heldout_transfer, model_file_heldout_transfer, options, env=env_dict[env], noise=False, add_new_rule_inputs=True, num_new_inputs=10)
        else:
            trial_data = _test(model_path_heldout, model_file_heldout, options, env=env_dict[env], noise=False)

        if env == "DlyFullReach" or env == "DlyFullCircleClk" or env == "DlyFullCircleCClk" or env == "DlyFigure8" or env == "DlyFigure8Inv":

            halfway = int((trial_data["epoch_bounds"]["movement"][0] + trial_data["epoch_bounds"]["movement"][1]) / 2)
            trial_data_h[env+"1"] = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]
            trial_data_h[env+"2"] = trial_data["h"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]

        else:

            trial_data_h[env] = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]

    # Get all unique pairs of unit activity across tasks
    combination_labels = list(itertools.combinations(trial_data_h, 2))
    for combination_label in combination_labels:
        combinations_h[combination_label] = (
            trial_data_h[combination_label[0]],
            trial_data_h[combination_label[1]]
        )
    
    print("Computing Angles...")
    angles_h = _gather_angles(combinations_h, options["batch_size"])
    print("Computing Shapes...")
    shapes_h = _gather_shapes(combinations_h, options["batch_size"])

    all_metrics = {
        "neural_angles": angles_h,
        "neural_shapes": shapes_h,
    }

    # Make each bar plot
    for metric in all_metrics:
        _plot_bar(all_subsets, all_metrics[metric], exp_path, metric, all_subset_labels, all_subset_colors, heldout_task)


def trajectory_alignment_cclkhalfcircle(model_name):
    all_subsets = [cclkhalfcircle_extension, cclkhalfcircle_retraction]
    all_subset_labels = ["extension_tasks", "retraction_tasks"]
    all_subset_colors = ["blue", "pink"]
    _trajectory_alignment(model_name, all_subsets, all_subset_labels, all_subset_colors, "DlyHalfCircleCClk")

def trajectory_alignment_cclkfullcircle(model_name):
    all_subsets = [cclkfullcircle_extension, cclkfullcircle_retraction]
    all_subset_labels = ["extension_tasks", "retraction_tasks"]
    all_subset_colors = ["blue", "pink"]
    _trajectory_alignment(model_name, all_subsets, all_subset_labels, all_subset_colors, "DlyFullCircleCClk")



if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "trajectory_alignment_cclkfullcircle":
        trajectory_alignment_cclkfullcircle(args.model_name)
    elif args.experiment == "trajectory_alignment_cclkhalfcircle":
        trajectory_alignment_cclkhalfcircle(args.model_name)

    elif args.experiment == "compute_interpolated_fps_cclkhalfcircle_cclkfullcircle_movement":
        compute_interpolated_fps_cclkhalfcircle_cclkfullcircle_movement(args.model_name)
    elif args.experiment == "compute_interpolated_fps_cclkhalfcircle_cclkfullcircle_movement":
        compute_interpolated_fps_cclkhalfcircle_cclkfullcircle_movement(args.model_name)
    elif args.experiment == "run_all_compute_interpolated_fps":
        run_all_compute_interpolated_fps(args.model_name)

    elif args.experiment == "plot_interpolated_fps_cclkhalfcircle_cclkfullcircle_movement":
        plot_interpolated_fps_cclkhalfcircle_cclkfullcircle_movement(args.model_name)
    elif args.experiment == "plot_interpolated_fps_cclkhalfcircle_cclkfullcircle_movement":
        plot_interpolated_fps_cclkhalfcircle_cclkfullcircle_movement(args.model_name)
    elif args.experiment == "run_all_plot_interpolated_fps":
        run_all_plot_interpolated_fps(args.model_name)


    else:
        raise ValueError("Experiment not in this file")