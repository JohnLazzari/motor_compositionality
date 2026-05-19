import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import math
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tqdm as tqdm
from sklearn.decomposition import PCA
import sklearn
import matplotlib.patches as mpatches
from utils.exp_utils import (
    env_dict,
    pvalues,
    mov_bounds,
    delay_bounds,
    unique_pairs_dict,
    get_middle_movement,
    composite_input_optimization,
    distances_from_combinations,
    angles_from_combinations,
    shapes_from_combinations,
    convert_motif_dict_to_list,
    test_sequential_inputs,
    load_pickle,
)
from utils.manifold_utils import compute_vaf_ratio
from modules.envs.reach import Reach
from modules.envs.clk_curved_reach import ClkCurvedReach
from modules.envs.cclk_curved_reach import CClkCurvedReach
from modules.envs.sinusoid import Sinusoid
from modules.envs.inv_sinusoid import InvSinusoid
from modules.envs.reach_back import ReachBack
from modules.envs.clk_cycle import ClkCycle
from modules.envs.cclk_cycle import CClkCycle
from modules.envs.figure_eight import Figure8
from modules.envs.inv_figure_eight import InvFigure8
import itertools
import seaborn as sns
from DSA import DSA
import scipy
from modules.test import Test
from utils.plot_utils import (
    save_fig,
    standard_2d_ax,
    ax_3d_no_grid,
    no_ticks_2d_ax,
    empty_2d_ax,
    create_dir,
)
from sklearn.linear_model import LinearRegression

plt.rcParams.update({"font.size": 18})  # Sets default font size for all text

# This is not included in a task pairing
full_movements = [
    "ReachBack",
    "ClkCycle",
    "CClkCycle",
    "Figure8",
    "InvFigure8",
]

env_dict_ext = {
    "Reach": Reach,
    "ClkCurvedReach": ClkCurvedReach,
    "CClkCurvedReach": CClkCurvedReach,
    "Sinusoid": Sinusoid,
    "InvSinusoid": InvSinusoid,
}

env_dict_ret = {
    "ReachBack": ReachBack,
    "ClkCycle": ClkCycle,
    "CClkCycle": CClkCycle,
    "Figure8": Figure8,
    "InvFigure8": InvFigure8,
}

# These are included in task pairings
extension_movements_half = [
    "Reach",
    "ClkCurvedReach",
    "CClkCurvedReach",
    "Sinusoid",
    "InvSinusoid",
]

extension_movements_full = [
    "ReachBack1",
    "ClkCycle1",
    "CClkCycle1",
    "Figure81",
    "InvFigure81",
]

retraction_movements_full = [
    "ReachBack2",
    "ClkCycle2",
    "CClkCycle2",
    "Figure82",
    "InvFigure82",
]

extension_half_combinations = list(itertools.combinations(extension_movements_half, 2))
extension_full_combinations = list(itertools.combinations(extension_movements_full, 2))
extension_tasks = [*extension_half_combinations, *extension_full_combinations]

retraction_full_combinations = list(
    itertools.combinations(retraction_movements_full, 2)
)
retraction_tasks = retraction_full_combinations

subset_tasks = [
    ("ClkCurvedReach", "ClkCycle1"),
    ("Sinusoid", "Figure81"),
    ("Reach", "ReachBack1"),
    ("CClkCurvedReach", "CClkCycle1"),
    ("InvSinusoid", "InvFigure81"),
]

rotated_tasks = [
    ("ClkCurvedReach", "CClkCurvedReach"),
    ("Sinusoid", "InvSinusoid"),
    ("ClkCycle1", "CClkCycle1"),
    ("ClkCycle2", "CClkCycle2"),
    ("Figure81", "InvFigure81"),
    ("Figure82", "InvFigure82"),
]

extension_retraction_tasks = list(
    itertools.product(extension_movements_half, retraction_movements_full)
)


def _get_mean_act(
    model_name,
    epoch,
    movement_type,
    speed_cond=5,
    delay_cond=1,
    batch_size=32,
    add_new_rule_inputs=False,
):
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
    test = Test(model_path, model_name, add_new_rule_inputs=add_new_rule_inputs)

    options = {
        "batch_size": batch_size,
        "reach_conds": np.arange(0, 32, int(32 / batch_size)),
        "speed_cond": speed_cond,
        "delay_cond": delay_cond,
    }

    env_hs = []

    if movement_type == "extension":
        envs_to_use = extension_movements_half.copy()
    elif movement_type == "extension_retraction":
        envs_to_use = full_movements.copy()
    else:
        raise ValueError

    for env in envs_to_use:
        trial_data = test.trial(
            options,
            env_dict[env],
        )

        end_delay = trial_data["epoch_bounds"]["delay"][1] - 1
        end_stable = trial_data["epoch_bounds"]["stable"][1] - 1

        end_movement = trial_data["epoch_bounds"]["movement"][1] - 1
        end_hold = trial_data["epoch_bounds"]["hold"][1] - 1

        if epoch == "stable":
            h = trial_data["h"][:, end_stable]
        elif epoch == "delay":
            h = trial_data["h"][:, end_delay]
        elif epoch == "hold":
            h = trial_data["h"][:, end_hold]
        elif epoch == "extension" and movement_type == "extension_retraction":
            middle_movement = get_middle_movement(trial_data)
            h = trial_data["h"][:, middle_movement]
        elif epoch == "extension" and movement_type == "extension":
            h = trial_data["h"][:, end_movement]
        elif epoch == "retraction":
            assert movement_type == "extension_retraction"
            h = trial_data["h"][:, end_movement]
        else:
            raise ValueError

        env_hs.append(h.mean(dim=0).unsqueeze(0))

    return env_hs, envs_to_use


# TODO try to add more motifs to this, direction and speed axes?
def epoch_pcs(
    model_name, epoch, movement_type, add_new_rule_inputs=False, plot_3d=False
):
    exp_path = f"results/{model_name}/compositionality/pcs"
    create_dir(exp_path)

    env_hs, envs_to_use = _get_mean_act(
        model_name, epoch, movement_type, add_new_rule_inputs=add_new_rule_inputs
    )

    epoch_pca = PCA(n_components=3)
    epoch_pca.fit(torch.cat(env_hs, dim=0))

    colors = plt.cm.tab10(np.linspace(0, 1, len(env_dict)))
    env_color_dict = {}

    # Adding colors for easier indexing
    for env, color in zip(env_dict, colors):
        env_color_dict[env] = color

    handles = []

    if plot_3d:
        _, ax = ax_3d_no_grid()
    else:
        _, ax = no_ticks_2d_ax()

    for env_data, env in zip(env_hs, envs_to_use):
        # Create patches with no border
        handles.append(
            mpatches.Patch(color=env_color_dict[env], label=env, edgecolor="none")
        )

        # transform
        h_proj = epoch_pca.transform(env_data)

        # Plot the 3D line
        if plot_3d:
            ax.scatter(
                h_proj[-1, 0],
                h_proj[-1, 1],
                h_proj[-1, 2],
                color=env_color_dict[env],
                s=250,
                alpha=0.75,
            )
        else:
            ax.scatter(
                h_proj[-1, 0],
                h_proj[-1, 1],
                color=env_color_dict[env],
                s=250,
                alpha=0.75,
            )

    # Set labels for axes
    save_fig(os.path.join(exp_path, f"{epoch}_{movement_type}_pcs"), eps=True)


def _plot_metric_scatter(
    all_combinations,
    combinations,
    combination_colors,
    metric1,
    metric2,
    exp_path,
    metric1_name,
    metric2_name,
):
    _, ax = standard_2d_ax()

    task_metric1 = convert_motif_dict_to_list(all_combinations, metric1)
    task_metric2 = convert_motif_dict_to_list(all_combinations, metric2)

    metric1_list = np.array(task_metric1).reshape((-1, 1))
    metric2_list = np.array(task_metric2).reshape((-1, 1))

    regression = LinearRegression()
    regression.fit(metric1_list, metric2_list)
    print(
        f"R^2 {metric1_name} to {metric2_name}: ",
        regression.score(metric1_list, metric2_list),
    )
    x = np.linspace(0, max(metric1_list))
    ax.plot(x, regression.coef_ * x + regression.intercept_, color="black")

    for c, combination in enumerate(combinations[:-1]):
        task_metric1_comb = convert_motif_dict_to_list(combination, metric1)
        task_metric2_comb = convert_motif_dict_to_list(combination, metric2)
        ax.scatter(
            task_metric1_comb,
            task_metric2_comb,
            s=100,
            alpha=0.25,
            color=combination_colors[c],
        )
    save_fig(
        os.path.join(exp_path, "movement", f"{metric1_name} vs {metric2_name}"),
        eps=True,
    )


def _plot_metric_bar(
    combinations, metric, exp_path, metric_name, combination_labels, combination_colors
):
    _, ax = standard_2d_ax()

    combination_means = []
    combination_stds = []
    combination_data = {}
    for c, combination in enumerate(combinations):
        task_metric = convert_motif_dict_to_list(combination, metric)

        """
            This is for finding examples for two_task_pcs.
            Delete this once done for sure, change as needed to find different examples
        """

        if (
            combination_labels[c] == "extension_tasks"
            and metric_name == "muscle_shapes"
        ):
            min_val = np.argmax(task_metric)
            condition_val = min_val % 32
            min_val /= 32
            min_val = math.floor(min_val)
            print(
                f"ext pair with highest muscle shape is: {combination[min_val]}, {condition_val}"
            )

        combination_data[combination_labels[c]] = task_metric
        combination_means.append(sum(task_metric) / len(task_metric))
        combination_stds.append(np.std(task_metric, ddof=1))

    # Convert values to list
    data_values = list(combination_data.values())

    ax.axhline(combination_means[-1], color="dimgrey", linestyle="dashed")
    parts = ax.violinplot(data_values[:-1], showmeans=True)

    # Custom colors
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(combination_colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
        pc.set_linewidth(1.2)
    parts["cbars"].set_edgecolor("black")
    parts["cmins"].set_edgecolor("black")
    parts["cmaxes"].set_edgecolor("black")
    parts["cmeans"].set_color("black")

    if "angles" in metric_name:
        plt.yticks([0, 1.5])
    elif "shapes" in metric_name:
        plt.yticks([0, 1])
    plt.xticks([])
    save_fig(os.path.join(exp_path, "movement", metric_name), eps=True)

    combination_list = list(combination_data.keys())
    pvalues(combination_list, combination_data, metric_name)


def _gather_env_pairs_from_trial(model_name, system, options):
    model_path = f"checkpoints/{model_name}"
    test = Test(model_path, model_name)

    assert system == "h" or system == "muscle"

    trial_data = {}
    for env in env_dict:
        trial_data = test.trial(options, env_dict[env])
        mov_beg, mov_end = mov_bounds(trial_data)

        if env in full_movements:
            halfway = int((mov_beg + mov_end) / 2)
            trial_data[env + "1"] = trial_data[system][:, mov_beg:halfway]
            trial_data[env + "2"] = trial_data[system][:, halfway:mov_end]
        else:
            trial_data[env] = trial_data[system][:, mov_beg:mov_end]

    # Get all unique pairs of unit activity across tasks
    combination_labels = list(itertools.combinations(trial_data.keys(), 2))
    combinations = unique_pairs_dict(combination_labels, trial_data)
    return trial_data, combinations, combination_labels


def trajectory_alignment(model_name):
    """
    Utilize metrics such as angle, distance, and disparity between trajectories to see
    how different task trajectories are aligned to one another

    This function will plot all available metrics for both the muscles and network
    """

    exp_path = f"results/{model_name}/compositionality/alignment"

    options = {
        "batch_size": 32,
        "reach_conds": np.tile(np.arange(0, 32, 1), int(32 / 32)),
        "speed_cond": 5,
    }

    _, combinations_h, combination_labels = _gather_env_pairs_from_trial(
        model_name, "h", options
    )
    _, combinations_muscle, _ = _gather_env_pairs_from_trial(
        model_name, "muscle", options
    )

    print("Computing Distances...")
    distances_h = distances_from_combinations(combinations_h, options["batch_size"])
    distances_muscle = distances_from_combinations(
        combinations_muscle, options["batch_size"]
    )

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
        combination_labels,
    ]

    all_subset_labels = [
        "subset_tasks",
        "retraction_tasks",
        "extension_tasks",
        "extension_retraction_tasks",
        "all_tasks",
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
        _plot_metric_bar(
            all_subsets,
            all_metrics[metric],
            exp_path,
            metric,
            all_subset_labels,
            all_subset_colors,
        )

    # -------------------------------------------------------------- SHAPE DISTRIBUTIONS

    all_shapes_h = convert_motif_dict_to_list(combination_labels, shapes_h)
    all_shapes_muscle = convert_motif_dict_to_list(combination_labels, shapes_muscle)

    bins = tuple(np.linspace(0, 1, 15))
    weights_data_h = np.ones_like(all_shapes_h) / len(all_shapes_h)
    weights_data_muscle = np.ones_like(all_shapes_muscle) / len(all_shapes_muscle)
    plt.hist(all_shapes_h, color="blue", alpha=0.5, bins=bins, weights=weights_data_h)
    plt.hist(
        all_shapes_muscle,
        color="purple",
        alpha=0.5,
        bins=bins,
        weights=weights_data_muscle,
    )
    plt.axvline(
        sum(all_shapes_h) / len(all_shapes_h),
        color="blue",
        linestyle="dashed",
        linewidth=2,
    )
    plt.axvline(
        sum(all_shapes_muscle) / len(all_shapes_muscle),
        color="purple",
        linestyle="dashed",
        linewidth=2,
    )
    plt.xlim([0, 1])
    save_fig(os.path.join(exp_path, "movement", "neural_muscle_shape_dists"), eps=True)

    # ------------------------------------------------------------- ANGLE DISTRIBUTIONS

    angle_h_dist = convert_motif_dict_to_list(combination_labels, angles_h)
    angle_muscle_dist = convert_motif_dict_to_list(combination_labels, angles_muscle)

    bins = tuple(np.linspace(0, 1.5, 15))
    weights_data_h = np.ones_like(angle_h_dist) / len(angle_h_dist)
    weights_data_muscle = np.ones_like(angle_muscle_dist) / len(angle_muscle_dist)
    plt.hist(angle_h_dist, color="blue", alpha=0.5, bins=bins, weights=weights_data_h)
    plt.hist(
        angle_muscle_dist,
        color="purple",
        alpha=0.5,
        bins=bins,
        weights=weights_data_muscle,
    )
    plt.axvline(
        sum(angle_h_dist) / len(angle_h_dist),
        color="blue",
        linestyle="dashed",
        linewidth=2,
    )
    plt.axvline(
        sum(angle_muscle_dist) / len(angle_muscle_dist),
        color="purple",
        linestyle="dashed",
        linewidth=2,
    )
    plt.xlim([0, 1.5])
    save_fig(os.path.join(exp_path, "movement", "neural_muscle_angle_dists"), eps=True)

    # -------------------------------------- SCATTER PLOTS

    for idx1, metric1 in enumerate(all_metrics):
        for idx2, metric2 in enumerate(all_metrics):
            if idx1 != idx2:
                _plot_metric_scatter(
                    combination_labels,
                    all_subsets,
                    all_subset_colors,
                    all_metrics[metric1],
                    all_metrics[metric2],
                    exp_path,
                    metric1,
                    metric2,
                )


def _get_vaf_combination(combination_labels, data, mode, comp_range):
    # Initialize the full pc dict
    all_vaf_list_means = []
    all_vaf_list_stds = []

    condition_tuple_dict = {}
    condition_label_dict = {}

    for combination in combination_labels:
        if combination not in data:
            combination = (combination[1], combination[0])
        for c, (task1_condition, task2_condition) in enumerate(
            zip(data[combination][0], data[combination][1])
        ):
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
            ratio_list = compute_vaf_ratio(
                condition, mode=mode, num_comps=pc, control=False
            )
            condition_vaf_list.extend(ratio_list)
        all_vaf_list_means.append(np.array(condition_vaf_list).mean())
        all_vaf_list_stds.append(np.array(condition_vaf_list).std())

    return np.array(all_vaf_list_means), np.array(all_vaf_list_stds)


def task_vaf_ratio(model_name):
    exp_path = f"results/{model_name}/compositionality/task_vaf_ratio"
    options = {
        "batch_size": 32,
        "reach_conds": np.tile(np.arange(0, 32, 1), int(32 / 32)),
        "speed_cond": 5,
    }
    _, combinations_h, combination_labels = _gather_env_pairs_from_trial(
        model_name, "h", options
    )
    _, combinations_muscle, _ = _gather_env_pairs_from_trial(
        model_name, "muscle", options
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
        "extension_retraction_tasks": "orange",
    }

    comp_range = 11

    # Plotting vaf for different number of pc components neural
    subset_pc_dict_means = {}
    subset_pc_dict_stds = {}
    for subset in all_subsets:
        subset_pc_dict_means[subset], subset_pc_dict_stds[subset] = (
            _get_vaf_combination(all_subsets[subset], combinations_h, "h", comp_range)
        )
    all_task_pc_means, all_task_pc_stds = _get_vaf_combination(
        combination_labels, combinations_h, "h", comp_range
    )

    _, ax = standard_2d_ax()
    x = np.arange(1, comp_range)
    for subset in subset_pc_dict_means:
        ax.plot(
            x,
            subset_pc_dict_means[subset],
            linewidth=4,
            alpha=0.75,
            color=all_subsets_colors[subset],
        )
        ax.fill_between(
            x,
            subset_pc_dict_means[subset] - subset_pc_dict_stds[subset],
            subset_pc_dict_means[subset] + subset_pc_dict_stds[subset],
            color=all_subsets_colors[subset],
            alpha=0.25,
        )
    ax.plot(x, all_task_pc_means, linewidth=4, alpha=0.75, color="grey")
    ax.fill_between(
        x,
        all_task_pc_means - all_task_pc_stds,
        all_task_pc_means + all_task_pc_stds,
        color="grey",
        alpha=0.25,
    )

    comp_range = 7
    ax.set_ylim((0.0, 1.1))
    save_fig(os.path.join(exp_path, "vaf_ratio_neural"), eps=True)

    # Plotting vaf for different number of pc components muscle
    subset_pc_dict_means = {}
    subset_pc_dict_stds = {}
    for subset in all_subsets:
        subset_pc_dict_means[subset], subset_pc_dict_stds[subset] = (
            _get_vaf_combination(
                all_subsets[subset], combinations_muscle, "muscle_acts", comp_range
            )
        )
    all_task_pc_means, all_task_pc_stds = _get_vaf_combination(
        combination_labels, combinations_muscle, "muscle_acts", comp_range
    )

    _, ax = standard_2d_ax()
    x = np.arange(1, comp_range)
    for subset in subset_pc_dict_means:
        ax.plot(
            x,
            subset_pc_dict_means[subset],
            linewidth=4,
            alpha=0.75,
            color=all_subsets_colors[subset],
        )
        ax.fill_between(
            x,
            subset_pc_dict_means[subset] - subset_pc_dict_stds[subset],
            subset_pc_dict_means[subset] + subset_pc_dict_stds[subset],
            color=all_subsets_colors[subset],
            alpha=0.25,
        )
    ax.plot(x, all_task_pc_means, linewidth=4, alpha=0.75, color="grey")
    ax.fill_between(
        x,
        all_task_pc_means - all_task_pc_stds,
        all_task_pc_means + all_task_pc_stds,
        color="grey",
        alpha=0.25,
    )

    ax.set_ylim((0, 1.1))
    save_fig(os.path.join(exp_path, "vaf_ratio_muscle"), eps=True)


######################################################
#               DSA Experiments                      #
######################################################


def dynamics_geometry_scatter(model_name, exp_path, load_file_name, save_file_name):
    model_path = f"checkpoints/{model_name}"

    _, ax = standard_2d_ax()

    procrustes_data = load_pickle(os.path.join(model_path, load_file_name))
    similarities = procrustes_data["similarities"]
    colors = procrustes_data["colors"]

    reduced = PCA(n_components=2).fit_transform(similarities)
    ax.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.75, s=250)
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, save_file_name), eps=True)


def dynamics_geometry_heatmap(model_name, exp_path, load_file_name, save_file_name):
    model_path = f"checkpoints/{model_name}"

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'

    procrustes_data = load_pickle(os.path.join(model_path, load_file_name))
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
    save_fig(os.path.join(exp_path, save_file_name), eps=True)


def dsa_similarity_matrix(model_name):
    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    options = {
        "batch_size": 32 * 4,
        "reach_conds": np.tile(np.arange(0, 32, 1), 4),
        "speed_cond": 5,
    }

    trial_data_h = []
    trial_data_colors = []

    trial_data, _, _ = _gather_env_pairs_from_trial(model_name, "h", options)

    for env in trial_data:
        pca = PCA(n_components=12)
        h_data = trial_data[env]
        h_reduced = pca.fit_transform(h_data.reshape((-1, h_data.shape[-1])))
        h_reduced = h_reduced.reshape((h_data.shape[0], h_data.shape[1], 12))

        trial_data_h.append(h_reduced)

        # 1 is for extension movements in extension-retraction tasks
        if "1" in env:
            trial_data_colors.append("pink")
        # 2 is for retraction movements in extension-retraction tasks
        elif "2" in env:
            trial_data_colors.append("purple")
        # else is just extension movements
        else:
            trial_data_colors.append("blue")

    dsa = DSA(
        trial_data_h,
        n_delays=90,
        rank=150,
        verbose=True,
        score_method="euclidean",
        device="cpu",
    )
    similarities = dsa.fit_score()

    dsa_data = {"similarities": similarities, "colors": trial_data_colors}

    with open(os.path.join(model_path, "dsa_similarity.txt"), "wb") as f:
        pickle.dump(dsa_data, f)

    dynamics_geometry_scatter(
        model_name, exp_path, "dsa_similarity.txt", "neural_dsa_scatter"
    )
    dynamics_geometry_heatmap(
        model_name, exp_path, "dsa_similarity.txt", "neural_dsa_similarity_vis"
    )


def procrustes_similarity_matrix(model_name):
    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    options = {
        "batch_size": 32 * 4,
        "reach_conds": np.tile(np.arange(0, 32, 1), 4),
        "speed_cond": 5,
    }

    trial_data_h = []
    trial_data_colors = []

    trial_data, _, _ = _gather_env_pairs_from_trial(model_name, "h", options)

    for env in trial_data:
        pca = PCA(n_components=12)
        h_data = trial_data[env]
        h_reduced = pca.fit_transform(h_data.reshape((-1, h_data.shape[-1])))
        h_reduced = h_reduced.reshape((h_data.shape[0], h_data.shape[1], 12))

        trial_data_h.append(h_reduced)

        # 1 is for extension movements in extension-retraction tasks
        if "1" in env:
            trial_data_colors.append("pink")
        # 2 is for retraction movements in extension-retraction tasks
        elif "2" in env:
            trial_data_colors.append("purple")
        # else is just extension movements
        else:
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

    with open(os.path.join(model_path, "procrustes_similarity.txt"), "wb") as f:
        pickle.dump(procrustes_data, f)

    dynamics_geometry_scatter(
        model_name, exp_path, "procrustes_similarity.txt", "neural_procrustes_scatter"
    )
    dynamics_geometry_heatmap(
        model_name,
        exp_path,
        "procrustes_similarity.txt",
        "neural_procrustes_similarity_vis",
    )


def silhouette_scores(data, labels, num_clusters=3):
    silhouette_values = sklearn.metrics.silhouette_samples(data, labels)
    means_lst = []
    for label in range(num_clusters):
        means_lst.append(silhouette_values[labels == label].mean())
    return means_lst


def task_similarity_classification(model_name, load_file_name, save_name):
    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    data = load_pickle(os.path.join(model_path, load_file_name))
    similarities = data["similarities"]
    pca = PCA(n_components=2)
    similarities = pca.fit_transform(similarities)
    labels = np.ones(shape=(15,))
    for i, color in enumerate(data["colors"]):
        if color == "blue":
            labels[i] = 0
        elif color == "pink":
            labels[i] = 1
        elif color == "purple":
            labels[i] = 2

    means_dsa = silhouette_scores(similarities, labels, 3)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x = ["Ext.", "Ext. (long)", "Ret."]
    plt.bar(
        x,
        means_dsa,
        color=["blue", "pink", "purple"],
        capsize=10,
        edgecolor="black",
        alpha=0.75,
    )
    plt.xticks([])
    save_fig(os.path.join(exp_path, save_name), eps=True)


######################################################
#            Rule Input Experiments                  #
######################################################


# Get the loss from every composite input on each environment and get the heat map
def composite_rule_input_heat_map(model_name):
    exp_path = f"results/{model_name}/compositionality/composite_rule_inputs/heat_map"
    load_name = f"checkpoints/{model_name}/composite_rule_inputs.pkl"
    trial_data = load_pickle(load_name)

    for env in env_dict_ext:
        fig, ax = no_ticks_2d_ax()
        rule_input = trial_data[env]["rule_input"][:, :5].numpy()
        im = ax.imshow(rule_input, cmap="RdBu", vmin=2, vmax=-2)
        _ = fig.colorbar(im, ax=ax, fraction=0.07, pad=0.04)
        fig.tight_layout()
        save_fig(os.path.join(exp_path, f"extension_heat_map_{env}"), eps=True)


# Get the loss from every composite input on each environment and get the heat map
def composite_rule_input_kinematics(model_name):
    def plot_env_kinematics(xy):
        _, ax = empty_2d_ax()
        for i, batch in enumerate(xy):
            ax.plot(batch[:, 0], batch[:, 1], linewidth=4, color=colors[i])
            ax.scatter(batch[0, 0], batch[0, 1], s=250, marker="^", color=colors[i])
            ax.scatter(batch[-1, 0], batch[-1, 1], s=250, marker="X", color=colors[i])

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

    env_hs, _ = _get_mean_act(
        model_name, "delay", "extension", delay_cond=2, batch_size=32
    )
    env_hs = np.concatenate(env_hs)
    epoch_pca = PCA(n_components=3)
    epoch_pca.fit(env_hs)

    _, ax = ax_3d_no_grid()
    for e, env in enumerate(env_dict_ext):
        delay_start, delay_end = delay_bounds(trial_data)

        composite_h = trial_data[env]["h"][:, delay_start:delay_end]
        composite_h = composite_h.mean(dim=0)

        all_data_for_min = np.concatenate([composite_h, env_hs])
        red_all_data_for_min = epoch_pca.transform(all_data_for_min)
        min_val = np.min(red_all_data_for_min)

        reduced_baseline = epoch_pca.transform(env_hs)
        ax.scatter(
            reduced_baseline[e, 0],
            reduced_baseline[e, 1],
            reduced_baseline[e, 2],
            s=200,
            marker="o",
            color=colors_envs[e],
        )
        ax.scatter(
            reduced_baseline[e, 0],
            reduced_baseline[e, 1],
            min_val,
            s=200,
            marker="o",
            color=colors_envs[e],
            alpha=0.10,
        )

        reduced_composite = epoch_pca.transform(composite_h)
        ax.scatter(
            reduced_composite[-1, 0],
            reduced_composite[-1, 1],
            min_val,
            s=200,
            marker="X",
            color=colors_envs[e],
            alpha=0.10,
        )
        ax.scatter(
            reduced_composite[-1, 0],
            reduced_composite[-1, 1],
            reduced_composite[-1, 2],
            s=200,
            marker="X",
            color=colors_envs[e],
        )
    save_fig(os.path.join(exp_path, "extension_init_all"), eps=True)


def composite_input_loss(model_name):
    exp_path = f"results/{model_name}/compositionality/composite_rule_inputs/losses"
    load_name = f"checkpoints/{model_name}/composite_rule_inputs.pkl"
    trial_data = load_pickle(load_name)
    colors_envs = plt.cm.tab10(np.linspace(0, 1, len(env_dict)))

    _, ax = standard_2d_ax()
    for e, env in enumerate(trial_data):
        loss = trial_data[env]["test_loss"]
        ax.plot(loss, linewidth=4, color=colors_envs[e], alpha=0.75)
    save_fig(os.path.join(exp_path, "optimization_losses"), eps=True)


def run_composite_input_optimization(model_name):
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    options = {
        "batch_size": 8,
        "reach_conds": np.arange(0, 32, 4),
        "speed_cond": 9,
        "custom_delay": 150,
    }

    all_trial_data = {}
    for env in env_dict_ext:
        trial_data = composite_input_optimization(
            model_path, model_file, options, env_dict_ext[env], env
        )
        all_trial_data[env] = trial_data

    # Save all information of inputs across envs
    save_name = "composite_rule_inputs.pkl"
    fname = os.path.join(model_path, save_name)
    with open(fname, "wb") as f:
        pickle.dump(all_trial_data, f)


def sequential_rule_inputs(model_name, extension, retraction):
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = (
        f"results/{model_name}/compositionality/sequential_rule_inputs/kinematics"
    )
    colors_conds = plt.cm.inferno(np.linspace(0, 1, 8))
    options = {
        "batch_size": 4,
        "reach_conds": np.arange(0, 32, 8),
        "speed_cond": 5,
        "delay_cond": 2,
    }

    extension_env = env_dict[extension]
    retraction_env = env_dict[retraction]

    trial_data = test_sequential_inputs(
        model_path, model_file, options, extension_env, retraction_env
    )
    kinematics = trial_data["xy"]
    middle_movement = get_middle_movement(trial_data)

    fig, ax = empty_2d_ax()
    for cond in range(kinematics.shape[0]):
        ax.plot(
            kinematics[cond, :middle_movement, 0],
            kinematics[cond, :middle_movement, 1],
            linewidth=4,
            color=colors_conds[cond],
            alpha=0.25,
            linestyle="dashed",
        )
        ax.plot(
            kinematics[cond, middle_movement:, 0],
            kinematics[cond, middle_movement:, 1],
            linewidth=4,
            color=colors_conds[cond],
        )
        ax.scatter(
            kinematics[cond, 0, 0],
            kinematics[cond, 0, 1],
            s=100,
            marker="^",
            color=colors_conds[cond],
        )
        ax.scatter(
            kinematics[cond, -1, 0],
            kinematics[cond, -1, 1],
            s=100,
            marker="x",
            color=colors_conds[cond],
        )
    save_fig(os.path.join(exp_path, f"{extension}_{retraction}"), eps=True)
