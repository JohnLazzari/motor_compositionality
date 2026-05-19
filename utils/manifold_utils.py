import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import torch
import os
from utils.plot_utils import standard_2d_ax, save_fig
from utils.exp_utils import mov_bounds, delay_bounds, unique_pairs
import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm
import itertools
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from utils.exp_utils import env_dict
from modules.test import Test

plt.rcParams.update({"font.size": 18})  # Sets default font size for all text


def compute_principal_angles(
    combinations, mode, hid_size, num_comps=None, control=True
):
    """
    Perform manifold analysis (principle angles and VAF)

    params:
        system: "neural" or "muscle"
        epoch: "delay" or "movement"
    """

    angles_list = []
    control_list = []

    if mode == "h":
        num_comps = 12 if num_comps is None else num_comps
    elif mode == "muscle_acts":
        num_comps = 3 if num_comps is None else num_comps
    else:
        raise ValueError

    if control:
        # Create a random manifold as a control
        random_matrices = np.random.randn(10000, hid_size, num_comps)
        random_bases = np.empty(shape=(10000, num_comps, hid_size))
        for basis in range(10000):
            q, _ = np.linalg.qr(random_matrices[basis])
            random_bases[basis] = q.T
    else:
        random_bases = None

    for combination in combinations:
        # ------------------------------------ GET PRINCIPLE ANGLES

        pca1 = PCA()
        pca2 = PCA()

        task1_data = combination[0].reshape((-1, combination[0].shape[-1])).numpy()
        task2_data = combination[1].reshape((-1, combination[1].shape[-1])).numpy()

        pca1.fit(task1_data)
        pca2.fit(task2_data)

        pca1_comps = pca1.components_[:num_comps]
        pca2_comps = pca2.components_[:num_comps]

        # Get principle angles
        inner_prod_mat = pca1_comps @ pca2_comps.T  # Should be m x m
        _, s, _ = np.linalg.svd(inner_prod_mat)
        angles = np.degrees(np.arccos(s))
        angles_list.append(angles)

    if control:
        assert random_bases is not None
        # Get principle angles control
        for _ in range(10000):
            a, b = np.random.choice(random_bases.shape[0], size=2, replace=False)
            inner_prod_mat = random_bases[a] @ random_bases[b].T  # Should be m x m
            _, s, _ = np.linalg.svd(inner_prod_mat)
            angles = np.degrees(np.arccos(s))
            control_list.append(angles)
        control_array = np.stack(control_list, axis=0)

        return angles_list, control_array

    else:
        return angles_list


def compute_vaf_ratio(combinations, mode, hid_size, num_comps=None, control=True):
    # Only use two muscle PCs for this task, but use three for the one above

    vaf_ratio_list = []
    vaf_ratio_list_control = []

    if mode == "h":
        num_comps = 12 if num_comps is None else num_comps
        baseline_dim = hid_size
        percentile = 90
    elif mode == "muscle_acts":
        num_comps = 2 if num_comps is None else num_comps
        baseline_dim = 6
        percentile = 90
    else:
        raise ValueError

    if control:
        # Create a random manifold as a control
        random_matrices = np.random.randn(5000, baseline_dim, num_comps)
        random_bases = np.empty(shape=(5000, num_comps, baseline_dim))
        for basis in range(5000):
            q, _ = np.linalg.qr(random_matrices[basis])
            random_bases[basis] = q.T
    else:
        random_bases = None

    for combination in combinations:
        pca1 = PCA()
        pca2 = PCA()

        task1_data = combination[0].reshape((-1, combination[0].shape[-1])).numpy()
        task2_data = combination[1].reshape((-1, combination[1].shape[-1])).numpy()

        pca1.fit(task1_data)
        pca2.fit(task2_data)

        pca1_comps = pca1.components_[:num_comps]
        pca2_comps = pca2.components_[:num_comps]

        # ------------------------------------ TRUE ACROSS AND WITHIN TASK VAFs

        # Get VAF
        across_task_vaf_task1 = (pca2_comps @ task1_data.T).T.var(
            axis=0
        ).sum() / task1_data.var(axis=0).sum()
        within_task_vaf_task1 = (pca1_comps @ task1_data.T).T.var(
            axis=0
        ).sum() / task1_data.var(axis=0).sum()
        ratio_task1 = across_task_vaf_task1 / within_task_vaf_task1
        vaf_ratio_list.append(ratio_task1)

        across_task_vaf_task2 = (pca1_comps @ task2_data.T).T.var(
            axis=0
        ).sum() / task2_data.var(axis=0).sum()
        within_task_vaf_task2 = (pca2_comps @ task2_data.T).T.var(
            axis=0
        ).sum() / task2_data.var(axis=0).sum()
        ratio_task2 = across_task_vaf_task2 / within_task_vaf_task2
        vaf_ratio_list.append(ratio_task2)

        if control:
            # ------------------------------------ CONTROL ACROSS TASK VAFs

            # Get random VAFs
            across_task_vaf = (random_bases @ task1_data.T).var(axis=2).sum(
                axis=1
            ) / task1_data.var(axis=0).sum()
            vaf_ratio_list_control.append(
                np.percentile(across_task_vaf, percentile) / within_task_vaf_task1
            )

            # Get random VAFs
            across_task_vaf = (random_bases @ task2_data.T).var(axis=2).sum(
                axis=1
            ) / task2_data.var(axis=0).sum()
            vaf_ratio_list_control.append(
                np.percentile(across_task_vaf, percentile) / within_task_vaf_task2
            )

    if control:
        return vaf_ratio_list, vaf_ratio_list_control
    else:
        return vaf_ratio_list


def gather_principal_angles(model_name, system, comparison):
    model_path = f"checkpoints/{model_name}"
    test = Test(model_path, model_name)

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only

    if system == "neural":
        mode = "h"
        x = np.arange(1, 13)
        num_units = test.hid_size
    elif system == "muscle":
        mode = "muscle_acts"
        x = np.arange(1, 4)
        num_units = 6
    else:
        raise ValueError("Not a valid system")

    options = {
        "batch_size": 32,
        "reach_conds": torch.arange(0, 32, 1),
        "delay_cond": 0,
        "speed_cond": 5,
    }

    if comparison == "task":
        trial_data_mode = {}
        combinations = []
        for env in env_dict:
            trial_data = test.trial(options, env_dict[env])
            mov_beg, mov_end = mov_bounds(trial_data)
            movement_data = trial_data[mode][:, mov_beg:mov_end]
            trial_data_mode[env] = movement_data

        # Get all unique pairs of unit activity across tasks
        combination_labels = list(itertools.combinations(trial_data_mode, 2))
        combinations = unique_pairs(combination_labels, trial_data_mode)

    elif comparison == "epoch":
        combinations = []
        combination_labels = []
        for env in env_dict:
            trial_data = test.trial(options, env_dict[env])
            combination_labels.append(env)

            mov_beg, mov_end = mov_bounds(trial_data)
            delay_beg, delay_end = delay_bounds(trial_data)

            combinations.append(
                (
                    trial_data[mode][:, delay_beg:delay_end],
                    trial_data[mode][:, mov_beg:mov_end],
                )
            )

    elif comparison == "condition":
        options = {
            "batch_size": 4,
            "reach_conds": torch.arange(0, 32, 8),
            "delay_cond": 0,
            "speed_cond": 5,
        }

        combinations = []
        combination_labels = list(itertools.combinations([0, 1, 2, 3], 2))
        for env in env_dict:
            trial_data = test.trial(options, env_dict[env])
            mov_beg, mov_end = mov_bounds(trial_data)
            movement_act = trial_data[mode][:, mov_beg:mov_end]
            movement_act = [act for act in movement_act]

            # Get all unique pairs of unit activity across tasks
            combinations = unique_pairs(combination_labels, movement_act)
    else:
        raise ValueError

    # Keeping this here for now in case I need it later, remove otherwise
    angles_list, control_array = compute_principal_angles(combinations, mode, num_units)

    return x, angles_list, control_array


def plot_principal_angles(
    angles_dict, control_array, x, color="blue", alpha=0.75, control_color="grey"
):
    # Take mean of each angle in control
    mean_control = np.percentile(control_array, 0.1, axis=0, keepdims=False)

    for angles in angles_dict:
        plt.plot(x, angles, linewidth=4, alpha=alpha, color=color)
    plt.plot(x, mean_control, linewidth=2, linestyle="dashed", color=control_color)

    # Access current axes and hide top/right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)


def gather_vaf_ratio(model_name, system, comparison):
    model_path = f"checkpoints/{model_name}"
    test = Test(model_path, model_name)

    if system == "neural":
        mode = "h"
    elif system == "muscle":
        mode = "muscle_acts"
    else:
        raise ValueError("Not a valid system")

    options = {
        "batch_size": 32,
        "reach_conds": torch.arange(0, 32, 1),
        "delay_cond": 0,
        "speed_cond": 5,
    }

    if comparison == "task":
        trial_data_mode = {}
        combinations = []
        for env in env_dict:
            trial_data = test.trial(options, env_dict[env])
            mov_beg, mov_end = mov_bounds(trial_data)
            trial_data_mode[env] = trial_data[mode][:, mov_beg:mov_end]

        # Get all unique pairs of unit activity across tasks
        combination_labels = list(itertools.combinations(trial_data_mode, 2))
        combinations = unique_pairs(combination_labels, trial_data_mode)

    elif comparison == "epoch":
        combinations = []
        for env in env_dict:
            trial_data = test.trial(options, env=env_dict[env])
            mov_beg, mov_end = mov_bounds(trial_data)
            delay_beg, delay_end = delay_bounds(trial_data)
            combinations.append(
                (
                    trial_data["h"][:, delay_beg:delay_end],
                    trial_data["h"][:, mov_beg:mov_end],
                )
            )

    elif comparison == "condition":
        options = {
            "batch_size": 4,
            "reach_conds": torch.arange(0, 32, 8),
            "delay_cond": 0,
            "speed_cond": 5,
        }

        combinations = []
        combination_labels = list(itertools.combinations([0, 1, 2, 3], 2))
        for env in env_dict:
            trial_data = test.trial(options, env_dict[env])
            mov_beg, mov_end = mov_bounds(trial_data)
            movement_act = trial_data[mode][:, mov_beg:mov_end]
            movement_act = [act for act in movement_act]
            # Get all unique pairs of unit activity across tasks
            combinations = unique_pairs(combination_labels, movement_act)
    else:
        raise ValueError

    vaf_ratio_list, vaf_ratio_control = compute_vaf_ratio(
        combinations, mode, test.hid_size
    )

    return vaf_ratio_list, vaf_ratio_control


def plot_vaf_ratio(
    vaf_ratio_list, vaf_ratio_control, color="purple", control_color="grey"
):
    bins = tuple(np.linspace(0, 1, 15))
    weights_data = np.ones_like(vaf_ratio_list) / len(vaf_ratio_list)
    control_mean = sum(vaf_ratio_control) / len(vaf_ratio_control)
    plt.hist(vaf_ratio_list, bins=bins, weights=weights_data, color=color, alpha=0.75)
    plt.axvline(control_mean, color=control_color, linestyle="dashed")
    plt.xlim(0, 1)

    # Access current axes and hide top/right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)


def _cca(task1, task2, system):
    if system == "h":
        n_components = 10
    elif system == "muscle_acts":
        n_components = 4
    else:
        raise ValueError

    task1_pca = PCA(n_components=n_components)
    task1_h = task1_pca.fit_transform(task1.reshape((-1, task1.shape[-1])))

    task2_pca = PCA(n_components=n_components)
    task2_h = task2_pca.fit_transform(task2.reshape((-1, task2.shape[-1])))

    cca = CCA(n_components=n_components)
    cca.fit(task1_h, task2_h)
    X_c, Y_c = cca.transform(task1_h, task2_h)

    return X_c, Y_c, cca


def task_ccs(model_name, system):
    model_path = f"checkpoints/{model_name}"
    test = Test(model_path, model_name)
    options = {
        "batch_size": 32,
        "reach_conds": torch.arange(0, 32, 1),
        "delay_cond": 2,
        "speed_cond": 5,
    }

    trial_data_mode = {}
    combinations = []
    for env in env_dict:
        trial_data = test.trial(options, env_dict[env])
        mov_beg, mov_end = mov_bounds(trial_data)
        trial_data_mode[env] = trial_data[system][:, mov_beg:mov_end]

    extensions = [
        "Reach",
        "ClkCurvedReach",
        "CClkCurvedReach",
        "Sinusoid",
        "InvSinusoid",
    ]

    retractions = [
        "ReachBack",
        "ClkCycle",
        "CClkCycle",
        "Figure8",
        "InvFigure8",
    ]

    extension_combinations = list(itertools.combinations(extensions, 2))
    retraction_combinations = list(itertools.combinations(retractions, 2))
    combination_labels = [*extension_combinations, *retraction_combinations]

    # Get all unique pairs of unit activity across tasks
    combinations = unique_pairs(combination_labels, trial_data_mode)

    all_ccs = []
    for combination in combinations:
        X_c, Y_c, _ = _cca(combination[0], combination[1], system)
        corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[-1])]
        all_ccs.append(corrs)

    return all_ccs


def network_muscle_mode_similarity(model_name):
    model_path = f"checkpoints/{model_name}"
    test = Test(model_path, model_name)
    exp_path = f"results/{model_name}/pc_angles"
    baseline_data = []

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only

    options = {
        "batch_size": 8,
        "reach_conds": torch.arange(0, 32, 4),
        "delay_cond": 0,
        "speed_cond": 5,
    }

    trial_data_hs = []
    trial_data_mas = []
    for env in env_dict:
        trial_data = test.trial(options, env_dict[env])
        mov_beg, mov_end = mov_bounds(trial_data)
        movement_h = trial_data["h"][:, mov_beg:mov_end]
        movement_muscle = trial_data["muscle_acts"][:, mov_beg:mov_end]
        for cond in range(options["batch_size"]):
            trial_data_hs.append(movement_h[cond])
            trial_data_mas.append(movement_muscle[cond])

    # Get all unique pairs of unit activity across tasks
    combinations_h = list(itertools.combinations(trial_data_hs, 2))
    combinations_mas = list(itertools.combinations(trial_data_mas, 2))

    angles_list_h, control_array_h = compute_principal_angles(
        combinations_h, baseline_data, "h", test.hid_size
    )
    angles_list_m, control_array_m = compute_principal_angles(
        combinations_mas, baseline_data, "muscle_acts", 6
    )

    angles_list_h = np.stack(angles_list_h)
    angles_list_m = np.stack(angles_list_m)

    mean_control_h = np.percentile(control_array_h, 0.1, axis=0, keepdims=False)
    mean_control_m = np.percentile(control_array_m, 0.1, axis=0, keepdims=False)

    comparison_h = angles_list_h < mean_control_h
    comparison_m = angles_list_m < mean_control_m

    percent_below_h = comparison_h.sum(axis=1) / 12 * 100  # shape: (5000,)
    percent_below_m = comparison_m.sum(axis=1) / 3 * 100  # shape: (5000,)

    bin_edges = np.arange(0, 110, 10)

    _, ax = standard_2d_ax(w=8, h=5)
    ax.hist(
        percent_below_h,
        bins=10,
        range=(0, 100),
        weights=np.ones_like(percent_below_h) / len(percent_below_h) * 100,
        color="blue",
        width=5,
    )
    ax.hist(
        percent_below_m,
        bins=10,
        range=(0, 100),
        weights=np.ones_like(percent_below_m) / len(percent_below_m) * 100,
        color="purple",
        width=5,
    )
    ax.set_xticks(bin_edges)

    save_fig(os.path.join(exp_path, "network_muscle_mode_hist"), eps=True)
