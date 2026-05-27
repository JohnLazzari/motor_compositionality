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


def _default_analysis_options(batch_size=32, reach_step=1, delay_cond=0, speed_cond=5):
    """Return the standard option dictionary used for manifold test trials."""
    return {
        "batch_size": batch_size,
        "reach_conds": torch.arange(0, 32, reach_step),
        "delay_cond": delay_cond,
        "deterministic": True,
        "speed_cond": speed_cond,
    }


def _system_mode(system):
    """Map a high-level system name to the trial-data key and unit count."""
    if system == "neural":
        return "h", None
    if system == "muscle":
        return "muscle_acts", 6
    raise ValueError("Not a valid system")


def _num_components(mode, num_comps, analysis):
    """Resolve the number of principal components for an analysis type."""
    if num_comps is not None:
        return num_comps
    if mode == "h":
        return 12
    if mode == "muscle_acts":
        return 3 if analysis == "angles" else 2
    raise ValueError


def _flatten_time_series(data):
    """Flatten batch and time dimensions while preserving feature columns."""
    return data.reshape((-1, data.shape[-1])).numpy()


def _fit_pca_components(data, num_comps):
    """Fit PCA to a flattened trial array and return the leading components."""
    pca = PCA()
    pca.fit(data)
    assert pca.components_ is not None
    return pca.components_[:num_comps]


def _random_bases(num_bases, baseline_dim, num_comps):
    """Create orthonormal random bases for control manifold comparisons."""
    random_matrices = np.random.randn(num_bases, baseline_dim, num_comps)
    random_bases = np.empty(shape=(num_bases, num_comps, baseline_dim))
    for basis in range(num_bases):
        q, _ = np.linalg.qr(random_matrices[basis])
        random_bases[basis] = q.T
    return random_bases


def _principal_angles(basis1, basis2):
    """Compute principal angles in degrees between two component bases."""
    inner_prod_mat = basis1 @ basis2.T
    _, s, _ = np.linalg.svd(inner_prod_mat)
    return np.degrees(np.arccos(s))


def _vaf_ratio(data, within_components, across_components):
    """Compute across-task VAF normalized by within-task VAF."""
    total_variance = data.var(axis=0).sum()
    across_vaf = (across_components @ data.T).T.var(axis=0).sum() / total_variance
    within_vaf = (within_components @ data.T).T.var(axis=0).sum() / total_variance
    return across_vaf / within_vaf, within_vaf


def _random_vaf_percentile(data, random_bases, percentile):
    """Compute the requested percentile of VAF over random control bases."""
    total_variance = data.var(axis=0).sum()
    across_task_vaf = (random_bases @ data.T).var(axis=2).sum(axis=1) / total_variance
    return np.percentile(across_task_vaf, percentile)


def _movement_data(test, options, mode):
    """Collect movement-epoch data for every environment in ``env_dict``."""
    trial_data_mode = {}
    for env in env_dict:
        trial_data = test.trial(options, env_dict[env])
        mov_beg, mov_end = mov_bounds(trial_data)
        trial_data_mode[env] = trial_data[mode][:, mov_beg:mov_end]
    return trial_data_mode


def _task_combinations(test, options, mode):
    """Return all unique task-pair combinations for movement-epoch data."""
    trial_data_mode = _movement_data(test, options, mode)
    combination_labels = list(itertools.combinations(trial_data_mode, 2))
    return unique_pairs(combination_labels, trial_data_mode)


def _epoch_combinations(test, options, mode):
    """Return delay-vs-movement combinations for each environment."""
    combinations = []
    for env in env_dict:
        trial_data = test.trial(options, env_dict[env])
        mov_beg, mov_end = mov_bounds(trial_data)
        delay_beg, delay_end = delay_bounds(trial_data)
        combinations.append(
            (
                trial_data[mode][:, delay_beg:delay_end],
                trial_data[mode][:, mov_beg:mov_end],
            )
        )
    return combinations


def _condition_combinations(test, options, mode):
    """Return condition-pair combinations from movement activity."""
    combinations = []
    combination_labels = list(itertools.combinations([0, 1, 2, 3], 2))
    for env in env_dict:
        trial_data = test.trial(options, env_dict[env])
        mov_beg, mov_end = mov_bounds(trial_data)
        movement_act = trial_data[mode][:, mov_beg:mov_end]
        movement_act = [act for act in movement_act]

        # This assignment preserves the original function behavior.
        combinations = unique_pairs(combination_labels, movement_act)
    return combinations


def _comparison_combinations(test, comparison, mode):
    """Build activity combinations for task, epoch, or condition comparisons."""
    options = _default_analysis_options()

    if comparison == "task":
        return _task_combinations(test, options, mode)
    if comparison == "epoch":
        return _epoch_combinations(test, options, mode)
    if comparison == "condition":
        options = _default_analysis_options(batch_size=4, reach_step=8)
        return _condition_combinations(test, options, mode)
    raise ValueError


def compute_principal_angles(
    combinations, mode, hid_size, num_comps=None, control=True
):
    """Compute principal angles for paired activity manifolds.

    Parameters
    ----------
    combinations : iterable
        Iterable of two-item tuples containing activity tensors to compare.
    mode : str
        Trial-data key, either ``"h"`` for neural activity or
        ``"muscle_acts"`` for muscle activity.
    hid_size : int
        Feature dimensionality used when constructing neural control bases.
    num_comps : int, optional
        Number of principal components to compare. Defaults depend on ``mode``.
    control : bool, optional
        Whether to return a random-basis control distribution.

    Returns
    -------
    list or tuple
        Principal angles for each combination, and optionally a stacked control
        array when ``control`` is true.
    """

    angles_list = []
    control_list = []
    num_comps = _num_components(mode, num_comps, analysis="angles")

    if control:
        # Create a random manifold as a control
        random_bases = _random_bases(10000, hid_size, num_comps)
    else:
        random_bases = None

    for combination in combinations:
        # ------------------------------------ GET PRINCIPLE ANGLES
        task1_data = _flatten_time_series(combination[0])
        task2_data = _flatten_time_series(combination[1])
        pca1_comps = _fit_pca_components(task1_data, num_comps)
        pca2_comps = _fit_pca_components(task2_data, num_comps)
        angles_list.append(_principal_angles(pca1_comps, pca2_comps))

    if control:
        assert random_bases is not None
        # Get principle angles control
        for _ in range(10000):
            a, b = np.random.choice(random_bases.shape[0], size=2, replace=False)
            control_list.append(_principal_angles(random_bases[a], random_bases[b]))
        control_array = np.stack(control_list, axis=0)

        return angles_list, control_array

    else:
        return angles_list


def compute_vaf_ratio(combinations, mode, hid_size, num_comps=None, control=True):
    """Compute normalized across-task variance-accounted-for ratios.

    Each paired combination is projected onto its own PCA basis and the other
    task's PCA basis. The returned ratio is across-task VAF divided by
    within-task VAF. Optional controls use random orthonormal bases.
    """
    # Only use two muscle PCs for this task, but use three for the one above

    vaf_ratio_list = []
    vaf_ratio_list_control = []

    if mode == "h":
        num_comps = _num_components(mode, num_comps, analysis="vaf")
        baseline_dim = hid_size
        percentile = 90
    elif mode == "muscle_acts":
        num_comps = _num_components(mode, num_comps, analysis="vaf")
        baseline_dim = 6
        percentile = 90
    else:
        raise ValueError

    if control:
        # Create a random manifold as a control
        random_bases = _random_bases(5000, baseline_dim, num_comps)
    else:
        random_bases = None

    for combination in combinations:
        task1_data = _flatten_time_series(combination[0])
        task2_data = _flatten_time_series(combination[1])
        pca1_comps = _fit_pca_components(task1_data, num_comps)
        pca2_comps = _fit_pca_components(task2_data, num_comps)

        # ------------------------------------ TRUE ACROSS AND WITHIN TASK VAFs

        # Get VAF
        ratio_task1, within_task_vaf_task1 = _vaf_ratio(
            task1_data, pca1_comps, pca2_comps
        )
        vaf_ratio_list.append(ratio_task1)

        ratio_task2, within_task_vaf_task2 = _vaf_ratio(
            task2_data, pca2_comps, pca1_comps
        )
        vaf_ratio_list.append(ratio_task2)

        if control:
            assert random_bases is not None
            # ------------------------------------ CONTROL ACROSS TASK VAFs

            # Get random VAFs
            vaf_ratio_list_control.append(
                _random_vaf_percentile(task1_data, random_bases, percentile)
                / within_task_vaf_task1
            )

            # Get random VAFs
            vaf_ratio_list_control.append(
                _random_vaf_percentile(task2_data, random_bases, percentile)
                / within_task_vaf_task2
            )

    if control:
        return vaf_ratio_list, vaf_ratio_list_control
    else:
        return vaf_ratio_list


def gather_principal_angles(model_name, system, comparison):
    """Run trials and compute principal angles for a comparison type.

    Parameters
    ----------
    model_name : str
        Name of the checkpoint directory and model file stem.
    system : str
        ``"neural"`` to analyze hidden activity or ``"muscle"`` to analyze
        muscle activations.
    comparison : str
        Comparison type: ``"task"``, ``"epoch"``, or ``"condition"``.

    Returns
    -------
    tuple
        X-axis component indices, principal angles, and random control angles.
    """
    model_path = f"checkpoints/{model_name}"
    test = Test(model_path, model_name)

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only

    mode, num_units = _system_mode(system)
    if system == "neural":
        x = np.arange(1, 13)
        num_units = test.hid_size
    elif system == "muscle":
        x = np.arange(1, 4)
    else:
        raise ValueError

    combinations = _comparison_combinations(test, comparison, mode)

    # Keeping this here for now in case I need it later, remove otherwise
    angles_list, control_array = compute_principal_angles(combinations, mode, num_units)

    return x, angles_list, control_array


def plot_principal_angles(
    angles_dict, control_array, x, color="blue", alpha=0.75, control_color="grey"
):
    """Plot principal-angle curves with a random-control reference line."""
    # Take mean of each angle in control
    mean_control = np.percentile(control_array, 0.1, axis=0, keepdims=False)

    for angles in angles_dict:
        plt.plot(x, angles, linewidth=4, alpha=alpha, color=color)
    plt.plot(x, mean_control, linewidth=2, linestyle="dashed", color=control_color)

    # Access current axes and hide top/right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)


def gather_vaf_ratio(model_name, system, comparison):
    """Run trials and compute VAF ratios for a comparison type.

    Parameters
    ----------
    model_name : str
        Name of the checkpoint directory and model file stem.
    system : str
        ``"neural"`` or ``"muscle"``.
    comparison : str
        Comparison type: ``"task"``, ``"epoch"``, or ``"condition"``.

    Returns
    -------
    tuple
        VAF ratios and random-control VAF ratios.
    """
    model_path = f"checkpoints/{model_name}"
    test = Test(model_path, model_name)

    mode, _ = _system_mode(system)

    if comparison == "epoch":
        # Preserve original behavior: epoch VAF comparisons always used hidden
        # activity, regardless of the requested system.
        combinations = _comparison_combinations(test, comparison, "h")
    else:
        combinations = _comparison_combinations(test, comparison, mode)

    vaf_ratio_list, vaf_ratio_control = compute_vaf_ratio(
        combinations, mode, test.hid_size
    )

    return vaf_ratio_list, vaf_ratio_control


def plot_vaf_ratio(
    vaf_ratio_list, vaf_ratio_control, color="purple", control_color="grey"
):
    """Plot VAF-ratio histogram with a random-control mean reference line."""
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
    """Project two task matrices with PCA and fit canonical correlations."""
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
    """Compute task-pair canonical correlations within extension/retraction sets."""
    model_path = f"checkpoints/{model_name}"
    test = Test(model_path, model_name)
    options = _default_analysis_options(delay_cond=2)

    trial_data_mode = _movement_data(test, options, system)

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
    """Compare neural and muscle manifold similarity across movement conditions."""
    model_path = f"checkpoints/{model_name}"
    test = Test(model_path, model_name)
    exp_path = f"results/{model_name}/pc_angles"
    baseline_data = []

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only

    options = _default_analysis_options(batch_size=8, reach_step=4)

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
