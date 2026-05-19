import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import os
import motornet as mn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from mrnntorch.analysis import mFixedPointFinder, mLinearization
import tqdm as tqdm
from sklearn.decomposition import PCA
from exp_utils import (
    env_dict,
    get_interpolation_input,
    load_pickle,
    split_movement_epoch,
)
from modules.test import Test
from utils.plot_utils import save_fig, standard_2d_ax, empty_3d, ax_3d_no_grid

# ---------------- Helper Functions -----------------


def _3d_pca_across_conditions(ax, condition_data):
    for condition in condition_data:
        ax.plot(
            condition[:, 0],
            condition[:, 1],
            condition[:, 2],
            linewidth=4,
            color="black",
            alpha=0.75,
            zorder=10,
        )
        ax.scatter(
            condition[0, 0],
            condition[0, 1],
            condition[0, 2],
            s=150,
            marker="^",
            color="black",
            zorder=20,
        )
        ax.scatter(
            condition[-1, 0],
            condition[-1, 1],
            condition[-1, 2],
            s=150,
            marker="X",
            color="black",
            zorder=20,
        )


def _choose_next_fp(anchor_state, fp_obj):
    dist = np.inf
    chosen_fp = None
    for idx in range(fp_obj.n):
        cur_dist = np.linalg.norm(anchor_state - fp_obj[idx].xstar)
        if cur_dist < dist:
            dist = cur_dist
            chosen_fp = fp_obj[idx]
    assert chosen_fp is not None
    return chosen_fp


def _plot_interpolated_fp_object(ax, fps, pca_obj, mrnn, i, dh, colors):
    for n in range(fps.n):
        zstar = pca_obj.transform(fps[n].xstar)
        linear = mLinearization(mrnn)
        input = torch.zeros(size=(mrnn.total_num_inputs,))
        real, _, _ = linear.eigendecomposition(
            input, fps[n].xstar, mrnn.activation(fps[n].xstar), dh=dh
        )
        # Stability of top eigenvalue
        stability = max(np.abs(real))
        if stability > 1:
            ax.scatter(
                i / 20,
                zstar[:, 0],
                zstar[:, 1],
                marker=".",
                alpha=0.75,
                edgecolors=colors[i],
                facecolors="w",
                s=250,
            )
        else:
            ax.scatter(
                i / 20,
                zstar[:, 0],
                zstar[:, 1],
                marker=".",
                alpha=0.75,
                color=colors[i],
                s=250,
            )


def _interpolate_inputs(input_1, input_2, steps=20):
    interpolated_input = input_1.unsqueeze(0) + torch.linspace(
        0, 1, steps=steps
    ).unsqueeze(1) * (input_2 - input_1).unsqueeze(0)
    return interpolated_input


def _build_interpolation_input(input_1, input_2, input_component):
    if input_component is None:
        interpolated_input = _interpolate_inputs(input_1, input_2)
    elif input_component == "rule":
        interpolated_input_rule = _interpolate_inputs(input_1[:10], input_2[:10])
        fixed_inp = input_1[10:].repeat(20, 1)
        interpolated_input = torch.cat([interpolated_input_rule, fixed_inp], dim=1)
    elif input_component == "proprioception":
        interpolated_input_proprioception = _interpolate_inputs(
            input_1[16:], input_2[16:]
        )
        fixed_inp_pre = input_1[:16].repeat(20, 1)
        interpolated_input = torch.cat(
            [fixed_inp_pre, interpolated_input_proprioception], dim=1
        )
    else:
        raise ValueError
    return interpolated_input


# ------------------- Primary Experiments ---------------------------


def interpolated_fps(
    model_name,
    task1,
    task2,
    epoch,
    task1_period="all",
    task2_period="all",
    input_component=None,
    add_new_rule_inputs=False,
    num_new_inputs=10,
):
    # Task 1 and task 2 period represent either collecting only half of the movement, or all, in specific periods
    # These values can be first, second, or all

    model_path = f"checkpoints/{model_name}"
    test = Test(
        model_path,
        model_name,
        add_new_rule_inputs=add_new_rule_inputs,
        num_new_inputs=num_new_inputs,
    )

    NOISE_SCALE = 0.5  # Standard deviation of noise added to initial states
    N_INITS = 1024  # The number of initial states to provide

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {
        "batch_size": 16,
        "reach_conds": torch.arange(0, 32, 2),
        "delay_cond": 1,
        "speed_cond": 5,
    }

    trial_data1 = test.trial(
        options,
        env_dict[task1],
    )
    trial_data2 = test.trial(
        options,
        env_dict[task2],
    )

    if epoch == "delay":
        # Get inputs and x and h from desired timepoint
        inp1 = trial_data1["obs"][:, trial_data1["epoch_bounds"]["delay"][1] - 1]
        inp2 = trial_data2["obs"][:, trial_data2["epoch_bounds"]["delay"][1] - 1]
    elif epoch == "movement":
        inp1 = get_interpolation_input(trial_data1, task1_period)
        inp2 = get_interpolation_input(trial_data2, task2_period)
    else:
        raise ValueError

    fpf_hps = {
        "max_iters": 250,
        "lr_init": 1.0,
        "outlier_distance_scale": 10.0,
        "verbose": True,
        "super_verbose": False,
        "tol_unique": 1,
        "do_compute_jacobians": True,
    }

    cond_fps_list = []
    for c, (cond1, cond2) in enumerate(zip(inp1, inp2)):
        # Setup environment and initialize it
        env = env_dict[task1](
            effector=mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        )
        # TODO get rid of useless resets by improving env init
        _, _ = env.reset(testing=True, options=options)

        middle_movement_1 = (
            env.epoch_bounds["movement"][0] + env.epoch_bounds["movement"][1]
        ) / 2

        if epoch == "delay":
            timesteps = env.epoch_bounds["delay"][1] - 1
        elif epoch == "movement":
            timesteps = middle_movement_1
        else:
            raise ValueError

        # Draw a line from fingertip to goal
        interpolated_input = _build_interpolation_input(cond1, cond2, input_component)

        fps_list = []
        fpf = mFixedPointFinder(test.policy.mrnn, **fpf_hps)

        # Going thorugh each interpolated input
        for i, inp in enumerate(interpolated_input):
            # Setup the fixed point finder
            init_state = trial_data1["h"][c, timesteps]

            """Draw random, noise corrupted samples of those state trajectories
            to use as initial states for the fixed point optimizations."""

            # Currently using original h for initial states
            initial_states = fpf.sample_states(
                init_state,
                n_inits=N_INITS,
                noise_scale=NOISE_SCALE,
            )

            # Run the fixed point finder
            unique_fps, _ = fpf.find_fixed_points(initial_states, inp[None, :])

            # Add fixed points and their info to dict
            fps_list.append({"fps": unique_fps, "interp_point": i})

        cond_fps_list.append(fps_list)

    # Save all information of fps across tasks to pickle file
    save_name = f"interpolated_fps_{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}"
    fname = os.path.join(model_path, save_name + ".pkl")
    print("interpolated fps saved at {:s}".format(fname))
    with open(fname, "wb") as f:
        pickle.dump(cond_fps_list, f)


def max_eigenvalues(fp_dict, init_state):
    """
    Plot maximum eigenvalues of selected fixed points across interpolation steps.

    At each step, the fixed point closest to the previously chosen one (or a trajectory point for the first step)
    is selected. Plots per-condition eigenvalue traces and saves the figure.
    """
    # Generate a plot of max eigenvalues
    max_eigs_conds = []
    for c, cond in enumerate(fp_dict):
        # this gets the unique fixed points for this condition
        interpolated_fps = [fp_dict["fps"] for fp_dict in cond]
        max_eigs = []
        chosen_fps = []
        for i, fps_step in enumerate(interpolated_fps):
            if i == 0:
                chosen_fp = _choose_next_fp(init_state[c], fps_step)
            else:
                chosen_fp = _choose_next_fp(chosen_fps[i - 1], fps_step)
            chosen_fps.append(chosen_fp)
            max_eig = chosen_fp.eigval_J_xstar[0, 0].real
            max_eigs.append(max_eig)

        max_eigs_conds.append(
            [np.abs(max_eigs[i + 1] - max_eigs[i]) for i in range(len(max_eigs) - 1)]
        )
    return max_eigs_conds


def euc_dists(fp_dict, init_state):
    """
    Plot Euclidean distances between matched fixed points across interpolation steps.

    Uses nearest-neighbor matching across steps to construct a smooth path through fixed points.
    Saves the resulting distance plot.
    """
    # Generate a plot of fp distances
    euc_dists_conds = []
    for c, cond in enumerate(fp_dict):
        interpolated_fps = [unique_fps["fps"] for unique_fps in cond]
        chosen_fps = []
        for i, fps_step in enumerate(interpolated_fps):
            if i == 0:
                chosen_fp = _choose_next_fp(init_state[c], fps_step)
            else:
                chosen_fp = _choose_next_fp(chosen_fps[i - 1], fps_step)
            chosen_fps.append(chosen_fp)

        dist_list = [
            np.linalg.norm(chosen_fps[i + 1] - chosen_fps[i])
            for i in range(len(chosen_fps) - 1)
        ]
        euc_dists_conds.append(
            [np.abs(dist_list[i + 1] - dist_list[i]) for i in range(len(dist_list) - 1)]
        )
    return euc_dists_conds


def plot_interpolated_fps(
    model_name,
    task1,
    task2,
    epoch,
    task1_period="all",
    task2_period="all",
    input_component=None,
    add_new_rule_inputs=False,
    num_new_inputs=10,
    save_metrics=False,
    y_dist=1,
):
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
    - Uses PCA projections computed only from task1 trial data for FP projection.
    """

    # Setup paths and load fixed points
    model_path = f"checkpoints/{model_name}"
    load_name = os.path.join(
        model_path,
        f"interpolated_fps_{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}.pkl",
    )
    exp_path = f"results/{model_name}/compositionality/interpolated_fps"

    test = Test(
        model_path,
        model_name,
        add_new_rule_inputs=add_new_rule_inputs,
        num_new_inputs=num_new_inputs,
    )

    fps = load_pickle(load_name)

    options = {
        "batch_size": 16,
        "reach_conds": torch.arange(0, 32, 2),
        "delay_cond": 1,
        "speed_cond": 5,
    }

    # Get trial data from model
    trial_data1 = test.trial(
        options,
        env_dict[task1],
    )
    trial_data2 = test.trial(
        options,
        env_dict[task2],
    )

    trial_data1_h_epoch = trial_data1["h"][
        :, trial_data1["epoch_bounds"][epoch][0] : trial_data1["epoch_bounds"][epoch][1]
    ]
    trial_data2_h_epoch = trial_data2["h"][
        :, trial_data2["epoch_bounds"][epoch][0] : trial_data2["epoch_bounds"][epoch][1]
    ]
    halfway_task1 = int(trial_data1_h_epoch.shape[1] / 2)

    trajectory_point = trial_data1_h_epoch[:, halfway_task1]

    colors_conds = plt.cm.inferno(np.linspace(0, 1, 16))
    colors_alpha = plt.cm.magma(np.linspace(0, 1, 20))

    # ------------------- Plot Fixed Points ------------------

    two_task_pca = PCA(n_components=2)
    two_task_pca.fit(trial_data1_h_epoch.reshape((-1, trial_data1_h_epoch.shape[-1])))

    _, ax = ax_3d_no_grid()
    # cond is a list containing the fps for each interpolated rule input for a given condition
    for c, cond in enumerate(fps):
        interpolated_fps = [unique_fps["fps"] for unique_fps in cond]
        for i, fps_step in enumerate(interpolated_fps):
            _plot_interpolated_fp_object(
                ax,
                fps_step,
                two_task_pca,
                test.policy.mrnn,
                i,
                dh=True,
                colors=colors_alpha,
            )

    save_fig(
        os.path.join(
            exp_path,
            f"{task1}_{task2}",
            f"{epoch}",
            f"{input_component}",
            f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}",
        ),
        eps=True,
    )

    # ------------------- Plot Max Eigenvalues ------------------

    _, ax = standard_2d_ax()
    max_eigs_conds = max_eigenvalues(fps, trajectory_point)

    x = np.arange(1, 21)
    for c, eigs in enumerate(max_eigs_conds):
        ax.scatter(x, eigs, marker="o", color=colors_conds[c], s=200, alpha=0.75)
        ax.set_ylim((0.5, 1.2))

    save_fig(
        os.path.join(
            exp_path,
            f"{task1}_{task2}",
            f"{epoch}",
            f"{input_component}",
            f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}_max_eigs",
        ),
        eps=True,
    )

    # ------------------- Plot Euclidean Distances ------------------

    _, ax = standard_2d_ax()
    euc_dists_conds = euc_dists(fps, trajectory_point)
    x = np.arange(1, 20)
    for c, dists in enumerate(euc_dists_conds):
        ax.scatter(x, dists, marker="o", color=colors_conds[c], s=200, alpha=0.75)
        ax.set_ylim((0, y_dist))

    save_fig(
        os.path.join(
            exp_path,
            f"{task1}_{task2}",
            f"{epoch}",
            f"{input_component}",
            f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}_dists",
        ),
        eps=True,
    )

    # ------------------- Plot 3D PCA Trajectories ------------------

    _, ax = empty_3d()

    trial_data1_h_epoch = split_movement_epoch(trial_data1, task1_period, "h")
    trial_data2_h_epoch = split_movement_epoch(trial_data2, task2_period, "h")

    # Get trajectories during task period
    combined_tasks = torch.cat([trial_data1_h_epoch, trial_data2_h_epoch], dim=1)

    two_task_pca_3d = PCA(n_components=3)
    two_task_pca_3d.fit(combined_tasks.reshape((-1, combined_tasks.shape[-1])))

    trial_data_1_projected = two_task_pca_3d.transform(
        trial_data1_h_epoch.reshape((-1, trial_data1_h_epoch.shape[-1]))
    )
    trial_data_2_projected = two_task_pca_3d.transform(
        trial_data2_h_epoch.reshape((-1, trial_data2_h_epoch.shape[-1]))
    )

    trial_data_1_projected = trial_data_1_projected.reshape(
        (trial_data1_h_epoch.shape[0], trial_data1_h_epoch.shape[1], 3)
    )

    trial_data_2_projected = trial_data_2_projected.reshape(
        (trial_data2_h_epoch.shape[0], trial_data2_h_epoch.shape[1], 3)
    )

    _3d_pca_across_conditions(ax, trial_data_1_projected)
    _3d_pca_across_conditions(ax, trial_data_2_projected)

    save_fig(
        os.path.join(
            exp_path,
            f"{task1}_{task2}",
            f"{epoch}",
            f"{input_component}",
            f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}_pca",
        ),
        eps=True,
    )
