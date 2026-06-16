import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import motornet as mn
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
    get_middle_movement,
    load_pickle,
    extension_dict,
    retraction_dict,
)
import itertools
import seaborn as sns
from DSA import DSA
import scipy
from modules.test import Test
from modules.multitask_training import MultitaskTrainer
from utils.plot_utils import (
    save_fig,
    standard_2d_ax,
    ax_3d_no_grid,
    no_ticks_2d_ax,
    empty_2d_ax,
    create_dir,
)

plt.rcParams.update({"font.size": 18})  # Sets default font size for all text


def get_mean_act(
    model_name,
    epoch,
    movement_type,
    system,
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
        "deterministic": True,
        "speed_cond": speed_cond,
        "delay_cond": delay_cond,
    }

    env_hs = []

    if movement_type == "extension":
        envs_to_use = extension_dict.copy()
    elif movement_type == "extension_retraction":
        envs_to_use = retraction_dict.copy()
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
            h = trial_data[system][:, end_stable]
        elif epoch == "delay":
            h = trial_data[system][:, end_delay]
        elif epoch == "hold":
            h = trial_data[system][:, end_hold]
        elif epoch == "extension" and movement_type == "extension_retraction":
            middle_movement = get_middle_movement(trial_data)
            h = trial_data[system][:, middle_movement]
        elif epoch == "extension" and movement_type == "extension":
            h = trial_data[system][:, end_movement]
        elif epoch == "retraction":
            assert movement_type == "extension_retraction"
            h = trial_data[system][:, end_movement]
        else:
            raise ValueError

        env_hs.append(h.mean(dim=0).unsqueeze(0))

    return env_hs, envs_to_use


def epoch_pcs(
    model_name, epoch, movement_type, system, add_new_rule_inputs=False, plot_3d=False
):
    exp_path = f"results/{model_name}/compositionality/pcs"
    create_dir(exp_path)

    assert system == "h" or system == "muscle_acts"

    env_hs, envs_to_use = get_mean_act(
        model_name,
        epoch,
        movement_type,
        system,
        add_new_rule_inputs=add_new_rule_inputs,
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
    save_fig(os.path.join(exp_path, system, f"{epoch}_{movement_type}_pcs"), eps=True)


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


def _replace_rule_input(composite_inp, obs):
    B, _ = obs.shape
    # This will not account for greater than two
    if composite_inp.dim() < 2:
        composite_inp = composite_inp.repeat(B, 1)
    obs_new_rule = torch.cat([composite_inp, obs[:, 10:]], dim=1)
    return obs_new_rule


def _create_composite_input(desired_movement, options):
    # desired movement represents the end result kinematic (say a curved reach)
    # The composite inputs that generate that movement will be manually crafted and tested

    # Desired movement can be any movement except reach and reachback
    if desired_movement == "Reach":
        reach_inp = torch.zeros(size=(options["batch_size"], 1))
        clkcr_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        cclkcr_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        sin_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        invsin_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        other_inputs = torch.zeros(size=(options["batch_size"], 5))

        optimizer = optim.Adam([clkcr_inp, cclkcr_inp, sin_inp, invsin_inp], lr=1e-1)

    # Desired movement can be any movement except reach and reachback
    elif desired_movement == "ClkCurvedReach":
        reach_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        clkcr_inp = torch.zeros(size=(options["batch_size"], 1))
        cclkcr_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        sin_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        invsin_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        other_inputs = torch.zeros(size=(options["batch_size"], 5))

        optimizer = optim.Adam([reach_inp, cclkcr_inp, sin_inp, invsin_inp], lr=1e-1)

    elif desired_movement == "CClkCurvedReach":
        reach_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        clkcr_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        cclkcr_inp = torch.zeros(size=(options["batch_size"], 1))
        sin_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        invsin_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        other_inputs = torch.zeros(size=(options["batch_size"], 5))

        optimizer = optim.Adam([reach_inp, clkcr_inp, sin_inp, invsin_inp], lr=1e-1)

    elif desired_movement == "Sinusoid":
        reach_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        clkcr_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        cclkcr_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        sin_inp = torch.zeros(size=(options["batch_size"], 1))
        invsin_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        other_inputs = torch.zeros(size=(options["batch_size"], 5))

        optimizer = optim.Adam([reach_inp, clkcr_inp, cclkcr_inp, invsin_inp], lr=1e-1)

    elif desired_movement == "InvSinusoid":
        reach_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        clkcr_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        cclkcr_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        sin_inp = nn.Parameter(torch.ones(size=(options["batch_size"], 1)))
        invsin_inp = torch.zeros(size=(options["batch_size"], 1))
        other_inputs = torch.zeros(size=(options["batch_size"], 5))

        optimizer = optim.Adam([reach_inp, clkcr_inp, cclkcr_inp, sin_inp], lr=1e-1)
    else:
        raise ValueError

    return (
        reach_inp,
        clkcr_inp,
        cclkcr_inp,
        sin_inp,
        invsin_inp,
        other_inputs,
        optimizer,
    )


def composite_input_optimization(
    model_path,
    model_name,
    options,
    env,
    desired_movement,
    noise=False,
    noise_act=0.1,
    noise_inp=0.01,
    add_new_rule_inputs=False,
    num_new_inputs=10,
    num_iters=250,
):
    """
    Runs a test episode in the specified environment using a trained RNN or GRU policy.

    Parameters:
    -----------
    model_path : str
        Path to the trained model directory.
    model_file : str
        Filename of the model checkpoint.
    options : dict
        Dictionary of environment and test configuration options (e.g., batch size).
    env : callable
        Environment constructor. Should accept an effector keyword argument.
    stim : torch.Tensor, optional
        Tensor to silence or stimulate specific hidden units. Default is None.
    noise : bool, optional
        Whether to inject noise into the network. Default is False.
    noise_act : float, optional
        Standard deviation of noise added to activations. Default is 0.1.
    noise_inp : float, optional
        Standard deviation of noise added to inputs. Default is 0.01.
    add_new_rule_inputs : bool, optional
        Whether to add additional rule inputs to the RNN. Default is False.
    num_new_inputs : int, optional
        Number of new rule inputs to add if `add_new_rule_inputs` is True. Default is 10.

    Returns:
    --------
    trial_data : dict
        Dictionary containing recorded trajectories and internal states for the episode, including:
            - 'h', 'x': hidden and internal states
            - 'action': actions taken by the model
            - 'obs': observations received
            - 'xy', 'tg': fingertip positions and target positions
            - 'muscle_acts': muscle activations
            - 'epoch_bounds': dictionary from the environment
    """

    test = Test(
        model_path,
        model_name,
        noise_level_act=noise_act,
        noise_level_inp=noise_inp,
        device="cpu",
        add_new_rule_inputs=add_new_rule_inputs,
        num_new_inputs=num_new_inputs,
    )
    test.batch_size = options["batch_size"]
    policy = test.policy

    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    inp1, inp2, inp3, inp4, inp5, inp6, optimizer = _create_composite_input(
        desired_movement, options
    )

    best_trial_data = {}
    best_loss = np.inf
    training_losses = []

    print(f"\nBeginning optimization for {desired_movement}")

    for it in range(num_iters):
        # initialize batch
        x = torch.zeros(size=(test.batch_size, test.hid_size))
        h = torch.zeros(size=(test.batch_size, test.hid_size))

        env_tmp = env(effector=effector, zero_feedback=test.zero_feedback)
        obs, info = env_tmp.reset(testing=True, options=options)

        terminated = False
        trial_data = {}
        timesteps = 0

        trial_data["h"] = []
        trial_data["x"] = []
        trial_data["action"] = []
        trial_data["muscle_acts"] = []
        trial_data["obs"] = []
        trial_data["xy"] = []
        trial_data["tg"] = []

        ep_t = 0
        # simulate whole episode
        while not terminated:
            # Check if doing composite inputs
            composite_inp = torch.cat([inp1, inp2, inp3, inp4, inp5, inp6], dim=1)
            obs = _replace_rule_input(composite_inp, obs)

            x, h, action = policy(obs, x, h, noise=noise)

            # Take step in motornet environment
            obs, _, terminated, info = env_tmp.step(timesteps, action=action)

            timesteps += 1

            # Save all information regarding episode step
            trial_data["h"].append(h.unsqueeze(1))  # trajectories
            trial_data["x"].append(x.unsqueeze(1))  # trajectories
            trial_data["action"].append(action.unsqueeze(1))  # targets
            trial_data["obs"].append(obs.unsqueeze(1))  # targets
            trial_data["xy"].append(
                info["states"]["fingertip"][:, None, :]
            )  # trajectories
            trial_data["tg"].append(info["goal"][:, None, :])  # targets
            trial_data["muscle_acts"].append(
                info["states"]["muscle"][:, 0].unsqueeze(1)
            )

            # small check just in case something goes wrong in here
            ep_t += 1
            if ep_t > 10000:
                terminated = True

        # Concatenate all data into single tensor
        for key in trial_data:
            trial_data[key] = torch.cat(trial_data[key], dim=1)

        loss = MultitaskTrainer.l1_dist(
            trial_data["xy"], trial_data["tg"]
        )  # L1 loss on position
        print(f"loss at iteration {it}: {loss.item()}")
        training_losses.append(loss.item())

        trial_data["rule_input"] = composite_inp.detach().clone()

        if loss < best_loss:
            best_trial_data = trial_data
            best_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for key in trial_data:
        best_trial_data[key] = best_trial_data[key].detach()
    best_trial_data["epoch_bounds"] = env_tmp.epoch_bounds
    best_trial_data["test_loss"] = training_losses

    return best_trial_data


def test_sequential_inputs(
    model_path,
    model_name,
    options,
    extension,
    retraction,
    noise=False,
    noise_act=0.1,
    noise_inp=0.01,
):
    """
    Runs a test episode in the specified environment using a trained RNN or GRU policy.

    Parameters:
    -----------
    model_path : str
        Path to the trained model directory.
    model_file : str
        Filename of the model checkpoint.
    options : dict
        Dictionary of environment and test configuration options (e.g., batch size).
    env : callable
        Environment constructor. Should accept an effector keyword argument.
    stim : torch.Tensor, optional
        Tensor to silence or stimulate specific hidden units. Default is None.
    noise : bool, optional
        Whether to inject noise into the network. Default is False.
    noise_act : float, optional
        Standard deviation of noise added to activations. Default is 0.1.
    noise_inp : float, optional
        Standard deviation of noise added to inputs. Default is 0.01.
    add_new_rule_inputs : bool, optional
        Whether to add additional rule inputs to the RNN. Default is False.
    num_new_inputs : int, optional
        Number of new rule inputs to add if `add_new_rule_inputs` is True. Default is 10.

    Returns:
    --------
    trial_data : dict
        Dictionary containing recorded trajectories and internal states for the episode, including:
            - 'h', 'x': hidden and internal states
            - 'action': actions taken by the model
            - 'obs': observations received
            - 'xy', 'tg': fingertip positions and target positions
            - 'muscle_acts': muscle activations
            - 'epoch_bounds': dictionary from the environment
    """

    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    test = Test(
        model_path,
        model_name,
        noise_level_act=noise_act,
        noise_level_inp=noise_inp,
        device="cpu",
    )
    extension_env = extension(effector=effector, zero_feedback=test.zero_feedback)
    retraction_env = retraction(effector=effector, zero_feedback=test.zero_feedback)
    test.batch_size = options["batch_size"]
    policy = test.policy

    # initialize batch
    x = torch.zeros(size=(test.batch_size, test.hid_size))
    h = torch.zeros(size=(test.batch_size, test.hid_size))

    obs, info = retraction_env.reset(testing=True, options=options)
    _, _ = extension_env.reset(testing=True, options=options)

    extension_rule_input = extension_env.rule_input
    retraction_rule_input = retraction_env.rule_input

    middle_movement = int(
        (
            retraction_env.epoch_bounds["movement"][1]
            + retraction_env.epoch_bounds["movement"][0]
        )
        / 2
    )
    terminated = False
    trial_data = {}
    timesteps = 0

    trial_data["h"] = []
    trial_data["x"] = []
    trial_data["action"] = []
    trial_data["muscle_acts"] = []
    trial_data["obs"] = []
    trial_data["xy"] = []
    trial_data["tg"] = []

    # simulate whole episode
    while not terminated:  # will run until `max_ep_duration` is reached
        with torch.no_grad():
            if timesteps < middle_movement:
                obs = _replace_rule_input(extension_rule_input, obs)

            if timesteps >= middle_movement:
                obs = _replace_rule_input(retraction_rule_input, obs)

            x, h, action = policy(obs, x, h, noise=noise)

            # Take step in motornet environment
            obs, _, terminated, info = retraction_env.step(timesteps, action=action)

        timesteps += 1

        # Save all information regarding episode step
        trial_data["h"].append(h.unsqueeze(1))  # trajectories
        trial_data["x"].append(x.unsqueeze(1))  # trajectories
        trial_data["action"].append(action.unsqueeze(1))  # targets
        trial_data["obs"].append(obs.unsqueeze(1))  # targets
        trial_data["xy"].append(info["states"]["fingertip"][:, None, :])  # trajectories
        trial_data["tg"].append(info["goal"][:, None, :])  # targets
        trial_data["muscle_acts"].append(info["states"]["muscle"][:, 0].unsqueeze(1))

    # Concatenate all data into single tensor
    for key in trial_data:
        trial_data[key] = torch.cat(trial_data[key], dim=1)

    loss = MultitaskTrainer.l1_dist(
        trial_data["xy"], trial_data["tg"]
    )  # L1 loss on position
    trial_data["test_loss"] = loss

    trial_data["epoch_bounds"] = retraction_env.epoch_bounds

    return trial_data


def sequential_input_kinematics(model_name, extension, retraction):
    model_path = f"checkpoints/{model_name}"
    exp_path = (
        f"results/{model_name}/compositionality/sequential_rule_inputs/kinematics"
    )
    colors_conds = plt.cm.inferno(np.linspace(0, 1, 8))
    options = {
        "batch_size": 4,
        "reach_conds": np.arange(0, 32, 8),
        "speed_cond": 5,
        "deterministic": True,
        "delay_cond": 2,
    }

    extension_env = env_dict[extension]
    retraction_env = env_dict[retraction]

    trial_data = test_sequential_inputs(
        model_path, model_name, options, extension_env, retraction_env
    )
    kinematics = trial_data["xy"]
    middle_movement = get_middle_movement(trial_data)

    _, ax = empty_2d_ax()
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
