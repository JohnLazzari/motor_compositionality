import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp, interpolate_trial

import warnings
warnings.filterwarnings("ignore")

from train import train_2link
import motornet as mn
from model import RNNPolicy, GRUPolicy, OrthogonalNet
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



env_dict = {
    "DlyHalfReach": DlyHalfReach, 
    "DlyHalfCircleClk": DlyHalfCircleClk, 
    "DlyHalfCircleCClk": DlyHalfCircleCClk, 
    "DlySinusoid": DlySinusoid, 
    "DlySinusoidInv": DlySinusoidInv,
    "DlyFullReach": DlyFullReach,
    "DlyFullCircleClk": DlyFullCircleClk,
    "DlyFullCircleCClk": DlyFullCircleCClk,
    "DlyFigure8": DlyFigure8,
    "DlyFigure8Inv": DlyFigure8Inv
}



def _test(
    model_path, 
    model_file, 
    options, 
    env, 
    stim=None, 
    feedback_mask=None, 
    noise=False, 
    noise_act=0.1, 
    noise_inp=0.01,
    add_new_rule_inputs=False,
    num_new_inputs=10
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
    feedback_mask : torch.Tensor, optional
        Mask to ablate parts of the observation vector. Default is None.
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

    device = "cpu"
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    env = env(effector=effector)

    hp = load_hp(model_path)
    hp = hp.copy()
    hp["batch_size"] = options["batch_size"]

    # Loading in model
    if hp["network"] == "rnn":
        policy = RNNPolicy(
            hp["inp_size"],
            hp["hid_size"],
            effector.n_muscles, 
            activation_name=hp["activation_name"],
            noise_level_act=noise_act, 
            noise_level_inp=noise_inp, 
            constrained=hp["constrained"], 
            dt=hp["dt"],
            t_const=hp["t_const"],
            device=device,
            add_new_rule_inputs=add_new_rule_inputs,
            num_new_inputs=num_new_inputs
        )
        checkpoint = torch.load(os.path.join(model_path, model_file), map_location=torch.device('cpu'))
        policy.load_state_dict(checkpoint['agent_state_dict'])
    elif hp["network"] == "gru":
        policy = GRUPolicy(hp["inp_size"], hp["hid_size"], effector.n_muscles, batch_first=True)
        checkpoint = torch.load(os.path.join(model_path, model_file), map_location=torch.device('cpu'))
        policy.load_state_dict(checkpoint['agent_state_dict'])
    else:
        raise ValueError("Not a valid architecture")

    # initialize batch
    x = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
    h = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
    
    obs, info = env.reset(testing=True, options=options)
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

        # Check if ablating feedback
        if feedback_mask is not None:
            obs = obs * feedback_mask

        with torch.no_grad():
            # Check if silencing units 
            if stim is not None:
                x, h, action = policy(obs, x, h, stim, noise=noise)
            else:
                x, h, action = policy(obs, x, h, noise=noise)

            # Take step in motornet environment
            obs, reward, terminated, info = env.step(timesteps, action=action)

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
    trial_data["epoch_bounds"] = env.epoch_bounds

    return trial_data




def split_movement_epoch(trial_data, task_period, system):

    """
    Extracts a portion of the movement epoch data from the specified system.

    Parameters:
    -----------
    trial_data : dict
        Contains 'epoch_bounds' and system data arrays.
    task_period : str
        One of "first", "second", or "all" to specify part of the movement epoch.
    system : str
        Key for the system data (e.g., "neural", "behavioral").

    Returns:
    --------
    trial_data_h_epoch : np.ndarray
        2D array of system data for the selected epoch segment.
    """

    middle_movement = int((trial_data["epoch_bounds"]["movement"][1] + trial_data["epoch_bounds"]["movement"][0]) / 2)

    if task_period == "first":
        trial_data_h_epoch = trial_data[system][:, trial_data["epoch_bounds"]["movement"][0]:middle_movement]
    elif task_period == "second":
        trial_data_h_epoch = trial_data[system][:, middle_movement:trial_data["epoch_bounds"]["movement"][1]]
    elif task_period == "all":
        trial_data_h_epoch = trial_data[system][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
    
    return trial_data_h_epoch




def get_interpolation_input(trial_data, task_period):

    """
    Returns a single observation from the movement epoch for interpolation.

    Parameters:
    -----------
    trial_data : dict
        Contains 'epoch_bounds' and 'obs' array.
    task_period : str
        One of "first", "second", or "all" to select the interpolation point.

    Returns:
    --------
    inp : np.ndarray
        1D array (features,) from the selected time point.
    """

    middle_movement = int((trial_data["epoch_bounds"]["movement"][1] + trial_data["epoch_bounds"]["movement"][0]) / 2)

    if task_period == "first":
        interpolation_point = int((middle_movement + trial_data["epoch_bounds"]["movement"][0]) / 2)
        inp = trial_data["obs"][:, interpolation_point]
    elif task_period == "second":
        interpolation_point = int((middle_movement + trial_data["epoch_bounds"]["movement"][1]) / 2)
        inp = trial_data["obs"][:, interpolation_point]
    elif task_period == "all":
        # This option only makes sense if using a half task
        inp = trial_data["obs"][:, middle_movement]
    
    return inp




def distances_from_combinations(combinations, batch_size):
    """
    Computes the Euclidean (L2) distances between pairs of tensors in each combination.

    Args:
        combinations (dict): Dictionary where each key is a combination (e.g., a tuple of condition identifiers),
                             and each value is a tuple of two tensors of shape [batch_size, ...].
        batch_size (int): Number of items per condition to compare.

    Returns:
        dict: Mapping from combination to a list of distances (float) per batch element.
    """
    dist = {}
    for combination in combinations:
        condition_list = []
        for c in range(batch_size):
            h1 = combinations[combination][0][c]
            h2 = combinations[combination][1][c]
            condition_dist = torch.linalg.norm(h1 - h2)
            condition_list.append(condition_dist.item())
        dist[combination] = condition_list

    return dist



def average_angular_distance(h1, h2):
    """
    Computes the average angular distance (in radians) between corresponding rows of two matrices.

    Args:
        h1 (Tensor): Tensor of shape [N, D], representing N vectors.
        h2 (Tensor): Tensor of shape [N, D], representing N vectors.

    Returns:
        Tensor: A scalar tensor representing the mean angular distance.
    """
    inner_products = torch.diagonal(h1 @ h2.T)
    norms = torch.linalg.norm(h1, dim=1) * torch.linalg.norm(h2, dim=1)
    cos_angles = inner_products / norms
    return torch.arccos(cos_angles).mean()




def angles_from_combinations(combinations, batch_size):
    """
    Computes average angular distances between tensor pairs in each combination.

    Args:
        combinations (dict): Dictionary where each key is a combination,
                             and each value is a tuple of two tensors [batch_size, N, D].
        batch_size (int): Number of items per condition.

    Returns:
        dict: Mapping from combination to a list of average angular distances (float) per batch element.
    """
    angles = {}
    for combination in combinations:
        condition_list = []
        for c in range(batch_size):
            h1 = combinations[combination][0][c]
            h2 = combinations[combination][1][c]
            condition_angle = average_angular_distance(h1, h2)
            condition_list.append(condition_angle.item())
        angles[combination] = condition_list

    return angles




def shapes_from_combinations(combinations, batch_size):
    """
    Computes Procrustes shape disparity between trajectory tensor pairs in each combination.

    Args:
        combinations (dict): Dictionary where each key is a combination,
                             and each value is a tuple of tensors [batch_size, T, D].
        batch_size (int): Number of trajectories per combination.

    Returns:
        dict: Mapping from combination to a list of Procrustes disparities (float) per trajectory pair.
    """
    traj_dists = {}
    for combination in combinations:
        condition_list = []
        for c in range(batch_size):
            h1 = combinations[combination][0][c]
            h2 = combinations[combination][1][c]
            _, _, disparity = scipy.spatial.procrustes(h1, h2)
            condition_list.append(disparity)
        traj_dists[combination] = condition_list

    return traj_dists




def convert_motif_dict_to_list(target_dict, data):
    """
    Flattens values in `data` corresponding to keys in `target_dict`, accounting for reversed key ordering.

    Args:
        target_dict (dict): Dictionary with keys (e.g., combinations) to look up in `data`.
        data (dict): Dictionary mapping combinations to lists of values.

    Returns:
        list: Flattened list of values corresponding to the combinations in `target_dict`.
    """
    target_data = []
    for combination in target_dict:
        if combination in data:
            target_data.extend(data[combination])
        # This is just in case the order of the combination is off
        elif (combination[1], combination[0]) in data:
            target_data.extend(data[(combination[1], combination[0])])
    return target_data




def pvalues(label_list, data_dict, metric):
    combination_labels = list(itertools.combinations(label_list, 2))
    print("\n")
    # Print out significance here
    for combination in combination_labels:
        result = scipy.stats.mannwhitneyu(data_dict[combination[0]], data_dict[combination[1]])
        pvalue = result[1]
        if pvalue < 0.001:
            pvalue_str = f"***, {pvalue}"
        elif pvalue < 0.01:
            pvalue_str = f"**, {pvalue}"
        elif pvalue < 0.05:
            pvalue_str = f"*, {pvalue}"
        else:
            pvalue_str = "Not Significant"
        print(f"pvalue for {combination[0]} and {combination[1]} in metric {metric} is: {pvalue_str}")
    print("\n")