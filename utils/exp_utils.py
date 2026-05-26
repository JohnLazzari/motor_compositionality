import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import motornet as mn
import torch
import torch.nn as nn
import torch.optim as optim
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
import numpy as np
import tqdm as tqdm
import itertools
import scipy
from scipy.interpolate import interp1d
import pickle


env_dict = {
    "Reach": Reach,
    "ClkCurvedReach": ClkCurvedReach,
    "CClkCurvedReach": CClkCurvedReach,
    "Sinusoid": Sinusoid,
    "InvSinusoid": InvSinusoid,
    "ReachBack": ReachBack,
    "ClkCycle": ClkCycle,
    "CClkCycle": CClkCycle,
    "Figure8": Figure8,
    "InvFigure8": InvFigure8,
}

extension_dict = {
    "Reach": Reach,
    "ClkCurvedReach": ClkCurvedReach,
    "CClkCurvedReach": CClkCurvedReach,
    "Sinusoid": Sinusoid,
    "InvSinusoid": InvSinusoid,
}

retraction_dict = {
    "ReachBack": ReachBack,
    "ClkCycle": ClkCycle,
    "CClkCycle": CClkCycle,
    "Figure8": Figure8,
    "InvFigure8": InvFigure8,
}


def save_pickle(file, obj):
    # save hyperparameters
    with open(file, "wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_pickle(file):
    with open(file, "rb") as f:
        # Load the data from the file
        mult_train = pickle.load(f)
    return mult_train


def mov_bounds(trial_data):
    """
    Helper function to get movement epoch bounds
    """
    mov_beg = trial_data["epoch_bounds"]["movement"][0]
    mov_end = trial_data["epoch_bounds"]["movement"][1]
    return mov_beg, mov_end


def delay_bounds(trial_data):
    """
    Helper function to get delay epoch bounds
    """
    delay_beg = trial_data["epoch_bounds"]["movement"][0]
    delay_end = trial_data["epoch_bounds"]["movement"][1]
    return delay_beg, delay_end


def unique_pairs(labels, data):
    """
    Helper function to get unique pairs from data using labels
    """
    # Get all unique pairs of unit activity across tasks
    combinations = []
    for label in labels:
        combinations.append(
            (
                data[label[0]],
                data[label[1]],
            )
        )
    return combinations


def unique_pairs_dict(labels, data):
    """
    Helper function to get unique pairs from data using labels
    """
    # Get all unique pairs of unit activity across tasks
    combinations = {}
    for label in labels:
        combinations[label] = (
            data[label[0]],
            data[label[1]],
        )

    return combinations


def test_sequential_inputs(
    model_path,
    model_file,
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
    extension_env = extension(effector=effector)
    retraction_env = retraction(effector=effector)

    hp = load_hp(model_path)
    hp = hp.copy()
    hp["batch_size"] = options["batch_size"]

    policy = load_model(model_path, model_file, hp, noise_act, noise_inp, device)

    # initialize batch
    x = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
    h = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))

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

            if timesteps == middle_movement:
                for t in range(2):
                    obs = _replace_rule_input(retraction_rule_input, obs)
                    x, h, _ = policy(obs, x, h, noise=noise)

            x, h, action = policy(obs, x, h, noise=noise)

            # Take step in motornet environment
            obs, reward, terminated, info = retraction_env.step(
                timesteps, action=action
            )

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

    loss = l1_dist(trial_data["xy"], trial_data["tg"])  # L1 loss on position
    trial_data["test_loss"] = loss

    trial_data["epoch_bounds"] = retraction_env.epoch_bounds

    return trial_data


def _replace_rule_input(composite_inp, obs):
    B, N = obs.shape
    # This will not account for greater than two
    if composite_inp.dim() < 2:
        composite_inp = composite_inp.repeat(B, 1)
    obs_new_rule = torch.cat([composite_inp, obs[:, 10:]], dim=1)
    return obs_new_rule


def get_middle_movement(trial_data):
    return int(
        (
            trial_data["epoch_bounds"]["movement"][1]
            + trial_data["epoch_bounds"]["movement"][0]
        )
        / 2
    )


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

    middle_movement = get_middle_movement(trial_data)

    if task_period == "first":
        trial_data_h_epoch = trial_data[system][
            :, trial_data["epoch_bounds"]["movement"][0] : middle_movement
        ]
    elif task_period == "second":
        trial_data_h_epoch = trial_data[system][
            :, middle_movement : trial_data["epoch_bounds"]["movement"][1]
        ]
    elif task_period == "all":
        trial_data_h_epoch = trial_data[system][
            :,
            trial_data["epoch_bounds"]["movement"][0] : trial_data["epoch_bounds"][
                "movement"
            ][1],
        ]
    else:
        raise ValueError

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

    middle_movement = int(
        (
            trial_data["epoch_bounds"]["movement"][1]
            + trial_data["epoch_bounds"]["movement"][0]
        )
        / 2
    )

    if task_period == "extension":
        interpolation_point = int(
            (middle_movement + trial_data["epoch_bounds"]["movement"][0]) / 2
        )
        inp = trial_data["obs"][:, interpolation_point]
    elif task_period == "retraction":
        interpolation_point = int(
            (middle_movement + trial_data["epoch_bounds"]["movement"][1]) / 2
        )
        inp = trial_data["obs"][:, interpolation_point]
    elif task_period == "all":
        # This option only makes sense if using a half task
        inp = trial_data["obs"][:, middle_movement]
    else:
        raise ValueError

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
        result = scipy.stats.mannwhitneyu(
            data_dict[combination[0]], data_dict[combination[1]]
        )
        pvalue = result[1]
        if pvalue < 0.001:
            pvalue_str = f"***, {pvalue}"
        elif pvalue < 0.01:
            pvalue_str = f"**, {pvalue}"
        elif pvalue < 0.05:
            pvalue_str = f"*, {pvalue}"
        else:
            pvalue_str = "Not Significant"
        print(
            f"pvalue for {combination[0]} and {combination[1]} in metric {metric} is: {pvalue_str}"
        )
    print("\n")


def interpolate_trial(ys, desired_x):
    """
    ys is the time series [timesteps, neurons]
    desired x is the total number of points desired after interpolating
    """
    # range for x is somewhat arb, going with 0-1
    xs = torch.linspace(0, 1, ys.shape[0])
    new_xs = torch.linspace(0, 1, desired_x)

    int_neurons = []
    # Loop through each neuron to get single timeseries
    for n in range(ys.shape[1]):
        t_series = interp1d(xs, ys[:, n])(new_xs)
        int_neurons.append(torch.tensor(t_series))
    new_t_series = torch.stack(int_neurons, dim=1)
    return new_t_series


def random_orthonormal_basis(n, seed=None):
    """
    Generate an n‑dimensional random orthonormal basis.

    Parameters
    ----------
    n : int
        Dimension of the ambient space (must be ≥ 1).
    seed : int, optional
        Random‑seed for reproducibility.

    Returns
    -------
    Q : ndarray, shape (n, n)
        Columns form an orthonormal basis (QᵀQ = I).
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))  # random matrix
    Q, _ = np.linalg.qr(A)  # QR factorization ⇒ Q is orthonormal

    # Fix possible sign ambiguity so the first non‑zero entry in each column is positive
    # (optional, just for consistency)
    signs = np.sign(np.diag(Q.T @ A))
    Q *= signs

    return Q
