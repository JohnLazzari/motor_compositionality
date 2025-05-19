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
    """ Function will save all relevant data from a test run of a given env

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        joint_state (_type_, optional): _description_. Defaults to None.
        env (str, optional): _description_. Defaults to "RandomReach".
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

