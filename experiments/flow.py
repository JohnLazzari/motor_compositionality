import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp, interpolate_trial

import warnings
warnings.filterwarnings("ignore")

from train import train_2link
import motornet as mn
from model import RNNPolicy, GRUPolicy
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
from exp_utils import _test, env_dict

# TODO change for new setting with stable and hold epochs
def plot_flow_fields(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/flow"

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "delay_cond": 1, "speed_cond": 5}

    hp = load_hp(model_path)
    
    device = "cpu"
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
            device=device
        )
    elif hp["network"] == "gru":
        policy = GRUPolicy(hp["inp_size"], hp["hid_size"], effector.n_muscles, batch_first=True)
    else:
        raise ValueError("Not a valid architecture")

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=torch.device('cpu'))
    policy.load_state_dict(checkpoint['agent_state_dict'])

    env_fps = {}

    font = {'size' : 18}
    plt.rcParams['figure.figsize'] = [4, 4]
    plt.rcParams['axes.linewidth'] = 1 # set the value globally
    plt.rc('font', **font)

    for env in env_dict:
        trial_data = _test(model_path, model_file, options, env=env_dict[env])
        env_pca = PCA(n_components=2)
        for b, condition in enumerate(trial_data["h"]):

            print(f"Env: {env},  Condition: {b}")

            env_pca.fit(trial_data["h"][b])
            reduced_h = env_pca.transform(condition)
            coords, x_vel, y_vel, speed = flow_field(
                policy.mrnn, 
                trial_data["h"][b:b+1], 
                trial_data["obs"][b:b+1], 
                time_skips=10, 
                x_offset=30, 
                y_offset=30
            )

            for t, coords_t in enumerate(coords):
                # Add line collection
                fig, ax = plt.subplots()
                # Create plot
                ax.streamplot(
                    coords_t[:, :, 0], 
                    coords_t[:, :, 1], 
                    x_vel[t], 
                    y_vel[t], 
                    color=speed[t], 
                    cmap="plasma", 
                    linewidth=3, 
                    arrowsize=2, 
                    zorder=0,
                )

                ax.plot(reduced_h[:t*10, 0], reduced_h[:t*10, 1], linewidth=3)

                # Other plotting parameters
                ax.set_yticks([])
                ax.set_xticks([])
                plt.tight_layout()
                save_fig(os.path.join(exp_path, env, f"cond_{b}", f"t_{t}_flow.png"))





if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "plot_flow_fields":
        plot_flow_fields(args.model_name) 
    else:
        raise ValueError("Experiment not in this file")