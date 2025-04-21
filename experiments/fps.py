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


def compute_fps(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"

    NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
    N_INITS = 1024 # The number of initial states to provide

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

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env])

        '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
        descriptions of available hyperparameters.'''
        fpf_hps = {
            'max_iters': 250,
            'lr_init': 1.,
            'outlier_distance_scale': 10.0,
            'verbose': False, 
            'super_verbose': False,
            'tol_unique': 2,
            'do_compute_jacobians': False}
        
        env_fps_list = []
        for b, condition in enumerate(trial_data["h"]):
            for t, timepoint in enumerate(condition):
                if t % 10 == 0:

                    print(f"Env: {env},  Condition: {b},  Timepoint: {t}")

                    # Setup the fixed point finder
                    fpf = FixedPointFinder(policy.mrnn, **fpf_hps)

                    '''Draw random, noise corrupted samples of those state trajectories
                    to use as initial states for the fixed point optimizations.'''
                    initial_states = fpf.sample_states(timepoint[None, None, :],
                        n_inits=N_INITS,
                        noise_scale=NOISE_SCALE)

                    # Run the fixed point finder
                    unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs=trial_data["obs"][b, t:t+1, :])

                    # Add fixed points and their info to dict
                    env_fps_list.append({"fps": unique_fps, "t": t, "condition": b, "state_traj": condition})

        # Save all fixed points for environment
        env_fps[env] = env_fps_list

    # Save all information of fps across tasks to pickle file
    save_name = 'model_fps'
    fname = os.path.join(model_path, save_name + '.pkl')
    print('Variance saved at {:s}'.format(fname))
    with open(fname, 'wb') as f:
        pickle.dump(env_fps, f)





# TODO make this much better
def plot_fps(model_name):

    model_path = f"checkpoints/{model_name}"
    load_name = os.path.join(model_path, "model_fps.pkl")
    exp_path = f"results/{model_name}/fps"

    fps = load_pickle(load_name)

    colors = plt.cm.inferno(np.linspace(0, 1, 8)) 

    half_envs = [
        "DlyHalfReach", 
        "DlyHalfCircleClk", 
        "DlyHalfCircleCClk", 
        "DlySinusoid", 
        "DlySinusoidInv",
    ]
    full_envs = [
        "DlyFullReach",
        "DlyFullCircleClk",
        "DlyFullCircleCClk",
        "DlyFigure8",
        "DlyFigure8Inv"
    ]

    for env in fps:
        if env in half_envs:
            timepoints = [150]
        elif env in full_envs:
            timepoints = [150]
        env_fps = fps[env]
        for i, t in enumerate(timepoints):
            all_condition_fps = []
            all_condition_trajs = []
            for fp in env_fps:
                if fp["t"] == t:
                    all_condition_fps.append(fp["fps"])
                    all_condition_trajs.append(fp["state_traj"])
            
            all_condition_trajs = torch.stack(all_condition_trajs)
            save_name = f"{env}_t{t}_fps.png"
            # Visualize identified fixed points with overlaid RNN state trajectories
            # All visualized in the 3D PCA space fit the the example RNN states.
            start_traj = 0 if i == 0 else timepoints[i-1]
            end_traj = t
            fig=None
            for cond, (unique_fps, state_traj) in enumerate(zip(all_condition_fps, all_condition_trajs)):
                fig = plot_utils.plot_fps(unique_fps, pca_traj=all_condition_trajs, state_traj=state_traj[None, ...], plot_start_time=start_traj, plot_stop_time=end_traj, fig=fig, traj_color=colors[cond])
            save_fig(exp_path + "/" + save_name)






if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "compute_fps":
        compute_fps(args.model_name) 
    elif args.experiment == "plot_fps":
        plot_fps(args.model_name) 
    else:
        raise ValueError("Experiment not in this file")