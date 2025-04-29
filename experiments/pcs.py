import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp

import warnings
warnings.filterwarnings("ignore")

import torch
import os
from utils import load_hp, create_dir, save_fig, interpolate_trial
import matplotlib.pyplot as plt
import numpy as np
import config
import tqdm as tqdm
from sklearn.decomposition import PCA
from exp_utils import _test, env_dict

""" The functions here are currently doing pca on each environment then plotting, this may change
"""


def _get_pcs(model_name, epoch, env, system, batch_size=8, speed_cond=5):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    hp = load_hp(model_path)

    options = {
        "batch_size": batch_size, 
        "reach_conds": np.arange(0, 32, int(32 / batch_size)), 
        "speed_cond": speed_cond, 
        "delay_cond": 2
    }

    trial_data = _test(model_path, model_file, options, env=env_dict[env])

    if system == "neural":
        mode = "h"
        size = hp["hid_size"]
    elif system == "muscle":
        mode = "muscle_acts"
        size = 6

    if epoch == "delay":
        env_h = trial_data[mode][:, trial_data["epoch_bounds"]["delay"][0]:trial_data["epoch_bounds"]["delay"][1]]
    elif epoch == "movement":
        env_h = trial_data[mode][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
    else:
        raise ValueError("not valid epoch")

    pca_3d = PCA()
    pca_3d.fit(env_h.reshape((-1, size)))

    return pca_3d, env_h




def _plot_pca3d(model_name, epoch, system):

    exp_path = f"results/{model_name}/pca"
    create_dir(exp_path)

    for env in env_dict:

        pca_3d, env_h = _get_pcs(model_name, epoch, env, system)

        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, env_h.shape[0])) 
        # Create a figure
        fig = plt.figure()
        # Add a 3D subplot
        ax = fig.add_subplot(111, projection='3d')

        for i, h in enumerate(env_h):

            if i == 0:
                alpha = 1
            else:
                alpha = 0.5

            # transform
            h_proj = pca_3d.transform(h)

            # Plot the 3D line
            ax.plot(h_proj[:, 0], h_proj[:, 1], h_proj[:, 2], color=colors[i], linewidth=4, alpha=alpha)

            # Set labels for axes
            ax.set_title(f'{env} PCs')

            # Start and end points (start is triangle, end is x)
            ax.scatter(h_proj[0, 0], h_proj[0, 1], h_proj[0, 2], marker="^", color=colors[i], s=250, zorder=10, alpha=alpha)
            ax.scatter(h_proj[-1, 0], h_proj[-1, 1], h_proj[-1, 2], marker="X", color=colors[i], s=250, zorder=10, alpha=alpha)
        
        # Set background to white
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        save_fig(os.path.join(exp_path, "3d", system, f"{env}_{epoch}_trajectory.png"))


        plt.rc('figure', figsize=(4, 6))
        # Create a figure
        fig = plt.figure()
        # Add a 3D subplot
        ax = fig.add_subplot(111)
        ax.plot([np.sum(pca_3d.explained_variance_ratio_[:i]) for i in range(pca_3d.components_.shape[0])], marker='o', color="black", alpha=0.5, linewidth=4)
        save_fig(os.path.join(exp_path, "3d", system, f"{env}_{epoch}_var"))




def plot_neural_pca3d_delay(model_name):
    _plot_pca3d(model_name, "delay", "neural")
def plot_neural_pca3d_movement(model_name):
    _plot_pca3d(model_name, "movement", "neural")

def plot_motor_pca3d_delay(model_name):
    _plot_pca3d(model_name, "delay", "muscle")
def plot_motor_pca3d_movement(model_name):
    _plot_pca3d(model_name, "movement", "muscle")




def _plot_pca2d(model_name, epoch, system):

    exp_path = f"results/{model_name}/pca"
    create_dir(exp_path)

    for env in env_dict:

        pca_3d, env_h = _get_pcs(model_name, epoch, env, system)

        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, env_h.shape[0])) 
        # Create a figure
        fig = plt.figure()
        # Add a 3D subplot
        ax = fig.add_subplot(111)

        for i, h in enumerate(env_h):

            # transform
            h_proj = pca_3d.transform(h)
            # Plot the 3D line
            ax.plot(h_proj[:, 0], h_proj[:, 1], color=colors[i], linewidth=4)

            # Start and end positions
            ax.scatter(h_proj[0, 0], h_proj[0, 1], marker="^", color=colors[i], s=250, zorder=10)
            ax.scatter(h_proj[-1, 0], h_proj[-1, 1], marker="X", color=colors[i], s=250, zorder=10)
        
        # No ticks or tick labels
        ax.set_xticks([])
        ax.set_yticks([])

        # No axis spines (borders)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # No axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # No grid
        ax.grid(False)

        # White background (optional, usually default)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        save_fig(os.path.join(exp_path, "2d", system, f"{env}_{epoch}_trajectory.png"))

        plt.rc('figure', figsize=(4, 6))
        # Create a figure
        fig = plt.figure()
        # Add a 3D subplot
        ax = fig.add_subplot(111)
        ax.plot([np.sum(pca_3d.explained_variance_ratio_[:i]) for i in range(pca_3d.components_.shape[0])], marker='o', color="black", alpha=0.5, linewidth=4)
        save_fig(os.path.join(exp_path, "3d", system, f"{env}_{epoch}_var"))




def plot_neural_pca2d_delay(model_name):
    _plot_pca2d(model_name, "delay", "neural")
def plot_neural_pca2d_movement(model_name):
    _plot_pca2d(model_name, "movement", "neural")

def plot_motor_pca2d_delay(model_name):
    _plot_pca2d(model_name, "delay", "muscle")
def plot_motor_pca2d_movement(model_name):
    _plot_pca2d(model_name, "movement", "muscle")




def plot_movement_vs_delay_space(model_name):
    
    exp_path = f"results/{model_name}/pca"
    create_dir(exp_path)

    for env in env_dict:

        pca_mov, env_h_mov = _get_pcs(model_name, "movement", env, "neural")
        pca_delay, env_h_delay = _get_pcs(model_name, "delay", env, "neural")

        all_h = torch.cat([env_h_delay, env_h_mov], dim=1)

        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, env_h_mov.shape[0])) 
        # Create a figure
        fig = plt.figure()
        # Add a 3D subplot
        ax = fig.add_subplot(111, projection="3d")

        for i, h in enumerate(all_h):

            # transform
            h_proj_mov = pca_mov.transform(h)
            h_proj_delay = pca_delay.transform(h)

            # Plot the 3D line
            ax.plot(h_proj_mov[:, 0], h_proj_mov[:, 1], h_proj_delay[:, 0], color=colors[i], linewidth=4)

            # Start and end positions
            ax.scatter(h_proj_mov[0, 0], h_proj_mov[0, 1], h_proj_delay[0, 0], marker="^", color=colors[i], s=250, zorder=10)
            ax.scatter(h_proj_mov[-1, 0], h_proj_mov[-1, 1], h_proj_delay[-1, 0], marker="X", color=colors[i], s=250, zorder=10)
        
        # No axis spines (borders)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # No axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # No grid
        ax.grid(False)

        # White background (optional, usually default)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        save_fig(os.path.join(exp_path, "move_vs_delay", f"{env}_trajectory.png"))




def plot_pca_speeds(model_name):

    exp_path = f"results/{model_name}/pca"
    create_dir(exp_path)

    # kind of useless but whatever
    speed_batches = []
    speed_batch_lengths = []
    for speed in range(10):
        _, env_hs = _get_pcs(model_name, batch_size=1, speed_cond=speed)
        speed_batch_lengths.append([env_h.shape[1] for env_h in env_hs])
        speed_batches.append(env_hs)
    
    max_timestep = max(speed_batch_lengths)

    for (env_data, env) in zip(interpolated_batches, env_dict):

        speed_pca = PCA(n_components=2)
        speed_pca.fit()

        movement_start = 75 - 25
        
        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, env_data.shape[0])) 

        # Create a figure
        fig = plt.figure()
        # Add a 3D subplot
        ax = fig.add_subplot(111)

        for i, h in enumerate(env_data):

            # transform
            h_proj = speed_pca.transform(h)
            # Plot the 3D line
            ax.plot(h_proj[:movement_start, 0], h_proj[:movement_start, 1], color=colors[i], linewidth=4, linestyle="dashed")
            ax.plot(h_proj[movement_start:, 0], h_proj[movement_start:, 1], color=colors[i], linewidth=4)

            # Start and end positions
            ax.scatter(h_proj[0, 0], h_proj[0, 1], marker="^", color=colors[i], s=250, zorder=10)
            ax.scatter(h_proj[-1, 0], h_proj[-1, 1], marker="X", color=colors[i], s=250, zorder=10)
        
        # No ticks or tick labels
        ax.set_xticks([])
        ax.set_yticks([])

        # No axis spines (borders)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # No axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # No grid
        ax.grid(False)

        # White background (optional, usually default)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        save_fig(os.path.join(exp_path, "2d", f"{env}_tg_trajectory.png"))





if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "plot_neural_pca2d_delay":
        plot_neural_pca2d_delay(args.model_name) 
    elif args.experiment == "plot_neural_pca2d_movement":
        plot_neural_pca2d_movement(args.model_name) 
    elif args.experiment == "plot_motor_pca2d_delay":
        plot_motor_pca2d_delay(args.model_name) 
    elif args.experiment == "plot_motor_pca2d_movement":
        plot_motor_pca2d_movement(args.model_name) 

    elif args.experiment == "plot_neural_pca3d_delay":
        plot_neural_pca3d_delay(args.model_name) 
    elif args.experiment == "plot_neural_pca3d_movement":
        plot_neural_pca3d_movement(args.model_name) 
    elif args.experiment == "plot_motor_pca3d_delay":
        plot_motor_pca3d_delay(args.model_name) 
    elif args.experiment == "plot_motor_pca3d_movement":
        plot_motor_pca3d_movement(args.model_name) 

    elif args.experiment == "plot_pca_speeds":
        plot_pca_speeds(args.model_name) 
    elif args.experiment == "plot_movement_vs_delay_space":
        plot_movement_vs_delay_space(args.model_name) 
    else:
        raise ValueError("Experiment not in this file")