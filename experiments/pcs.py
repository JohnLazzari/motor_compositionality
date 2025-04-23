import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp

import warnings
warnings.filterwarnings("ignore")

import torch
import os
from utils import load_hp, create_dir, save_fig 
import matplotlib.pyplot as plt
import numpy as np
import config
import tqdm as tqdm
from sklearn.decomposition import PCA
from exp_utils import _test, env_dict


def _get_pcs(model_name, batch_size=8, epoch=None, use_reach_conds=True, speed_cond=5, delay_cond=1, noise=False):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    hp = load_hp(model_path)

    if use_reach_conds:
        reach_conds = torch.arange(0, 32, int(32 / batch_size))
    else:
        reach_conds = None

    options = {
        "batch_size": batch_size, 
        "reach_conds": reach_conds, 
        "speed_cond": speed_cond, 
        "delay_cond": delay_cond
    }

    env_hs = []
    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=noise)

        if epoch is None:
            env_hs.append(trial_data["h"][:, trial_data["epoch_bounds"]["delay"][0]:])
        elif epoch == "delay":
            env_hs.append(trial_data["h"][:, trial_data["epoch_bounds"]["delay"][1]-1].unsqueeze(1))
        elif epoch == "stable":
            env_hs.append(trial_data["h"][:, trial_data["epoch_bounds"]["stable"][1]-1].unsqueeze(1))
        elif epoch == "movement":
            env_hs.append(trial_data["h"][:, trial_data["epoch_bounds"]["movement"][1]-1].unsqueeze(1))
        else:
            raise ValueError("not valid epoch")

    pca_3d = PCA(n_components=3)
    pca_3d.fit(torch.cat(env_hs, dim=1).reshape((-1, hp["hid_size"])))

    return pca_3d, env_hs




def plot_pca3d(model_name):

    exp_path = f"results/{model_name}/pca"
    create_dir(exp_path)

    pca_3d, env_hs = _get_pcs(model_name)

    for (env_data, env) in zip(env_hs, env_dict):

        # shift start time because pca does not include stable period
        movement_start = 75 - 25
        
        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, env_data.shape[0])) 

        # Create a figure
        fig = plt.figure()
        # Add a 3D subplot
        ax = fig.add_subplot(111, projection='3d')

        for i, h in enumerate(env_data):

            # transform
            h_proj = pca_3d.transform(h)

            # Plot the 3D line
            ax.plot(h_proj[:movement_start, 0], h_proj[:movement_start, 1], h_proj[:movement_start, 2], color=colors[i], linewidth=4, linestyle="dashed")
            ax.plot(h_proj[movement_start:, 0], h_proj[movement_start:, 1], h_proj[movement_start:, 2], color=colors[i], linewidth=4)

            # Set labels for axes
            ax.set_title(f'{env} PCs')

            # Start and end points (start is triangle, end is x)
            ax.scatter(h_proj[0, 0], h_proj[0, 1], h_proj[0, 2], marker="^", color=colors[i], s=250, zorder=10)
            ax.scatter(h_proj[-1, 0], h_proj[-1, 1], h_proj[-1, 2], marker="X", color=colors[i], s=250, zorder=10)
        
        # Set background to white
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Remove 3D panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Remove axis lines ("spines")
        ax._axis3don = False

        save_fig(os.path.join(exp_path, "3d", f"{env}_tg_trajectory.png"))




def plot_pca2d(model_name):

    exp_path = f"results/{model_name}/pca"
    create_dir(exp_path)

    pca_3d, env_hs = _get_pcs(model_name)

    for (env_data, env) in zip(env_hs, env_dict):

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
            h_proj = pca_3d.transform(h)
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
    
    if args.experiment == "plot_pca2d":
        plot_pca2d(args.model_name) 
    elif args.experiment == "plot_pca3d":
        plot_pca3d(args.model_name) 
    else:
        raise ValueError("Experiment not in this file")