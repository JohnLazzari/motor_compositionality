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
from analysis.jPCA import JPCA
from analysis.jpca_util import plot_projections
from exp_utils import _test, env_dict

""" The functions here are currently doing pca on each environment then plotting, this may change
"""

plt.rcParams.update({'font.size': 18})  # Sets default font size for all text


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

    return pca_3d, env_h, trial_data["epoch_bounds"]




def _plot_pca3d(model_name, epoch, system):

    exp_path = f"results/{model_name}/pca"
    create_dir(exp_path)

    for env in env_dict:

        pca_3d, env_h, _ = _get_pcs(model_name, epoch, env, system)

        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, env_h.shape[0])) 
        # Create a figure
        fig = plt.figure(figsize=(4, 4))
        # Add a 3D subplot
        ax = fig.add_subplot(111, projection='3d')

        all_proj = pca_3d.transform(env_h.reshape((-1, env_h.shape[-1])))
        min_z = np.min(all_proj[:, 2])

        for i, h in enumerate(env_h):

            # transform
            h_proj = pca_3d.transform(h)

            # Plot the 3D line
            ax.plot(h_proj[:, 0], h_proj[:, 1], h_proj[:, 2], color=colors[i], linewidth=4, alpha=0.75, zorder=10)
            ax.plot(h_proj[:, 0], h_proj[:, 1], min_z, color="grey", linewidth=2, alpha=0.5)

            # Start and end points (start is triangle, end is x)
            ax.scatter(h_proj[0, 0], h_proj[0, 1], h_proj[0, 2], marker="^", color=colors[i], s=250, zorder=10)
            ax.scatter(h_proj[-1, 0], h_proj[-1, 1], h_proj[-1, 2], marker="X", color=colors[i], s=250, zorder=10)
        
        # No grid
        ax.grid(False)
        # Set background to white
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        save_fig(os.path.join(exp_path, "3d", system, f"{env}_{epoch}_trajectory"), eps=True)

        plt.rc('figure', figsize=(3, 6))
        # Create a figure
        fig = plt.figure()
        # Add a 3D subplot
        ax = fig.add_subplot(111)

        if system == "neural":
            comps = 20
        elif system == "muscle":
            comps = 6

        for c, condition in enumerate(env_h):
            # transform
            cond_proj = pca_3d.transform(condition)
            total_var = condition.var(dim=0).sum()
            ax.plot(
                np.arange(1, comps+1),
                [cond_proj[:, :i].var(axis=0).sum() / total_var for i in range(comps)], 
                marker='o', color=colors[c], alpha=0.5, linewidth=4, markersize=20)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        save_fig(os.path.join(exp_path, "3d", system, f"{env}_{epoch}_var"), eps=True)




def plot_neural_pca3d_delay(model_name):
    _plot_pca3d(model_name, "delay", "neural")
def plot_neural_pca3d_movement(model_name):
    _plot_pca3d(model_name, "movement", "neural")

def plot_motor_pca3d_delay(model_name):
    _plot_pca3d(model_name, "delay", "muscle")
def plot_motor_pca3d_movement(model_name):
    _plot_pca3d(model_name, "movement", "muscle")




def _plot_movement_vs_delay_space(model_name, system):
    
    exp_path = f"results/{model_name}/pca"
    create_dir(exp_path)

    for env in env_dict:

        pca_mov, env_h_mov, epoch_bounds = _get_pcs(model_name, "movement", env, system)
        pca_delay, env_h_delay, epoch_bounds = _get_pcs(model_name, "delay", env, system)

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
            line, = ax.plot(
                h_proj_mov[epoch_bounds["delay"][0]-25:epoch_bounds["delay"][1]-25, 0], 
                h_proj_mov[epoch_bounds["delay"][0]-25:epoch_bounds["delay"][1]-25, 1], 
                h_proj_delay[epoch_bounds["delay"][0]-25:epoch_bounds["delay"][1]-25, 0], 
                color=colors[i], linewidth=4, linestyle="dashed", alpha=0.5)
            line.set_dashes([1, 1])
            ax.plot(
                h_proj_mov[epoch_bounds["movement"][0]-25:epoch_bounds["movement"][1]-25, 0], 
                h_proj_mov[epoch_bounds["movement"][0]-25:epoch_bounds["movement"][1]-25, 1], 
                h_proj_delay[epoch_bounds["movement"][0]-25:epoch_bounds["movement"][1]-25, 0], 
                color=colors[i], linewidth=4, alpha=0.5)

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

        ax.set_xlabel(f'Movement PC 1')
        ax.set_ylabel(f'Movement PC 2')
        ax.set_zlabel(f'Delay PC 1')
        ax.view_init(elev=10, azim=30)
        save_fig(os.path.join(exp_path, "move_vs_delay", system, f"{env}_trajectory.png"))




def plot_movement_vs_delay_space_neural(model_name):
    _plot_movement_vs_delay_space(model_name, "neural")
def plot_movement_vs_delay_space_movement(model_name):
    _plot_movement_vs_delay_space(model_name, "muscle")




def _plot_pca_speeds(model_name, epoch, system):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pca"
    create_dir(exp_path)

    env_speeds = {}
    for env in env_dict:
        speed_conds = []
        for speed in range(5):

            options = {
                "batch_size": 8, 
                "reach_conds": np.arange(0, 32, int(32 / 8)), 
                "speed_cond": speed, 
                "delay_cond": 2
            }

            trial_data = _test(model_path, model_file, options, env=env_dict[env])

            if epoch == "delay":
                start = trial_data["epoch_bounds"]["delay"][0]
                end = trial_data["epoch_bounds"]["delay"][1]
            elif epoch == "movement":
                start = trial_data["epoch_bounds"]["movement"][0] + 10
                end = trial_data["epoch_bounds"]["movement"][1]

            if system == "neural":
                speed_conds.append(trial_data["h"][:, start:end])
            elif system == "muscle":
                speed_conds.append(trial_data["muscle_acts"][:, start:end])
        
        interpolated_speed_conds = []
        for speed_cond in speed_conds:
            direction_conditions = []
            for direction in speed_cond:
                direction_conditions.append(interpolate_trial(direction, speed_conds[-1].shape[1]))
            interpolated_speed_conds.append(torch.stack(direction_conditions))

        # Should now be [directions, speeds, timesteps, neurons]
        env_speeds[env] = torch.stack(interpolated_speed_conds, dim=1)
    
    for env in env_dict:
        # Direction is a single direction containing 10 speeds
        for d, direction in enumerate(env_speeds[env]):

            speed_pca = PCA()
            speed_pca.fit(direction.reshape((-1, direction.shape[-1])))

            # Get kinematics and activity in a center out setting
            # On random and delay
            colors = plt.cm.plasma(np.linspace(0, 1, direction.shape[0])) 

            # Create a figure
            fig = plt.figure(figsize=(4, 4))
            # Add a 3D subplot
            ax = fig.add_subplot(111, projection="3d")

            all_proj = speed_pca.transform(direction.reshape((-1, direction.shape[-1])))
            min_z = np.min(all_proj[:, 2])

            # Going through each of the 10 speeds for a single direction
            for s, speed in enumerate(direction):

                # transform
                h_proj = speed_pca.transform(speed)

                # Plot the 3D line
                ax.plot(h_proj[:, 0], h_proj[:, 1], h_proj[:, 2], color=colors[s], linewidth=4, alpha=0.75, zorder=10)
                ax.plot(h_proj[:, 0], h_proj[:, 1], min_z, color="grey", linewidth=2, alpha=0.5)

                # Start and end positions
                ax.scatter(h_proj[0, 0], h_proj[0, 1], h_proj[0, 2], marker="^", color=colors[s], s=250, zorder=10)
                ax.scatter(h_proj[-1, 0], h_proj[-1, 1], h_proj[-1, 2], marker="X", color=colors[s], s=250, zorder=10)
        
            # No grid
            ax.grid(False)

            save_fig(os.path.join(exp_path, "3d", system, "speeds", epoch, f"{env}_direction_{d}_speed_pca"), eps=True)

            # --------------------------------- JPCA
            
            jpca = JPCA()
            formatted_data = [np.array(condition) for condition in direction]
            jpca.fit(formatted_data)

            (projected, 
            full_data_var,
            pca_var_capt,
            jpca_var_capt) = jpca.fit(formatted_data)

            plot_projections(projected, speeds=True)
            save_fig(os.path.join(exp_path, "jpcs", "speeds", epoch, f"{env}_direction_{d}_speed_jpca"), eps=True)

            # -------------------------------- Variance Explained

            plt.rc('figure', figsize=(3, 6))
            # Create a figure
            fig = plt.figure()
            # Add a 3D subplot
            ax = fig.add_subplot(111)

            if system == "neural":
                comps = 20
            elif system == "muscle":
                comps = 6

            for s, speed in enumerate(direction):
                # transform
                cond_proj = speed_pca.transform(speed)
                total_var = speed.var(dim=0).sum()
                ax.plot(
                    np.arange(1, comps+1),
                    [cond_proj[:, :i].var(axis=0).sum() / total_var for i in range(comps)], 
                    marker='o', color=colors[s], alpha=0.5, linewidth=4, markersize=20)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            save_fig(os.path.join(exp_path, "3d", system, "speeds", epoch, f"{env}_direction_{d}_var"), eps=True)




def plot_neural_pca_speeds_delay(model_name):
    _plot_pca_speeds(model_name, "delay", "neural")
def plot_neural_pca_speeds_movement(model_name):
    _plot_pca_speeds(model_name, "movement", "neural")

def plot_motor_pca_speeds_delay(model_name):
    _plot_pca_speeds(model_name, "delay", "muscle")
def plot_motor_pca_speeds_movement(model_name):
    _plot_pca_speeds(model_name, "movement", "muscle")




def _plot_jpcs(model_name, epoch):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pca"
    hp = load_hp(model_path)

    options = {
        "batch_size": 8, 
        "reach_conds": np.arange(0, 32, int(32 / 8)), 
        "speed_cond": 5, 
        "delay_cond": 2
    }

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env])

        mode = "h"
        size = hp["hid_size"]

        if epoch == "delay":
            env_h = trial_data[mode][:, trial_data["epoch_bounds"]["delay"][0]:trial_data["epoch_bounds"]["delay"][1]]
        elif epoch == "movement":
            env_h = trial_data[mode][:, trial_data["epoch_bounds"]["movement"][0]+10:trial_data["epoch_bounds"]["movement"][1]]
        else:
            raise ValueError("not valid epoch")

        jpca = JPCA()
        formatted_data = [np.array(condition) for condition in env_h]
        jpca.fit(formatted_data)

        (projected, 
        full_data_var,
        pca_var_capt,
        jpca_var_capt) = jpca.fit(formatted_data)

        plot_projections(projected)
        save_fig(os.path.join(exp_path, "jpcs", f"{env}_{epoch}"), eps=True)




def plot_jpcs_delay(model_name):
    _plot_jpcs(model_name, "delay")
def plot_jpcs_movement(model_name):
    _plot_jpcs(model_name, "movement")




def _plot_all_task_trajectories(model_name, system):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pca"
    create_dir(exp_path)

    options = {
        "batch_size": 8, 
        "reach_conds": np.arange(0, 32, int(32 / 8)), 
        "speed_cond": 5, 
        "delay_cond": 2
    }

    # Create a figure
    fig = plt.figure()
    # Add a 3D subplot
    ax = fig.add_subplot(111, projection="3d")

    task_hs = []
    for env in env_dict:
        trial_data = _test(model_path, model_file, options, env=env_dict[env])
        task_hs.append(trial_data[system][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]])
    all_task_hs = torch.cat(task_hs, dim=1)

    pca_3d = PCA(n_components=3)
    pca_3d.fit(all_task_hs.reshape((-1, all_task_hs.shape[-1])))

    colors = plt.cm.tab10(np.linspace(0, 1, 10)) 

    for e, hs in enumerate(task_hs):
        for c, condition in enumerate(hs):

            # transform
            h_proj = pca_3d.transform(condition)

            # Plot the 3D line
            ax.plot(h_proj[:, 0], h_proj[:, 1], h_proj[:, 2], linewidth=4, color=colors[e], alpha=0.5, zorder=10)
            ax.scatter(h_proj[0, 0], h_proj[0, 1], h_proj[0, 2], color=colors[e], marker="^", s=100, zorder=10)
            ax.scatter(h_proj[-1, 0], h_proj[-1, 1], h_proj[-1, 2], color=colors[e], marker="X", s=100, zorder=10)
            
    # No grid
    ax.grid(False)
    # Set background to white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    save_fig(os.path.join(exp_path, "3d", system, f"all_task_trajectories"), eps=True)




def plot_all_task_trajectories_neural(model_name):
    _plot_all_task_trajectories(model_name, "h")
def plot_all_task_trajectories_muscle(model_name):
    _plot_all_task_trajectories(model_name, "muscle_acts")




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

    elif args.experiment == "plot_neural_pca_speeds_delay":
        plot_neural_pca_speeds_delay(args.model_name) 
    elif args.experiment == "plot_neural_pca_speeds_movement":
        plot_neural_pca_speeds_movement(args.model_name) 
    elif args.experiment == "plot_motor_pca_speeds_delay":
        plot_motor_pca_speeds_delay(args.model_name) 
    elif args.experiment == "plot_motor_pca_speeds_movement":
        plot_motor_pca_speeds_movement(args.model_name) 

    elif args.experiment == "plot_jpcs_delay":
        plot_jpcs_delay(args.model_name) 
    elif args.experiment == "plot_jpcs_movement":
        plot_jpcs_movement(args.model_name) 
    elif args.experiment == "plot_pca_speeds":
        plot_pca_speeds(args.model_name) 
    elif args.experiment == "plot_movement_vs_delay_space_neural":
        plot_movement_vs_delay_space_neural(args.model_name) 
    elif args.experiment == "plot_movement_vs_delay_space_movement":
        plot_movement_vs_delay_space_movement(args.model_name) 
    
    elif args.experiment == "plot_all_task_trajectories_neural":
        plot_all_task_trajectories_neural(args.model_name)
    elif args.experiment == "plot_all_task_trajectories_muscle":
        plot_all_task_trajectories_muscle(args.model_name)

    else:
        raise ValueError("Experiment not in this file")