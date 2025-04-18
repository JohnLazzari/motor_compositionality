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
import warnings
warnings.filterwarnings("ignore")

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

def train_rnn512_softplus():
    model_path = "checkpoints/rnn512_softplus"
    model_file = "rnn512_softplus.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 512 UNITS")
    # leave hp as default
    train_2link(model_path, model_file)

def train_rnn256_softplus():
    hp = {"hid_size": 256}
    model_path = "checkpoints/rnn256_softplus"
    model_file = "rnn256_softplus.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 256 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn1024_softplus():
    hp = {"hid_size": 1024}
    model_path = "checkpoints/rnn1024_softplus"
    model_file = "rnn1024_softplus.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 1024 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn512_relu():
    hp = {"activation_name": "relu"}
    model_path = "checkpoints/rnn512_relu"
    model_file = "rnn512_relu.pth"
    print("TRAINING RNN WITH RELU AND 512 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn256_relu():
    hp = {"hid_size": 256, "activation_name": "relu"}
    model_path = "checkpoints/rnn256_relu"
    model_file = "rnn256_relu.pth"
    print("TRAINING RNN WITH RELU AND 256 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn1024_relu():
    hp = {"hid_size": 1024, "activation_name": "relu"}
    model_path = "checkpoints/rnn1024_relu"
    model_file = "rnn1024_relu.pth"
    print("TRAINING RNN WITH RELU AND 1024 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn512_tanh():
    hp = {"activation_name": "tanh"}
    model_path = "checkpoints/rnn512_tanh"
    model_file = "rnn512_tanh.pth"
    print("TRAINING RNN WITH TANH AND 512 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn256_tanh():
    hp = {"hid_size": 256, "activation_name": "tanh"}
    model_path = "checkpoints/rnn256_tanh"
    model_file = "rnn256_tanh.pth"
    print("TRAINING RNN WITH TANH AND 256 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn1024_tanh():
    hp = {"hid_size": 1024, "activation_name": "tanh"}
    model_path = "checkpoints/rnn1024_tanh"
    model_file = "rnn1024_tanh.pth"
    print("TRAINING RNN WITH TANH AND 1024 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_gru512():
    hp = {"network": "gru"}
    model_path = "checkpoints/gru512"
    model_file = "gru512.pth"
    print("TRAINING GRU WITH 512 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_gru256():
    hp = {"hid_size": 256, "network": "gru"}
    model_path = "checkpoints/gru256"
    model_file = "gru256.pth"
    print("TRAINING GRU WITH 256 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_gru1024():
    hp = {"hid_size": 1024, "network": "gru"}
    model_path = "checkpoints/gru1024"
    model_file = "gru1024.pth"
    print("TRAINING GRU WITH 1024 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)




def _test(model_path, model_file, options, env, stim=None, feedback_mask=None):
    """ Function will save all relevant data from a test run of a given env

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        joint_state (_type_, optional): _description_. Defaults to None.
        env (str, optional): _description_. Defaults to "RandomReach".
    """

    hp = load_hp(model_path)
    hp = hp.copy()
    hp["batch_size"] = options["batch_size"]
    
    device = "cpu"
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    env = env(effector=effector)

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

    # initialize batch
    x = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
    h = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
    
    obs, info = env.reset(testing=True, options=options)
    terminated = False
    trial_data = {}
    timesteps = 0

    trial_data["h"] = []
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
                x, h, action = policy(obs, x, h, stim, noise=False)
            else:
                x, h, action = policy(obs, x, h, noise=False)

            # Take step in motornet environment
            obs, reward, terminated, info = env.step(timesteps, action=action)

        timesteps += 1

        # Save all information regarding episode step
        trial_data["h"].append(h.unsqueeze(1))  # trajectories
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



def plot_psth(model_name):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/psth"

    create_dir(exp_path)

    for env in env_dict:
        for speed in range(10):

            options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": speed, "delay_cond": 1}

            trial_data = _test(model_path, model_file, options, env=env_dict[env])
        
            # Get kinematics and activity in a center out setting
            # On random and delay
            colors = plt.cm.inferno(np.linspace(0, 1, trial_data["h"].shape[0])) 

            delay = trial_data["epoch_bounds"]["delay"][0]
            mov = trial_data["epoch_bounds"]["movement"][0]
            hold = trial_data["epoch_bounds"]["hold"][0]

            for i, h in enumerate(trial_data["h"]):
                plt.plot(torch.mean(h, dim=-1)[delay:], color=colors[i], linewidth=4)
                plt.axvline(mov-delay, linestyle="dashed", color="grey")
                plt.axvline(hold-delay, linestyle="dashed", color="grey")
            save_fig(os.path.join(exp_path, f"{env}_speed{speed}_tg_trajectory.png"))




def plot_pca(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pca"

    create_dir(exp_path)
    options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": 5, "delay_cond": 0}

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env])

        delay_start = trial_data["epoch_bounds"]["delay"][0]
        movement_start = trial_data["epoch_bounds"]["movement"][0]
    
        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, trial_data["h"].shape[0])) 

        pca_3d = PCA(n_components=3)
        pca_3d.fit(trial_data["h"].reshape((-1, trial_data["h"].shape[-1])))

        # Create a figure
        fig3d = plt.figure()
        # Add a 3D subplot
        ax3d = fig3d.add_subplot(111, projection='3d')
        for i, h in enumerate(trial_data["h"]):

            # transform
            h_proj = pca_3d.transform(h)
            # Plot the 3D line
            ax3d.plot(h_proj[delay_start:movement_start, 0], h_proj[delay_start:movement_start, 1], h_proj[delay_start:movement_start, 2], color=colors[i], linewidth=4, linestyle="dashed")
            ax3d.plot(h_proj[movement_start:, 0], h_proj[movement_start:, 1], h_proj[movement_start:, 2], color=colors[i], linewidth=4)
            # Set labels for axes
            ax3d.set_xlabel('PC 1')
            ax3d.set_ylabel('PC 2')
            ax3d.set_zlabel('PC 3')
            ax3d.set_title(f'{env} PCs')

            ax3d.scatter(h_proj[delay_start, 0], h_proj[delay_start, 1], h_proj[delay_start, 2], marker="^", color=colors[i], s=250, zorder=10)
            ax3d.scatter(h_proj[-1, 0], h_proj[-1, 1], h_proj[-1, 2], marker="X", color=colors[i], s=250, zorder=10)

        save_fig(os.path.join(exp_path + "/3d", f"{env}_tg_trajectory.png"))

        # Create a figure
        fig2d = plt.figure()
        # Add a 3D subplot
        ax2d = fig2d.add_subplot(111)
        for i, h in enumerate(trial_data["h"]):

            # transform
            h_proj = pca_3d.transform(h)
            # Plot the 3D line
            ax2d.plot(h_proj[delay_start:movement_start, 0], h_proj[delay_start:movement_start, 1], color=colors[i], linewidth=4, linestyle="dashed")
            ax2d.plot(h_proj[movement_start:, 0], h_proj[movement_start:, 1], color=colors[i], linewidth=4)
            # Set labels for axes
            ax2d.set_xlabel('PC 1')
            ax2d.set_ylabel('PC 2')
            ax2d.set_title(f'{env} PCs')

            ax2d.scatter(h_proj[delay_start, 0], h_proj[delay_start, 1], marker="^", color=colors[i], s=250, zorder=10)
            ax2d.scatter(h_proj[-1, 0], h_proj[-1, 1], marker="X", color=colors[i], s=250, zorder=10)

        save_fig(os.path.join(exp_path + "/2d", f"{env}_tg_trajectory.png"))





def plot_task_trajectories(model_name):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/trajectories"

    create_dir(exp_path)

    for env in env_dict:
        for speed in range(10):

            options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": speed}

            effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
            cur_env = env_dict[env](effector=effector)
            
            obs, info = cur_env.reset(testing=True, options=options)

            # Get kinematics and activity in a center out setting
            # On random and delay
            colors = plt.cm.inferno(np.linspace(0, 1, cur_env.traj.shape[1])) 

            for i, tg in enumerate(cur_env.traj):
                plt.scatter(tg[:, 0], tg[:, 1], s=10, color=colors)
                plt.scatter(tg[0, 0], tg[0, 1], s=150, marker='x', color="black")
                plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker='^', color="black")
            save_fig(os.path.join(exp_path, f"{env}_speed{speed}_tg_trajectory.png"))




def plot_task_input_output(model_name):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/input"

    create_dir(exp_path)

    for env in env_dict:

        options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": 5, "delay_cond": 0}

        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        cur_env = env_dict[env](effector=effector)
        
        obs, info = cur_env.reset(testing=True, options=options)

        for batch in range(options["batch_size"]):

            fig, ax = plt.subplots(5, 1)
            fig.set_size_inches(3, 6)
            plt.rc('font', size=6)

            ax[0].imshow(cur_env.rule_input[batch].unsqueeze(0).repeat(cur_env.max_ep_duration, 1).T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            # Remove top and right only (common for minimalist style)
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['bottom'].set_visible(False)
            ax[0].set_xticks([])
            ax[0].set_title("Rule Input")
            ax[0].axvline(cur_env.epoch_bounds["delay"][0], color="grey", linestyle="dashed")
            ax[0].axvline(cur_env.epoch_bounds["movement"][0], color="grey", linestyle="dashed")
            ax[0].axvline(cur_env.epoch_bounds["hold"][0], color="grey", linestyle="dashed")

            ax[1].plot(cur_env.speed_scalar[batch].unsqueeze(0).repeat(cur_env.max_ep_duration, 1), color="blue")
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[1].set_xticks([])
            ax[1].set_title("Speed Scalar")
            ax[1].axvline(cur_env.epoch_bounds["delay"][0], color="grey", linestyle="dashed")
            ax[1].axvline(cur_env.epoch_bounds["movement"][0], color="grey", linestyle="dashed")
            ax[1].axvline(cur_env.epoch_bounds["hold"][0], color="grey", linestyle="dashed")

            ax[2].plot(cur_env.go_cue[batch], color="blue")
            ax[2].spines['top'].set_visible(False)
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['bottom'].set_visible(False)
            ax[2].set_xticks([])
            ax[2].set_title("Go Cue")
            ax[2].axvline(cur_env.epoch_bounds["delay"][0], color="grey", linestyle="dashed")
            ax[2].axvline(cur_env.epoch_bounds["movement"][0], color="grey", linestyle="dashed")
            ax[2].axvline(cur_env.epoch_bounds["hold"][0], color="grey", linestyle="dashed")

            ax[3].imshow(cur_env.vis_inp[batch].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[3].spines['top'].set_visible(False)
            ax[3].spines['right'].set_visible(False)
            ax[3].spines['bottom'].set_visible(False)
            ax[3].set_xticks([])
            ax[3].set_title("Visual Input")
            ax[3].axvline(cur_env.epoch_bounds["delay"][0], color="grey", linestyle="dashed")
            ax[3].axvline(cur_env.epoch_bounds["movement"][0], color="grey", linestyle="dashed")
            ax[3].axvline(cur_env.epoch_bounds["hold"][0], color="grey", linestyle="dashed")

            ax[4].imshow(cur_env.traj[batch].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[4].spines['top'].set_visible(False)
            ax[4].spines['right'].set_visible(False)
            ax[4].spines['bottom'].set_visible(False)
            ax[4].set_xticks([])
            ax[4].set_title("Tg Output (Only Movement Epoch)")

            save_fig(os.path.join(exp_path, f"{env}_input_orientation{batch}"))




def plot_task_feedback(model_name):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/input"

    create_dir(exp_path)

    options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4)}

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env])
    
        for i, inp in enumerate(trial_data["obs"]):

            fig, ax = plt.subplots(7, 1)
            fig.set_size_inches(3, 6)
            plt.rc('font', size=6)

            ax[0].imshow(inp[:, :10].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            # Remove top and right only (common for minimalist style)
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['bottom'].set_visible(False)
            ax[0].set_xticks([])
            ax[0].set_title("Rule Input")

            ax[1].plot(inp[:, 10:11], color="blue")
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[1].set_xticks([])
            ax[1].set_title("Speed Scalar")

            ax[2].plot(inp[:, 11:12], color="blue")
            ax[2].spines['top'].set_visible(False)
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['bottom'].set_visible(False)
            ax[2].set_xticks([])
            ax[2].set_title("Go Cue")

            ax[3].imshow(inp[:, 12:14].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[3].spines['top'].set_visible(False)
            ax[3].spines['right'].set_visible(False)
            ax[3].spines['bottom'].set_visible(False)
            ax[3].set_xticks([])
            ax[3].set_title("Target Position")

            ax[4].imshow(inp[:, 14:16].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[4].spines['top'].set_visible(False)
            ax[4].spines['right'].set_visible(False)
            ax[4].spines['bottom'].set_visible(False)
            ax[4].set_xticks([])
            ax[4].set_title("Fingertip")

            ax[5].imshow(inp[:, 16:22].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[5].spines['top'].set_visible(False)
            ax[5].spines['right'].set_visible(False)
            ax[5].spines['bottom'].set_visible(False)
            ax[5].set_xticks([])
            ax[5].set_title("Muscle Length")
            
            ax[6].imshow(inp[:, 22:28].T, vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[6].spines['top'].set_visible(False)
            ax[6].spines['right'].set_visible(False)
            ax[6].spines['bottom'].set_visible(False)
            ax[6].set_title("Muscle Velocity")

            save_fig(os.path.join(exp_path, f"{env}_input_orientation{i}"))




def plot_task_kinematics(model_name):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/kinematics"

    create_dir(exp_path)

    for env in env_dict:
        for speed in range(10):

            options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": speed, "delay_cond": 1}

            trial_data = _test(model_path, model_file, options, env=env_dict[env])
        
            # Get kinematics and activity in a center out setting
            # On random and delay
            colors_time = plt.cm.inferno(np.linspace(0, 1, trial_data["xy"].shape[1])) 
            colors_xy = plt.cm.inferno(np.linspace(0, 1, trial_data["xy"].shape[0])) 

            for i, (tg, xy) in enumerate(zip(trial_data["tg"], trial_data["xy"])):
                plt.scatter(xy[:, 0], xy[:, 1], s=10, color=colors_time)
                plt.scatter(xy[0, 0], xy[0, 1], s=150, marker='x', color="black")
                plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker='^', color="black")
            save_fig(os.path.join(exp_path, "scatter", f"{env}_speed{speed}_kinematics.png"))

            # Plot x coordinate only 
            for i, xy in enumerate(trial_data["xy"]):
                plt.plot(xy[:, 0], color=colors_xy[i])
            save_fig(os.path.join(exp_path, "xpos", f"{env}_speed{speed}_xpos.png"))

            # Plot y coordinate only 
            for i, xy in enumerate(trial_data["xy"]):
                plt.plot(xy[:, 1], color=colors_xy[i])
            save_fig(os.path.join(exp_path, "ypos", f"{env}_speed{speed}_ypos.png"))




# TODO change for new setting with stable and hold epochs
def variance_by_rule(model_name):
    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    hp = load_hp(model_path)

    env_var_dict = {}
    var_dir_tensor = torch.empty(size=(hp["hid_size"], len(env_dict)))
    task_list = []

    # Need to know the current largest timestep possible given speeds (ignoring delay)
    max_timesteps = 300

    for i, env in enumerate(env_dict):
        dir_var_list = []
        for speed in range(10):

            options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "speed_cond": speed, "delay_cond": 0}
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            delay_time = trial_data["delay_time"]

            # Should be of shape batch, time, neurons
            h = trial_data["h"]

            # Focus on movement only
            interpolated_trials = torch.stack([interpolate_trial(direction[delay_time:], max_timesteps) for direction in h])
            dir_var_list.append(interpolated_trials)

        var_dir = torch.cat(dir_var_list).var(dim=0).mean(dim=0)
        var_dir_tensor[:, i] = var_dir
        task_list.append(env)
    
    total_var = var_dir_tensor
    env_var_dict["h_var_all"] = total_var.numpy()
    env_var_dict["keys"] = task_list
    
    save_name = 'variance_rule'
    fname = os.path.join(model_path, save_name + '.pkl')
    print('Variance saved at {:s}'.format(fname))
    with open(fname, 'wb') as f:
        pickle.dump(env_var_dict, f)




# TODO change for new setting with stable and hold epochs
def variance_by_epoch(model_name):
    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"

    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1)}

    env_var_dict = {}
    var_list = []
    task_list = []

    for env in env_dict:
        trial_data = _test(model_path, model_file, options, env=env_dict[env])
        # Should be of shape batch, time, neurons
        h = trial_data["h"]

        h_delay = h[:, :trial_data["delay_time"], :]
        h_movement = h[:, trial_data["delay_time"]:, :]

        task_var_delay = h_delay.var(dim=0).mean(dim=0)
        task_var_mov = h_movement.mean(dim=0).mean(dim=0)

        var_list.extend([task_var_delay, task_var_mov])
        task_list.extend([f"{env}_delay", f"{env}_movement"])

    env_var_dict["h_var_all"] = torch.stack(var_list, dim=1).numpy()
    env_var_dict["keys"] = task_list
    
    save_name = 'variance_epoch'
    fname = os.path.join(model_path, save_name + '.pkl')
    print('Variance saved at {:s}'.format(fname))
    with open(fname, 'wb') as f:
        pickle.dump(env_var_dict, f)




def plot_variance_by_rule(model_name):
    # Get selectivity and clusters for different movements
    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/variance"
    
    clustering = Analysis(model_path, "rule")
    clustering.plot_variance(os.path.join(exp_path, "variance_rule.png"))
    clustering.plot_cluster_score(os.path.join(exp_path, "cluster_score_rule.png"))
    clustering.plot_2Dvisualization(os.path.join(exp_path, "clusters_rule.png"))




def plot_variance_by_epoch(model_name):
    # Get selectivity and clusters for different movements
    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/variance/variance_epoch.png"
    
    clustering = Analysis(model_path, "epoch")
    clustering.plot_variance(os.path.join(exp_path, "variance_epoch.png"))
    clustering.plot_cluster_score(os.path.join(exp_path, "cluster_score_epoch.png"))
    clustering.plot_2Dvisualization(os.path.join(exp_path, "clusters_epoch.png"))




# TODO change for new setting with stable and hold epochs
def compute_fps(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"

    NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
    N_INITS = 1024 # The number of initial states to provide

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 2, "reach_conds": torch.arange(0, 32, 16), "delay_cond": 1, "speed_cond": 5}

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
            'max_iters': 100,
            'lr_init': 1.,
            'outlier_distance_scale': 10.0,
            'verbose': False, 
            'super_verbose': False,
            'tol_unique': 2,
            'do_compute_jacobians': False}
        
        env_fps_list = []
        for b, condition in enumerate(trial_data["h"]):
            for t, timepoint in enumerate(condition):
                if t % 150 == 0:

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




# TODO change for new setting with stable and hold epochs
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




def _principal_angles(model_name, system, comparison):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pc_angles"
    hp = load_hp(model_path)

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

    if system == "neural":
        mode = "h"
    elif sysetm == "muscle":
        mode = "muscle_acts"
    else:
        raise ValueError("Not a valid system")

    if comparison == "task":

        trial_data_envs = {}
        for env in env_dict:
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            trial_data_envs[env] = trial_data[mode][:, trial_data["epoch_bounds"]["delay"][0]:]
        # Get all unique pairs of unit activity across tasks
        combinations = list(itertools.combinations(trial_data_envs, 2))

    elif comparison == "epoch":

        combinations = []
        for env in env_dict:
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            combinations.append((
                trial_data["h"][:, :trial_data["epoch_bounds"]["delay"][0]], 
                trial_data["h"][:, trial_data["epoch_bounds"]["delay"][0]:]
            ))

    angles_dict = principal_angles(combinations)
    
    for angles in angles_dict:
        plt.plot(angles_dict[angles], label=angles)
    save_fig(os.path.join(exp_path, f"principal_angles.png"))




def _vaf_ratio(model_name, system, comparison):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pc_angles"
    hp = load_hp(model_path)

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

    if system == "neural":
        mode = "h"
    elif sysetm == "muscle":
        mode = "muscle_acts"
    else:
        raise ValueError("Not a valid system")

    if comparison == "task":

        trial_data_envs = {}
        for env in env_dict:
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            trial_data_envs[env] = trial_data[mode][:, trial_data["epoch_bounds"]["delay"][0]:]
        # Get all unique pairs of unit activity across tasks
        combinations = list(itertools.combinations(trial_data_envs, 2))

    elif comparison == "epoch":

        combinations = []
        for env in env_dict:
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            combinations.append((
                trial_data["h"][:, :trial_data["epoch_bounds"]["delay"][0]], 
                trial_data["h"][:, trial_data["epoch_bounds"]["delay"][0]:]
            ))

    vaf_ratio_list, vaf_ratio_list_control = vaf_ratio(combinations)
    
    bins = np.linspace(0, 1, 50)
    weights = np.ones_like(vaf_ratio_list) * 100 / len(vaf_ratio_list)
    plt.hist(vaf_ratio_list, bins=bins, weights=weights, color="purple")
    plt.hist(vaf_ratio_list_control, bins=bins, weights=weights, color="grey")
    plt.xlim(0, 1)
    save_fig(os.path.join(exp_path, "neural_vaf_ratio.png"))




def neural_principal_angles_task(model_name):
    _principal_angles(model_name, "neural", "task")
def neural_principal_angles_epoch(model_name):
    _principal_angles(model_name, "neural", "epoch")
def muscle_principal_angles_epoch(model_name):
    _principal_angles(model_name, "muscle", "task")
def muscle_principal_angles_epoch(model_name):
    _principal_angles(model_name, "muscle", "epoch")




def neural_vaf_ratio_task(model_name):
    _vaf_ratio(model_name, "neural", "task")
def neural_vaf_ratio_epoch(model_name):
    _vaf_ratio(model_name, "neural", "epoch")
def muscle_vaf_ratio_epoch(model_name):
    _vaf_ratio(model_name, "muscle", "task")
def muscle_vaf_ratio_epoch(model_name):
    _vaf_ratio(model_name, "muscle", "epoch")




def plot_dpca(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/dpca"
    hp = load_hp(model_path)

    max_timesteps = 300

    env_trials = torch.empty(size=(hp["hid_size"], 10, 10, 32, max_timesteps))
    for i, env in enumerate(env_dict):
        speed_trials = torch.empty(size=(hp["hid_size"], 10, 32, max_timesteps))
        for speed in range(10):
            options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": speed}
            trial_data = _test(model_path, model_file, options, env=env_dict[env])

            interpolated_trials = torch.stack([interpolate_trial(h[trial_data["epoch_bounds"]["movement"][0]:], max_timesteps) for h in trial_data["h"]])
            speed_trials[:, speed, ...] = interpolated_trials.permute(2, 0, 1)
        env_trials[:, i, ...] = speed_trials
    
    # mean center
    dpca = dPCA.dPCA(labels='esdt')
    Z = dpca.fit_transform(env_trials.numpy())    

    fig, ax = plt.subplots(1, 4)

    print(Z['t'].shape)
    print(Z['e'].shape)
    print(Z['s'].shape)
    print(Z['d'].shape)

    for s in range(S):
        plot(time,Z['t'][0,s])

    title('1st time component')
        
    for s in range(S):
        plot(time,Z['s'][0,s])
        
    title('1st stimulus component')
        
    for s in range(S):
        plot(time,Z['st'][0,s])
        
    title('1st mixing component')
    save_fig(os.path.join(exp_path, "dpcas.png"))




# TODO change for new setting with stable and hold epochs
def module_silencing(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/silencing/module_silencing.png"

    hp = load_hp(model_path)
    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

    clustering = Analysis(model_path, "rule")

    n_cluster = clustering.n_cluster
    change_loss_envs = torch.empty(size=(len(env_dict), n_cluster))
    for i, env in enumerate(env_dict):
        for cluster in range(n_cluster):

            print(f"Silencing cluster {cluster} in environment {env}")
            cur_cluster_indices = clustering.ind_active[clustering.labels == cluster]
            silencing_mask = torch.zeros(size=(options["batch_size"], 1, hp["hid_size"]))
            silencing_mask[..., cur_cluster_indices] = -10

            # Control trial
            trial_data_control = _test(model_path, model_file, options, env=env_dict[env])
            control_loss = l1_dist(trial_data_control["xy"], trial_data_control["tg"])

            # Stim trial
            trial_data_stim = _test(model_path, model_file, options, env=env_dict[env], stim=silencing_mask)
            stim_loss = l1_dist(trial_data_stim["xy"], trial_data_stim["tg"])
            
            loss_change = control_loss.item() - stim_loss.item()
            print(f"Change in loss: {loss_change}\n")
            change_loss_envs[i, cluster] = loss_change

    img = plt.imshow(change_loss_envs, vmin=-0.5, vmax=0.5, cmap="seismic")
    plt.xticks(ticks=np.arange(0, n_cluster), labels=[f"cluster_{i}" for i in range(n_cluster)], rotation=45)
    plt.yticks(ticks=np.arange(0, 10), labels=[env for env in env_dict], rotation=45)
    cbar = plt.colorbar(img)
    cbar.set_label('Change in Loss')
    save_fig(exp_path)




# TODO change for new setting with stable and hold epochs
def feedback_ablation(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/silencing/feedback_silencing.png"

    hp = load_hp(model_path)
    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 0, "speed_cond": 5}

    feedback_masks = {}
    # Build masks for ablating input
    rule_mask = torch.ones(size=(options["batch_size"], 28))
    rule_mask[:10] = 0
    feedback_masks["rule"] = rule_mask

    speed_mask = torch.ones(size=(options["batch_size"], 28))
    speed_mask[10:11] = 0
    feedback_masks["speed_scalar"] = speed_mask

    go_mask = torch.ones(size=(options["batch_size"], 28))
    go_mask[11:12] = 0
    feedback_masks["speed_scalar"] = go_mask

    tg_mask = torch.ones(size=(options["batch_size"], 28))
    tg_mask[12:14] = 0
    feedback_masks["vis_inp"] = tg_mask

    fg_mask = torch.ones(size=(options["batch_size"], 28))
    fg_mask[14:16] = 0
    feedback_masks["vis_feedback"] = fg_mask

    length_mask = torch.ones(size=(options["batch_size"], 28))
    length_mask[16:22] = 0
    feedback_masks["muscle_length"] = length_mask

    vel_mask = torch.ones(size=(options["batch_size"], 28))
    vel_mask[22:28] = 0
    feedback_masks["muscle_vel"] = vel_mask

    change_loss_envs = torch.empty(size=(len(env_dict), len(feedback_masks)))
    for i, env in enumerate(env_dict):
        for j, mask in enumerate(feedback_masks):

            print(f"Ablating {mask} in env {env}")

            # Control trial
            trial_data_control = _test(model_path, model_file, options, env=env_dict[env])
            control_loss = l1_dist(trial_data_control["xy"], trial_data_control["tg"])

            # Stim trial
            trial_data_stim = _test(model_path, model_file, options, env=env_dict[env], feedback_mask=feedback_masks[mask])
            stim_loss = l1_dist(trial_data_stim["xy"], trial_data_stim["tg"])
            
            loss_change = control_loss.item() - stim_loss.item()
            print(f"Change in loss: {loss_change}\n")
            change_loss_envs[i, j] = loss_change

    img = plt.imshow(change_loss_envs, vmin=-0.5, vmax=0.5, cmap="seismic")
    plt.xticks(ticks=np.arange(0, 6), labels=[mask for mask in feedback_masks], rotation=45)
    plt.yticks(ticks=np.arange(0, 10), labels=[env for env in env_dict], rotation=45)
    cbar = plt.colorbar(img)
    cbar.set_label('Change in Loss')
    save_fig(exp_path)




if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "train_rnn256_softplus":
        train_rnn256_softplus() 
    elif args.experiment == "train_rnn512_softplus":
        train_rnn512_softplus() 
    elif args.experiment == "train_rnn1024_softplus":
        train_rnn1024_softplus() 
    elif args.experiment == "train_rnn256_relu":
        train_rnn256_relu() 
    elif args.experiment == "train_rnn512_relu":
        train_rnn512_relu() 
    elif args.experiment == "train_rnn1024_relu":
        train_rnn1024_relu() 
    elif args.experiment == "train_rnn256_tanh":
        train_rnn256_tanh() 
    elif args.experiment == "train_rnn512_tanh":
        train_rnn512_tanh() 
    elif args.experiment == "train_rnn1024_tanh":
        train_rnn1024_tanh() 
    elif args.experiment == "train_gru256":
        train_gru256() 
    elif args.experiment == "train_gru512":
        train_gru512() 
    elif args.experiment == "train_gru1024":
        train_gru1024() 
    elif args.experiment == "plot_pca":
        plot_pca(args.model_name) 
    elif args.experiment == "plot_psth":
        plot_psth(args.model_name) 
    elif args.experiment == "plot_task_trajectories":
        plot_task_trajectories(args.model_name) 
    elif args.experiment == "plot_task_input_output":
        plot_task_input_output(args.model_name) 
    elif args.experiment == "plot_task_feedback":
        plot_task_feedback(args.model_name) 
    elif args.experiment == "plot_task_kinematics":
        plot_task_kinematics(args.model_name) 
    elif args.experiment == "variance_by_rule":
        variance_by_rule(args.model_name) 
    elif args.experiment == "variance_by_epoch":
        variance_by_epoch(args.model_name) 
    elif args.experiment == "plot_variance_by_rule":
        plot_variance_by_rule(args.model_name) 
    elif args.experiment == "plot_variance_by_epoch":
        plot_variance_by_epoch(args.model_name) 
    elif args.experiment == "compute_fps":
        compute_fps(args.model_name) 
    elif args.experiment == "plot_fps":
        plot_fps(args.model_name) 
    elif args.experiment == "neural_principle_angles":
        neural_principle_angles(args.model_name) 
    elif args.experiment == "muscle_principle_angles":
        muscle_principle_angles(args.model_name) 
    elif args.experiment == "module_silencing":
        module_silencing(args.model_name) 
    elif args.experiment == "feedback_ablation":
        feedback_ablation(args.model_name) 
    elif args.experiment == "plot_flow_fields":
        plot_flow_fields(args.model_name) 
    elif args.experiment == "plot_dpca":
        plot_dpca(args.model_name) 