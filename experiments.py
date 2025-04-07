from train import train_2link
import motornet as mn
from model import Policy
import torch
import os
from utils import load_hp, create_dir, save_fig
from envs import RandomReach, DlyRandomReach, Maze, CenterOutReach
import matplotlib.pyplot as plt
import numpy as np
import argparse

def train_2link_multi(config_path, model_path, model_file):
    # leave hp as default
    train_2link(config_path, model_path, model_file)

def _test(config_path, model_path, model_file, batch_size=256, joint_state=None, env="RandomReach"):
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
    hp["batch_size"] = batch_size
    
    device = "cpu"
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

    if env == "RandomReach":
        env = RandomReach(effector=effector)
    elif env == "DlyRandomReach":
        env = DlyRandomReach(effector=effector)
    elif env == "Maze":
        env = Maze(effector=effector)
    elif env == "CenterOutReach":
        env = CenterOutReach(effector=effector)

    # Loading in model
    policy = Policy(
        config_path, 
        effector.n_muscles, 
        activation_name=hp["activation_name"],
        noise_level_act=hp["noise_level_act"], 
        noise_level_inp=hp["noise_level_inp"], 
        constrained=hp["constrained"], 
        dt=hp["dt"],
        t_const=hp["t_const"],
        device=device
        )

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=torch.device('cpu'))
    policy.load_state_dict(checkpoint['agent_state_dict'])

    # initialize batch
    h = torch.zeros(size=(hp["batch_size"], policy.mrnn.total_num_units))
    
    if joint_state == "center":
        joint_state = torch.tensor([effector.pos_range_bound[0] * 0.5 + effector.pos_upper_bound[0] + 0.1, 
                                    effector.pos_range_bound[1] * 0.5 + effector.pos_upper_bound[1] + 0.5, 0, 0
        ]).unsqueeze(0).repeat(hp["batch_size"], 1)

    options = {"batch_size": hp["batch_size"], "joint_state": joint_state}

    obs, info = env.reset(options=options)
    terminated = False
    trial_data = {}
    timesteps = 0

    trial_data["h"] = []
    trial_data["action"] = []
    trial_data["obs"] = []
    trial_data["xy"] = []
    trial_data["tg"] = []

    # simulate whole episode
    while not terminated:  # will run until `max_ep_duration` is reached
        with torch.no_grad():
            x, h, action = policy(h, obs, noise=False)
            obs, reward, terminated, info = env.step(timesteps, action=action)

            trial_data["h"].append(h.unsqueeze(1))  # trajectories
            trial_data["action"].append(action.unsqueeze(1))  # targets
            trial_data["obs"].append(obs.unsqueeze(1))  # targets
            trial_data["xy"].append(info["states"]["fingertip"][:, None, :])  # trajectories
            trial_data["tg"].append(info["goal"][:, None, :])  # targets

            timesteps += 1

    for key in trial_data:
        trial_data[key] = torch.cat(trial_data[key], dim=1)

    return trial_data

def center_out(config_path, model_path, model_file, exp_path):

    create_dir(exp_path)

    trial_data = _test(config_path, model_path, model_file, env="CenterOutReach")
    
    # Get kinematics and activity in a center out setting
    # On random and delay
    colors = plt.cm.inferno(np.linspace(0, 1, trial_data["xy"].shape[0])) 

    for i, (tg, xy) in enumerate(zip(trial_data["tg"], trial_data["xy"])):
        color = colors[i]
        plt.plot(xy[:, 0], xy[:, 1], color=color)
        plt.scatter(xy[0, 0], xy[0, 1], s=150, marker='x', color=color)
        plt.scatter(tg[:, 0], tg[:, 1], s=150, marker='^', color=color)
    save_fig(os.path.join(exp_path, "kinematics.png"))

    for i, act in enumerate(trial_data["h"]):
        color = colors[i]
        plt.plot(torch.mean(act, dim=-1), color=color)
    save_fig(os.path.join(exp_path, "psth.png"))

def center_out_random(config_path, model_path, model_file, exp_path):

    create_dir(exp_path)

    trial_data = _test(config_path, model_path, model_file, joint_state="center")
    
    # Get kinematics and activity in a center out setting
    # On random and delay
    colors = plt.cm.inferno(np.linspace(0, 1, trial_data["xy"].shape[0])) 

    for i, (tg, xy) in enumerate(zip(trial_data["tg"], trial_data["xy"])):
        color = colors[i]
        plt.plot(xy[:, 0], xy[:, 1], color=color)
        plt.scatter(xy[0, 0], xy[0, 1], s=150, marker='x', color=color)
        plt.scatter(tg[:, 0], tg[:, 1], s=150, marker='^', color=color)
    save_fig(os.path.join(exp_path, "kinematics.png"))

    for i, act in enumerate(trial_data["h"]):
        color = colors[i]
        plt.plot(torch.mean(act, dim=-1), color=color)
    save_fig(os.path.join(exp_path, "psth.png"))

def center_out_dlyrandom(config_path, model_path, model_file, exp_path):

    create_dir(exp_path)

    trial_data = _test(config_path, model_path, model_file, joint_state="center", env="DlyRandomReach")

    # Get kinematics and activity in a center out setting
    # On random and delay
    colors = plt.cm.inferno(np.linspace(0, 1, trial_data["xy"].shape[0])) 

    for i, (tg, xy) in enumerate(zip(trial_data["tg"], trial_data["xy"])):
        color = colors[i]
        plt.plot(xy[:, 0], xy[:, 1], color=color)
        plt.scatter(xy[0, 0], xy[0, 1], s=150, marker='x', color=color)
        plt.scatter(tg[:, 0], tg[:, 1], s=150, marker='^', color=color)
    save_fig(os.path.join(exp_path, "kinematics.png"))

    for i, act in enumerate(trial_data["h"]):
        color = colors[i]
        plt.plot(torch.mean(act, dim=-1), color=color)
    plt.axvline(x=100, color='grey', linestyle='--')
    
    save_fig(os.path.join(exp_path, "psth.png"))

def center_out_maze(config_path, model_path, model_file, exp_path):
    # Save multiple examples of kinematics and activity on maze
    create_dir(exp_path)

    trial_data = _test(config_path, model_path, model_file, batch_size=1, joint_state="center", env="Maze")

    # Get kinematics and activity in a center out setting
    # On random and delay
    colors = plt.cm.inferno(np.linspace(0, 1, trial_data["xy"].shape[0])) 

    for i, (tg, xy) in enumerate(zip(trial_data["tg"], trial_data["xy"])):
        color = colors[i]
        plt.plot(xy[:, 0], xy[:, 1], color=color)
        plt.scatter(xy[0, 0], xy[0, 1], s=150, marker='x', color=color)
        plt.scatter(tg[:, 0], tg[:, 1], s=150, marker='^', color=color)
    save_fig(os.path.join(exp_path, "kinematics.png"))

    for i, act in enumerate(trial_data["h"]):
        color = colors[i]
        plt.plot(torch.mean(act, dim=-1), color=color)
    plt.axvline(x=100, color='grey', linestyle='--')
    
    save_fig(os.path.join(exp_path, "psth.png"))


def selectivity_and_clustering():
    # Get selectivity and clusters for different movements
    pass

if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="A simple argparse example")

    # Add arguments to the parser
    parser.add_argument("--config_path", type=str, default="configurations/mrnn.json")
    parser.add_argument("--model_path", type=str, default="checkpoints/mrnn")
    parser.add_argument("--model_file", type=str, default="mrnn.pth")
    parser.add_argument("--exp_path", type=str, default="results/mrnn/center_out_random")
    parser.add_argument("--experiment", type=str, default="train_2link_multi")

    # Parse the command-line arguments
    args = parser.parse_args()
    
    if args.experiment == "train_2link_multi":
        train_2link_multi(args.config_path, args.model_path, args.model_file) 
    elif args.experiment == "center_out_random":
        center_out_random(args.config_path, args.model_path, args.model_file, args.exp_path) 
    elif args.experiment == "center_out_dlyrandom":
        center_out_dlyrandom(args.config_path, args.model_path, args.model_file, args.exp_path) 
    elif args.experiment == "center_out_maze":
        center_out_maze(args.config_path, args.model_path, args.model_file, args.exp_path) 
    elif args.experiment == "center_out":
        center_out(args.config_path, args.model_path, args.model_file, args.exp_path) 