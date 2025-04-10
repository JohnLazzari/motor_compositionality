from train import train_2link
import motornet as mn
from model import RNNPolicy
import torch
import os
from utils import load_hp, create_dir, save_fig
from envs import DlyHalfReach, DlyHalfCircleClk, DlyHalfCircleCClk, DlySinusoid, DlySinusoidInv
from envs import DlyFullReach, DlyFullCircleClk, DlyFullCircleCClk, DlyFigure8, DlyFigure8Inv
import matplotlib.pyplot as plt
import numpy as np
import config

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
    # leave hp as default
    train_2link(config_path, model_path, model_file)

def _test(config_path, model_path, model_file, options, env):
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
    policy = RNNPolicy(
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

def plot_task_trajectories(config_path, model_path, model_file, exp_path):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """

    create_dir(exp_path)

    options = {"batch_size": 8, "reach_conds": torch.arange(0, 8, 1)}

    for env in env_dict:

        trial_data = _test(config_path, model_path, model_file, options, env=env_dict[env])
    
        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, trial_data["tg"].shape[1])) 

        for i, tg in enumerate(trial_data["tg"]):
            plt.scatter(tg[:, 0], tg[:, 1], s=10, color=colors)
            plt.scatter(tg[0, 0], tg[0, 1], s=150, marker='x', color="black")
            plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker='^', color="black")
        save_fig(os.path.join(exp_path, f"{env}_tg_trajectory.png"))

        for i, inp in enumerate(trial_data["obs"]):
            fig, ax = plt.subplots(5, 1)
            ax[0].imshow(inp[:, :10].T, cmap="Blues", aspect="auto")
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].plot(inp[:, 10:11], color="blue")
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[2].imshow(inp[:, 11:19].T, cmap="Blues", aspect="auto")
            ax[2].set_xticks([])
            ax[2].set_yticks([])
            ax[3].imshow(inp[:, 19:21].T, cmap="Blues", aspect="auto")
            ax[3].set_xticks([])
            ax[3].set_yticks([])
            ax[4].imshow(inp[:, 21:33].T, cmap="Blues", aspect="auto")
            save_fig(os.path.join(exp_path, f"{env}_input_orientation{i}"))
            
def plot_task_kinematics(config_path, model_path, model_file, exp_path):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """

    create_dir(exp_path)

    options = {"batch_size": 8, "reach_conds": torch.arange(0, 8, 1)}

    for env in env_dict:

        trial_data = _test(config_path, model_path, model_file, options, env=env_dict[env])
    
        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, trial_data["xy"].shape[1])) 

        for i, (tg, xy) in enumerate(zip(trial_data["tg"], trial_data["xy"])):
            plt.scatter(xy[:, 0], xy[:, 1], s=10, color=colors)
            plt.scatter(xy[0, 0], xy[0, 1], s=150, marker='x', color="black")
            plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker='^', color="black")
        save_fig(os.path.join(exp_path, f"{env}_kinematics.png"))

def selectivity_and_clustering():
    # Get selectivity and clusters for different movements
    pass

if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "train_2link_multi":
        train_2link_multi(args.config_path, args.model_path, args.model_file) 
    elif args.experiment == "plot_task_trajectories":
        plot_task_trajectories(args.config_path, args.model_path, args.model_file, args.exp_path) 
    elif args.experiment == "plot_task_kinematics":
        plot_task_kinematics(args.config_path, args.model_path, args.model_file, args.exp_path) 