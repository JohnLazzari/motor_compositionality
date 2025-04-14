from train import train_2link
import motornet as mn
from model import RNNPolicy, GRUPolicy
import torch
import os
from utils import load_hp, create_dir, save_fig, load_pickle
from envs import DlyHalfReach, DlyHalfCircleClk, DlyHalfCircleCClk, DlySinusoid, DlySinusoidInv
from envs import DlyFullReach, DlyFullCircleClk, DlyFullCircleCClk, DlyFigure8, DlyFigure8Inv
import matplotlib.pyplot as plt
import numpy as np
import config
from analysis.clustering import Analysis
import pickle
from analysis.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
import analysis.plot_utils as plot_utils
import tqdm as tqdm
import itertools
from sklearn.decomposition import PCA

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
    # leave hp as default
    train_2link(model_path, model_file)

def train_rnn256_softplus():
    hp = {"hid_size": 256}
    model_path = "checkpoints/rnn256_softplus"
    model_file = "rnn256_softplus.pth"
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn1024_softplus():
    hp = {"hid_size": 1024}
    model_path = "checkpoints/rnn1024_softplus"
    model_file = "rnn1024_softplus.pth"
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn512_relu():
    hp = {"activation_name": "relu"}
    model_path = "checkpoints/rnn512_relu"
    model_file = "rnn512_relu.pth"
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn256_relu():
    hp = {"hid_size": 256, "activation_name": "relu"}
    model_path = "checkpoints/rnn256_relu"
    model_file = "rnn256_relu.pth"
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn1024_relu():
    hp = {"hid_size": 1024, "activation_name": "relu"}
    model_path = "checkpoints/rnn1024_relu"
    model_file = "rnn1024_relu.pth"
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn512_tanh():
    hp = {"activation_name": "tanh"}
    model_path = "checkpoints/rnn512_tanh"
    model_file = "rnn512_tanh.pth"
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn256_tanh():
    hp = {"hid_size": 256, "activation_name": "tanh"}
    model_path = "checkpoints/rnn256_tanh"
    model_file = "rnn256_tanh.pth"
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn1024_tanh():
    hp = {"hid_size": 1024, "activation_name": "tanh"}
    model_path = "checkpoints/rnn1024_tanh"
    model_file = "rnn1024_tanh.pth"
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_gru512():
    hp = {"network": "gru"}
    model_path = "checkpoints/gru512"
    model_file = "gru512.pth"
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_gru256():
    hp = {"hid_size": 256, "network": "gru"}
    model_path = "checkpoints/gru256"
    model_file = "gru256.pth"
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_gru1024():
    hp = {"hid_size": 1024, "network": "gru"}
    model_path = "checkpoints/gru1024"
    model_file = "gru1024.pth"
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)




def _test(model_path, model_file, options, env):
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
    
    obs, info = env.reset(options=options)
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
        with torch.no_grad():
            x, h, action = policy(x, obs, noise=False)
            obs, reward, terminated, info = env.step(timesteps, action=action)

            trial_data["h"].append(h.unsqueeze(1))  # trajectories
            trial_data["action"].append(action.unsqueeze(1))  # targets
            trial_data["obs"].append(obs.unsqueeze(1))  # targets
            trial_data["xy"].append(info["states"]["fingertip"][:, None, :])  # trajectories
            trial_data["tg"].append(info["goal"][:, None, :])  # targets
            trial_data["muscle_acts"].append(info["states"]["muscle"][:, 0].unsqueeze(1))

            timesteps += 1

    for key in trial_data:
        trial_data[key] = torch.cat(trial_data[key], dim=1)
    trial_data["delay_time"] = env.delay_time

    return trial_data




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
    exp_path = f"results/{model_name}/trajectories"

    create_dir(exp_path)

    options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4)}

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env])
    
        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, trial_data["tg"].shape[1])) 

        for i, tg in enumerate(trial_data["tg"]):
            plt.scatter(tg[:, 0], tg[:, 1], s=10, color=colors)
            plt.scatter(tg[0, 0], tg[0, 1], s=150, marker='x', color="black")
            plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker='^', color="black")
        save_fig(os.path.join(exp_path, f"{env}_tg_trajectory.png"))




def plot_task_inputs(model_name):
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
            fig, ax = plt.subplots(6, 1)
            ax[0].imshow(inp[:, :10].T, cmap="seismic", aspect="auto")
            # Remove top and right only (common for minimalist style)
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['bottom'].set_visible(False)
            ax[0].set_xticks([])

            ax[1].plot(inp[:, 10:11], color="blue")
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[1].set_xticks([])

            ax[2].imshow(inp[:, 11:13].T, cmap="seismic", aspect="auto")
            ax[2].spines['top'].set_visible(False)
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['bottom'].set_visible(False)
            ax[2].set_xticks([])

            ax[3].imshow(inp[:, 13:15].T, cmap="seismic", aspect="auto")
            ax[3].spines['top'].set_visible(False)
            ax[3].spines['right'].set_visible(False)
            ax[3].spines['bottom'].set_visible(False)
            ax[3].set_xticks([])

            ax[4].imshow(inp[:, 15:21].T, cmap="seismic", aspect="auto")
            ax[4].spines['top'].set_visible(False)
            ax[4].spines['right'].set_visible(False)
            ax[4].spines['bottom'].set_visible(False)
            ax[4].set_xticks([])

            ax[5].imshow(inp[:, 21:27].T, cmap="seismic", aspect="auto")
            ax[5].spines['top'].set_visible(False)
            ax[5].spines['right'].set_visible(False)
            ax[5].spines['bottom'].set_visible(False)
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

    options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4)}

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env])
    
        # Get kinematics and activity in a center out setting
        # On random and delay
        colors = plt.cm.inferno(np.linspace(0, 1, trial_data["xy"].shape[1])) 

        for i, (tg, xy) in enumerate(zip(trial_data["tg"], trial_data["xy"])):
            plt.scatter(xy[:, 0], xy[:, 1], s=10, color=colors)
            plt.scatter(xy[0, 0], xy[0, 1], s=150, marker='x', color="black")
            plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker='^', color="black")
        save_fig(os.path.join(exp_path, f"{env}_kinematics.png"))




def variance_by_rule(model_name):
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
        task_var = h.var(dim=0)
        mean_task_var = task_var.mean(dim=0)

        var_list.append(mean_task_var)
        task_list.append(env)

    env_var_dict["h_var_all"] = torch.stack(var_list, dim=1).numpy()
    env_var_dict["keys"] = task_list
    
    save_name = 'variance_rule'
    fname = os.path.join(model_path, save_name + '.pkl')
    print('Variance saved at {:s}'.format(fname))
    with open(fname, 'wb') as f:
        pickle.dump(env_var_dict, f)




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
    exp_path = f"results/{model_name}/variance/variance_rule.png"
    
    clustering = Analysis(model_path, "rule")
    clustering.plot_variance(exp_path)




def plot_variance_by_epoch(model_name):
    # Get selectivity and clusters for different movements
    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/variance/variance_epoch.png"
    
    clustering = Analysis(model_path, "epoch")
    clustering.plot_variance(exp_path)




def plot_clusters(model_name):
    # Get selectivity and clusters for different movements
    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/clusters/clustering.png"

    clustering = Analysis(model_path, "rule")
    clustering.plot_2Dvisualization(exp_path)




def compute_fps(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"

    NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
    N_INITS = 1024 # The number of initial states to provide

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 8, "reach_conds": torch.arange(0, 8, 1)}

    hp = load_hp(model_path)
    hp = hp.copy()
    hp["batch_size"] = options["batch_size"]
    
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




def plot_fps(model_name):

    model_path = f"checkpoints/{model_name}"
    load_name = os.path.join(model_path, "model_fps.pkl")
    exp_path = f"results/{model_name}/fps"

    fps = load_pickle(load_name)

    timepoints = [40, 140]
    colors = plt.cm.inferno(np.linspace(0, 1, 8)) 

    for env in fps:
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




def neural_principle_angles(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pc_angles/neural_angles.png"

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1)}

    trial_data_envs = {}
    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env])
        trial_data_envs[env] = trial_data["h"]

    # Get all unique pairs of unit activity across tasks
    combinations = list(itertools.combinations(trial_data_envs, 2))
    angles_dict = {}
    for combination in combinations:
        
        pca1 = PCA()
        pca2 = PCA()

        trial1_data = trial_data_envs[combination[0]].reshape((-1, trial_data_envs[combination[0]].shape[-1]))
        trial2_data = trial_data_envs[combination[1]].reshape((-1, trial_data_envs[combination[1]].shape[-1]))

        pca1.fit(trial1_data)
        pca2.fit(trial2_data)

        pca1_comps = pca1.components_[:12]
        pca2_comps = pca2.components_[:12]

        inner_prod_mat = pca1_comps @ pca2_comps.T # Should be m x m
        U, s, Vh = np.linalg.svd(inner_prod_mat)
        angles = np.degrees(np.arccos(s))

        angles_dict[combination] = angles
    
    for angles in angles_dict:
        plt.plot(angles_dict[angles], label=angles)
    save_fig(exp_path)




def muscle_principle_angles(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/pc_angles/muscle_angles.png"

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1)}

    trial_data_envs = {}
    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env])
        trial_data_envs[env] = trial_data["muscle_acts"]

    # Get all unique pairs of unit activity across tasks
    combinations = list(itertools.combinations(trial_data_envs, 2))
    angles_dict = {}
    for combination in combinations:
        
        pca1 = PCA()
        pca2 = PCA()

        trial1_data = trial_data_envs[combination[0]].reshape((-1, trial_data_envs[combination[0]].shape[-1]))
        trial2_data = trial_data_envs[combination[1]].reshape((-1, trial_data_envs[combination[1]].shape[-1]))

        pca1.fit(trial1_data)
        pca2.fit(trial2_data)

        pca1_comps = pca1.components_[:3]
        pca2_comps = pca2.components_[:3]

        inner_prod_mat = pca1_comps @ pca2_comps.T # Should be m x m
        U, s, Vh = np.linalg.svd(inner_prod_mat)
        angles = np.degrees(np.arccos(s))

        angles_dict[combination] = angles
    
    for angles in angles_dict:
        plt.plot(angles_dict[angles], label=angles)
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
    elif args.experiment == "plot_task_trajectories":
        plot_task_trajectories(args.model_name) 
    elif args.experiment == "plot_task_inputs":
        plot_task_inputs(args.model_name) 
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
    elif args.experiment == "plot_clusters":
        plot_clusters(args.model_name) 
    elif args.experiment == "compute_fps":
        compute_fps(args.model_name) 
    elif args.experiment == "plot_fps":
        plot_fps(args.model_name) 
    elif args.experiment == "neural_principle_angles":
        neural_principle_angles(args.model_name) 
    elif args.experiment == "muscle_principle_angles":
        muscle_principle_angles(args.model_name) 