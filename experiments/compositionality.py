import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp, interpolate_trial

import warnings
warnings.filterwarnings("ignore")

import motornet as mn
from model import RNNPolicy, GRUPolicy
import torch
import os
from utils import load_hp, create_dir, save_fig, load_pickle
import matplotlib.pyplot as plt
import numpy as np
import config
import pickle
from analysis.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
import analysis.plot_utils as plot_utils
import tqdm as tqdm
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
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

        if epoch == "delay":
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




def _epoch_pcs(model_name, epoch):

    exp_path = f"results/{model_name}/compositionality/pcs"
    create_dir(exp_path)

    pca_3d, env_hs = _get_pcs(model_name, batch_size=256, use_reach_conds=False, epoch=epoch, noise=True)

    # Get kinematics and activity in a center out setting
    # On random and delay
    colors = plt.cm.tab10(np.linspace(0, 1, len(env_hs))) 

    # Create a figure
    fig = plt.figure()
    fig.set_size_inches(4, 4)

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection="3d")

    handles = []

    for i, (env_data, env) in enumerate(zip(env_hs, env_dict)):

        # Create patches with no border
        handles.append(mpatches.Patch(color=colors[i], label=env, edgecolor='none'))

        for h in env_data:

            # transform
            h_proj = pca_3d.transform(h)

            # Plot the 3D line
            ax.scatter(h_proj[-1, 0], h_proj[-1, 1], h_proj[-1, 2], color=colors[i], s=25, alpha=0.1)

    # Set labels for axes
    ax.set_xlabel(f'{epoch} PC 1')
    ax.set_ylabel(f'{epoch} PC 2')
    ax.set_zlabel(f'{epoch} PC 3')
    save_fig(os.path.join(exp_path, f"{epoch}_pcs.png"))




def stable_pcs(model_name):
    _epoch_pcs(model_name, "stable")
def delay_pcs(model_name):
    _epoch_pcs(model_name, "delay")
def movement_pcs(model_name):
    _epoch_pcs(model_name, "movement")




def _rule_interpolated_fps(model_name, task1, task2, epoch):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"

    NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
    N_INITS = 1024 # The number of initial states to provide

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "delay_cond": 1, "speed_cond": 5}

    hp = load_hp(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    trial_data1 = _test(model_path, model_file, options, env=env_dict[task1])
    trial_data2 = _test(model_path, model_file, options, env=env_dict[task2])

    if epoch == "delay":
        inp1 = trial_data1["obs"][:, trial_data1["epoch_bounds"]["delay"][1]-1]
        inp2 = trial_data2["obs"][:, trial_data2["epoch_bounds"]["delay"][1]-1]
    elif epoch == "movement":
        inp1 = trial_data1["obs"][:, trial_data1["epoch_bounds"]["movement"][1]-1]
        inp2 = trial_data2["obs"][:, trial_data2["epoch_bounds"]["movement"][1]-1]

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
        
    cond_fps_list = []
    for i, (cond1, cond2) in enumerate(zip(inp1, inp2)):

        # Draw a line from fingertip to goal 
        interpolated_input = cond1.unsqueeze(0) + torch.linspace(0, 1, steps=20).unsqueeze(1) * (cond2 - cond1).unsqueeze(0)

        fps_list = []
        for j, inp in enumerate(interpolated_input):

            # Setup the fixed point finder
            fpf = FixedPointFinder(policy.mrnn, **fpf_hps)

            '''Draw random, noise corrupted samples of those state trajectories
            to use as initial states for the fixed point optimizations.'''

            if epoch == "delay":
                initial_states = fpf.sample_states(trial_data1["h"][i, trial_data1["epoch_bounds"]["delay"][1]-1][None, None, :],
                    n_inits=N_INITS,
                    noise_scale=NOISE_SCALE)
            elif epoch == "movement":
                initial_states = fpf.sample_states(trial_data1["h"][i, trial_data1["epoch_bounds"]["movement"][1]-1][None, None, :],
                    n_inits=N_INITS,
                    noise_scale=NOISE_SCALE)

            # Run the fixed point finder
            unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs=inp[None, :])

            # Add fixed points and their info to dict
            fps_list.append({"fps": unique_fps, "interp_point": j})

        cond_fps_list.append(fps_list)

    # Save all information of fps across tasks to pickle file
    save_name = f'interpolated_fps_{task1}_{task2}_{epoch}'
    fname = os.path.join(model_path, save_name + '.pkl')
    print('interpolated fps saved at {:s}'.format(fname))
    with open(fname, 'wb') as f:
        pickle.dump(cond_fps_list, f)



# Similar tasks
def compute_interpolated_fps_halfcircleclk_halfcirclecclk_delay(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "delay")
def compute_interpolated_fps_halfcircleclk_halfcirclecclk_movement(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "movement")

# dissimilar tasks
def compute_interpolated_fps_halfreach_figure8inv_delay(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "delay")
def compute_interpolated_fps_halfreach_figure8inv_movement(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "movement")




def _plot_interpolated_fps(model_name, task1, task2, epoch):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    load_name = os.path.join(model_path, f"interpolated_fps_{task1}_{task2}_{epoch}.pkl")
    exp_path = f"results/{model_name}/compositionality/interpolated_fps"

    fps = load_pickle(load_name)

    colors = plt.cm.inferno(np.linspace(0, 1, 20)) 

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "delay_cond": 1, "speed_cond": 5}

    trial_data1 = _test(model_path, model_file, options, env=env_dict[task1])
    trial_data2 = _test(model_path, model_file, options, env=env_dict[task2])

    # cond is a list containing the fps for each interpolated rule input for a given condition
    for i, cond in enumerate(fps):
        
        interpolated_fps = [unique_fps["fps"] for unique_fps in cond]
        save_name = f"cond_{i}_interp_fps.png"

        task1_pca = PCA(n_components=2)
        task1_pca.fit(trial_data1["h"][i, trial_data1["epoch_bounds"][f"{epoch}"][0]:trial_data1["epoch_bounds"][f"{epoch}"][1]])

        # Create figure and 3D axes
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')  # or projection='3d'

        task1_h_reduced = task1_pca.transform(trial_data1["h"][i, trial_data1["epoch_bounds"][f"{epoch}"][0]:trial_data1["epoch_bounds"][f"{epoch}"][1]])
        task2_h_reduced = task1_pca.transform(trial_data2["h"][i, trial_data2["epoch_bounds"][f"{epoch}"][0]:trial_data2["epoch_bounds"][f"{epoch}"][1]])

        ax.plot(np.zeros_like(task1_h_reduced[:, 0]), task1_h_reduced[:, 0], task1_h_reduced[:, 1], linewidth=4, color="skyblue")
        ax.plot(np.ones_like(task2_h_reduced[:, 0]), task2_h_reduced[:, 0], task2_h_reduced[:, 1], linewidth=4, color="purple")

        ax.scatter(0, task1_h_reduced[0, 0], task1_h_reduced[0, 1], marker="^", s=100, color="black")
        ax.scatter(0, task1_h_reduced[-1, 0], task1_h_reduced[-1, 1], marker="x", s=100, color="black")

        ax.scatter(1, task2_h_reduced[0, 0], task2_h_reduced[0, 1], marker="^", s=100, color="black")
        ax.scatter(1, task2_h_reduced[-1, 0], task2_h_reduced[-1, 1], marker="x", s=100, color="black")

        for j, fps_step in enumerate(interpolated_fps):
            n_inits = fps_step.n
            for init_idx in range(n_inits):
                zstar = plot_utils.plot_fixed_point(
                            fps_step[init_idx],
                            task1_pca,
                            make_plot=False
                        )
                ax.plot((j/20)*np.ones_like(zstar)[:, 0], zstar[:, 0], zstar[:, 1], marker='.', alpha=0.5, color=colors[j], markersize=12)

        ax.grid(False)
        save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", save_name))




# Similar Tasks
def plot_interpolated_fps_halfcircleclk_halfcirclecclk_delay(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "delay")
def plot_interpolated_fps_halfcircleclk_halfcirclecclk_movement(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "movement")

# dissimilar Tasks
def plot_interpolated_fps_halfreach_figure8inv_delay(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "delay")
def plot_interpolated_fps_halfreach_figure8inv_movement(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "movement")




def _two_task_variance_explained(model_name, task1, task2):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/variance_explained"

    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 1, "speed_cond": 5}

    trial_data_1 = _test(model_path, model_file, options, env=env_dict[task1])
    trial_data_2 = _test(model_path, model_file, options, env=env_dict[task2])

    task1_h = trial_data_1["h"].reshape((-1, trial_data_1["h"].shape[-1]))
    task2_h = trial_data_2["h"].reshape((-1, trial_data_2["h"].shape[-1]))

    pca_task1 = PCA()
    pca_task1.fit(task1_h)

    variance_task_1 = []
    for comp_idx in range(1, 50):
        variance_explained = (pca_task1.components_[:comp_idx] @ task1_h.T.numpy()).var(axis=1).sum() / task1_h.var(axis=0).sum()
        variance_task_1.append(variance_explained)

    variance_task_2 = []
    for comp_idx in range(1, 50):
        variance_explained = (pca_task1.components_[:comp_idx] @ task2_h.T.numpy()).var(axis=1).sum() / task2_h.var(axis=0).sum()
        variance_task_2.append(variance_explained)
        
    plt.rc('figure', figsize=(4, 6))
    plt.plot(variance_task_1, color="black", marker="o", alpha=0.5, label=task1, markersize=10)
    plt.plot(variance_task_2, color="purple", marker="o", alpha=0.5, label=task2, markersize=10)
    plt.legend(loc="best")
    save_fig(os.path.join(exp_path, f"{task1}_{task2}.png"))




def ve_halfreach_figure8inv(model_name):
    _two_task_variance_explained(model_name, "DlyHalfReach", "DlyFigure8Inv")
def ve_halfcircleclk_halfcirclecclk(model_name):
    _two_task_variance_explained(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk")





def angles_vs_distance(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/variance_explained"

    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 1, "speed_cond": 5}

    trial_data_1 = _test(model_path, model_file, options, env=env_dict[task1])
    trial_data_2 = _test(model_path, model_file, options, env=env_dict[task2])

    task1_h = trial_data_1["h"].reshape((-1, trial_data_1["h"].shape[-1]))
    task2_h = trial_data_2["h"].reshape((-1, trial_data_2["h"].shape[-1]))

    pca_task1 = PCA()
    pca_task1.fit(task1_h)

    variance_task_1 = []
    for comp_idx in range(1, 50):
        variance_explained = (pca_task1.components_[:comp_idx] @ task1_h.T.numpy()).var(axis=1).sum() / task1_h.var(axis=0).sum()
        variance_task_1.append(variance_explained)

    variance_task_2 = []
    for comp_idx in range(1, 50):
        variance_explained = (pca_task1.components_[:comp_idx] @ task2_h.T.numpy()).var(axis=1).sum() / task2_h.var(axis=0).sum()
        variance_task_2.append(variance_explained)
        
    plt.rc('figure', figsize=(4, 6))
    plt.plot(variance_task_1, color="black", marker="o", alpha=0.5, label=task1, markersize=10)
    plt.plot(variance_task_2, color="purple", marker="o", alpha=0.5, label=task2, markersize=10)
    plt.legend(loc="best")
    save_fig(os.path.join(exp_path, f"{task1}_{task2}.png"))




if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    # Variance explained between task subspaces
    if args.experiment == "ve_halfreach_figure8inv":
        ve_halfreach_figure8inv(args.model_name)
    elif args.experiment == "ve_halfcircleclk_halfcirclecclk":
        ve_halfcircleclk_halfcirclecclk(args.model_name)

    # Compute Interpolated fps
    elif args.experiment == "compute_interpolated_fps_halfcircleclk_halfcirclecclk_delay":
        compute_interpolated_fps_halfcircleclk_halfcirclecclk_delay(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfcircleclk_halfcirclecclk_movement":
        compute_interpolated_fps_halfcircleclk_halfcirclecclk_movement(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfreach_figure8inv_delay":
        compute_interpolated_fps_halfreach_figure8inv_delay(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfreach_figure8inv_movement":
        compute_interpolated_fps_halfreach_figure8inv_movement(args.model_name)

    # Plot Interpolated fps
    elif args.experiment == "plot_interpolated_fps_halfcircleclk_halfcirclecclk_delay":
        plot_interpolated_fps_halfcircleclk_halfcirclecclk_delay(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfcircleclk_halfcirclecclk_movement":
        plot_interpolated_fps_halfcircleclk_halfcirclecclk_movement(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfreach_figure8inv_delay":
        plot_interpolated_fps_halfreach_figure8inv_delay(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfreach_figure8inv_movement":
        plot_interpolated_fps_halfreach_figure8inv_movement(args.model_name)

    # Epoch pcs
    elif args.experiment == "stable_pcs":
        stable_pcs(args.model_name)
    elif args.experiment == "delay_pcs":
        delay_pcs(args.model_name)
    elif args.experiment == "movement_pcs":
        movement_pcs(args.model_name)
    else:
        raise ValueError("Experiment not in this file")