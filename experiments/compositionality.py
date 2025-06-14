import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp, interpolate_trial

import warnings
warnings.filterwarnings("ignore")

import math
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
from analysis.manifold import principal_angles, vaf_ratio
import tqdm as tqdm
from sklearn.decomposition import PCA, NMF
import sklearn 
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.patches as mpatches
from exp_utils import _test, env_dict, split_movement_epoch, get_interpolation_input, pvalues
from exp_utils import distances_from_combinations, angles_from_combinations, shapes_from_combinations, convert_motif_dict_to_list
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import seaborn as sns
from DSA import DSA
import scipy
from utils import interpolate_trial
import pandas as pd
from plt_utils import standard_2d_ax, ax_3d_no_grid, empty_3d

plt.rcParams.update({'font.size': 18})  # Sets default font size for all text

# This is not included in a task pairing
full_movements = [
    "DlyFullReach",
    "DlyFullCircleClk",
    "DlyFullCircleCClk",
    "DlyFigure8",
    "DlyFigure8Inv"
]

# These are included in task pairings
extension_movements_half = [
    "DlyHalfReach",
    "DlyHalfCircleClk",
    "DlyHalfCircleCClk",
    "DlySinusoid",
    "DlySinusoidInv"
]

extension_movements_full = [
    "DlyFullReach1",
    "DlyFullCircleClk1",
    "DlyFullCircleCClk1",
    "DlyFigure81",
    "DlyFigure8Inv1"
]

retraction_movements_full = [
    "DlyFullReach2",
    "DlyFullCircleClk2",
    "DlyFullCircleCClk2",
    "DlyFigure82",
    "DlyFigure8Inv2"
]

extension_half_combinations = list(itertools.combinations(extension_movements_half, 2))
extension_full_combinations = list(itertools.combinations(extension_movements_full, 2))
extension_tasks = [*extension_half_combinations, *extension_full_combinations]

#all_extensions = [*extension_movements_half, *extension_movements_full]
#extension_tasks = list(itertools.combinations(all_extensions, 2))

retraction_full_combinations = list(itertools.combinations(retraction_movements_full, 2))
retraction_tasks = retraction_full_combinations

subset_tasks = [
    ("DlyHalfCircleClk", "DlyFullCircleClk1"),
    ("DlySinusoid", "DlyFigure81"),
    ("DlyHalfReach", "DlyFullReach1"),
    ("DlyHalfCircleCClk", "DlyFullCircleCClk1"),
    ("DlySinusoidInv", "DlyFigure8Inv1"),
]

rotated_tasks = [
    ("DlyHalfCircleClk", "DlyHalfCircleCClk"),
    ("DlySinusoid", "DlySinusoidInv"),
    ("DlyFullCircleClk1", "DlyFullCircleCClk1"),
    ("DlyFullCircleClk2", "DlyFullCircleCClk2"),
    ("DlyFigure81", "DlyFigure8Inv1"),
    ("DlyFigure82", "DlyFigure8Inv2"),
]

extension_retraction_tasks = list(itertools.product(extension_movements_half, retraction_movements_full))


def _get_pcs(model_name, batch_size=8, epoch=None, speed_cond=5, delay_cond=1, noise=False, system="neural"):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    hp = load_hp(model_path)

    options = {
        "batch_size": batch_size, 
        "speed_cond": speed_cond, 
        "delay_cond": delay_cond
    }

    if system == "neural":
        mode = "h"
        size = hp["hid_size"]
    elif system == "motor":
        mode = "muscle_acts"
        size = 6
    else:
        raise ValueError()

    env_hs = []
    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=noise)

        if epoch == "delay":
            env_hs.append(trial_data[mode][:, trial_data["epoch_bounds"]["delay"][1]-1].unsqueeze(1))
        elif epoch == "stable":
            env_hs.append(trial_data[mode][:, trial_data["epoch_bounds"]["stable"][1]-1].unsqueeze(1))
        elif epoch == "movement":
            env_hs.append(trial_data[mode][
                :, 
                trial_data["epoch_bounds"]["movement"][1]
            ].unsqueeze(1))
        else:
            raise ValueError("not valid epoch")

    pca_3d = PCA(n_components=3)
    pca_3d.fit(torch.cat(env_hs, dim=1).reshape((-1, size)))

    return pca_3d, env_hs




def _epoch_pcs(model_name, epoch, system):

    exp_path = f"results/{model_name}/compositionality/pcs"
    create_dir(exp_path)

    pca_3d, env_hs = _get_pcs(model_name, batch_size=256, epoch=epoch, noise=True, system=system)

    # Get kinematics and activity in a center out setting
    # On random and delay
    colors = plt.cm.tab10(np.linspace(0, 1, len(env_hs))) 
    handles = []

    fig, ax = ax_3d_no_grid()

    for i, (env_data, env) in enumerate(zip(env_hs, env_dict)):

        # Create patches with no border
        handles.append(mpatches.Patch(color=colors[i], label=env, edgecolor='none'))

        for h in env_data:

            # transform
            h_proj = pca_3d.transform(h)

            # Plot the 3D line
            ax.scatter(h_proj[-1, 0], h_proj[-1, 1], h_proj[-1, 2], color=colors[i], s=25, alpha=0.1)

    # Set labels for axes
    save_fig(os.path.join(exp_path, f"{system}_{epoch}_pcs"), eps=True)




def _two_task_pcs(model_name, task1, task2, task1_period, task2_period, system):

    # File names
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/pcs"

    # testing options
    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 1, "speed_cond": 5}
    colors = plt.cm.inferno(np.linspace(0, 1, 10)) 

    # Gather data from test runs
    trial_data1 = _test(model_path, model_file, options, env=env_dict[task1], noise=False)
    trial_data2 = _test(model_path, model_file, options, env=env_dict[task2], noise=False)

    # Get hidden activity during the first or second half of movement (or all)
    trial_data1_h_epoch = split_movement_epoch(trial_data1, task1_period, system)
    C_1, T_1, N_1 = trial_data1_h_epoch.shape
    trial_data2_h_epoch = split_movement_epoch(trial_data2, task2_period, system)
    C_2, T_2, N_2 = trial_data2_h_epoch.shape

    # Combine all trials and timepoints then do PCA
    combined_tasks = torch.cat([trial_data1_h_epoch, trial_data2_h_epoch], dim=1)
    pca_3d = PCA(n_components=3)
    pca_3d.fit(combined_tasks.reshape((-1, combined_tasks.shape[-1])))

    # Transform to get low dimensional trajectories for both tasks
    task1_projected = pca_3d.transform(trial_data1_h_epoch.reshape((-1, trial_data1_h_epoch.shape[-1]))).reshape((C_1, T_1, 3))
    task2_projected = pca_3d.transform(trial_data2_h_epoch.reshape((-1, trial_data2_h_epoch.shape[-1]))).reshape((C_2, T_2, 3))
    
    for c in range(options["batch_size"]):

        fig, ax = ax_3d_no_grid()
        shadow_traj = np.concatenate([task1_projected[c, :], task2_projected[c, :]])
        min_z = np.min(shadow_traj[:, 2])

        ax.plot(task1_projected[c, :, 0], task1_projected[c, :, 1], task1_projected[c, :, 2], linewidth=4, alpha=0.75, color=colors[7], zorder=10)
        ax.plot(task1_projected[c, :, 0], task1_projected[c, :, 1], min_z, linewidth=4, alpha=0.25, color="grey")
        ax.scatter(task1_projected[c, 0, 0], task1_projected[c, 0, 1], task1_projected[c, 0, 2], s=150, marker="^", color=colors[7])
        ax.scatter(task1_projected[c, -1, 0], task1_projected[c, -1, 1], task1_projected[c, -1, 2], s=150, marker="X", color=colors[7])

        ax.plot(task2_projected[c, :, 0], task2_projected[c, :, 1], task2_projected[c, :, 2], linewidth=4, alpha=0.75, color="black", zorder=10)
        ax.plot(task2_projected[c, :, 0], task2_projected[c, :, 1], min_z, linewidth=4, alpha=0.25, color="grey")
        ax.scatter(task2_projected[c, 0, 0], task2_projected[c, 0, 1], task2_projected[c, 0, 2], s=150, marker="^", color="black")
        ax.scatter(task2_projected[c, -1, 0], task2_projected[c, -1, 1], task2_projected[c, -1, 2], s=150, marker="X", color="black")

        save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{system}_{task1}_{task2}_cond_{c}_pcs"), eps=True)

        fig, ax = ax_3d_no_grid()
        mtx1, mtx2, _ = scipy.spatial.procrustes(task1_projected[c, :], task2_projected[c, :])
        shadow_traj = np.concatenate([mtx1, mtx2])
        min_zp = np.min(shadow_traj[:, 2])

        ax.plot(mtx1[:, 0], mtx1[:, 1], mtx1[:, 2], linewidth=4, alpha=0.75, color=colors[7], zorder=10)
        ax.plot(mtx1[:, 0], mtx1[:, 1], min_zp, linewidth=4, alpha=0.25, color="grey")
        ax.scatter(mtx1[0, 0], mtx1[0, 1], mtx1[0, 2], s=150, marker="^", color=colors[7])
        ax.scatter(mtx1[-1, 0], mtx1[-1, 1], mtx1[-1, 2], s=150, marker="X", color=colors[7])

        ax.plot(mtx2[:, 0], mtx2[:, 1], mtx2[:, 2], linewidth=4, alpha=0.75, color="black", zorder=10)
        ax.plot(mtx2[:, 0], mtx2[:, 1], min_zp, linewidth=4, alpha=0.25, color="grey")
        ax.scatter(mtx2[0, 0], mtx2[0, 1], mtx2[0, 2], s=150, marker="^", color="black")
        ax.scatter(mtx2[-1, 0], mtx2[-1, 1], mtx2[-1, 2], s=150, marker="X", color="black")

        save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{system}_{task1}_{task2}_cond_{c}_procrustes_pcs"), eps=True)






def stable_pcs_neural(model_name):
    _epoch_pcs(model_name, "stable", "neural")
def delay_pcs_neural(model_name):
    _epoch_pcs(model_name, "delay", "neural")
def movement_pcs_neural(model_name):
    _epoch_pcs(model_name, "movement", "neural")

def stable_pcs_motor(model_name):
    _epoch_pcs(model_name, "stable", "motor")
def delay_pcs_motor(model_name):
    _epoch_pcs(model_name, "delay", "motor")
def movement_pcs_motor(model_name):
    _epoch_pcs(model_name, "movement", "motor")

def neural_two_task_pcs_sinusoid_fullcircleclk(model_name):
    _two_task_pcs(model_name, "DlySinusoid", "DlyFullCircleClk", "all", "second", "h")
def muscle_two_task_pcs_sinusoid_fullcircleclk(model_name):
    _two_task_pcs(model_name, "DlySinusoid", "DlyFullCircleClk", "all", "second", "muscle_acts")

def neural_two_task_pcs_halfcircleclk_halfcirclecclk(model_name):
    _two_task_pcs(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "all", "all", "h")
def muscle_two_task_pcs_halfcircleclk_halfcirclecclk(model_name):
    _two_task_pcs(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "all", "all", "muscle_acts")




def _interpolated_fps(model_name, task1, task2,  epoch, task1_period="all", task2_period="all", input_component=None, 
    add_new_rule_inputs=False, num_new_inputs=10):

    # Task 1 and task 2 period represent either collecting only half of the movement, or all, in specific periods
    # These values can be first, second, or all

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"

    NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
    N_INITS = 1024 # The number of initial states to provide

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 16, "reach_conds": torch.arange(0, 32, 2), "delay_cond": 1, "speed_cond": 5}

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
            device=device,
            add_new_rule_inputs=add_new_rule_inputs,
            num_new_inputs=num_new_inputs
        )
    elif hp["network"] == "gru":
        policy = GRUPolicy(hp["inp_size"], hp["hid_size"], effector.n_muscles, batch_first=True)
    else:
        raise ValueError("Not a valid architecture")

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=torch.device('cpu'))
    policy.load_state_dict(checkpoint['agent_state_dict'])

    trial_data1 = _test(model_path, model_file, options, env=env_dict[task1], add_new_rule_inputs=add_new_rule_inputs, num_new_inputs=num_new_inputs)
    trial_data2 = _test(model_path, model_file, options, env=env_dict[task2], add_new_rule_inputs=add_new_rule_inputs, num_new_inputs=num_new_inputs)

    if epoch == "delay":

        # Get inputs and x and h from desired timepoint
        inp1 = trial_data1["obs"][:, trial_data1["epoch_bounds"]["delay"][1]-1]
        inp2 = trial_data2["obs"][:, trial_data2["epoch_bounds"]["delay"][1]-1]

    elif epoch == "movement":

        inp1 = get_interpolation_input(trial_data1, task1_period)
        inp2 = get_interpolation_input(trial_data2, task2_period)

    '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
    descriptions of available hyperparameters.'''

    fpf_hps = {
        'max_iters': 250,
        'lr_init': 1.,
        'outlier_distance_scale': 10.0,
        'verbose': True, 
        'super_verbose': False,
        'tol_unique': 1,
        'do_compute_jacobians': True}
        
    cond_fps_list = []
    for c, (cond1, cond2) in enumerate(zip(inp1, inp2)):

        # Setup environment and initialize it
        env = env_dict[task1](effector=mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle()))
        obs, info = env.reset(testing=True, options=options)

        # May need to change in the future if I do middle of movement
        x = torch.zeros(size=(1, hp["hid_size"]))
        h = torch.zeros(size=(1, hp["hid_size"]))

        if epoch == "delay":
            timesteps = env.epoch_bounds["delay"][1]-1
        elif epoch == "movement":
            timesteps = middle_movement1
            
        # Draw a line from fingertip to goal 
        if input_component == None:
            interpolated_input = cond1.unsqueeze(0) + \
                torch.linspace(0, 1, steps=20).unsqueeze(1) * (cond2 - cond1).unsqueeze(0)

        elif input_component == "rule":
            interpolated_input_rule = cond1[:10].unsqueeze(0) + \
                torch.linspace(0, 1, steps=20).unsqueeze(1) * (cond2[:10] - cond1[:10]).unsqueeze(0)
            fixed_inp = cond1[10:].repeat(20, 1)
            interpolated_input = torch.cat([interpolated_input_rule, fixed_inp], dim=1)

        elif input_component == "proprioception":
            fixed_inp_pre = cond1[:16].repeat(20, 1)
            interpolated_input_proprioception = cond1[16:].unsqueeze(0) + \
                torch.linspace(0, 1, steps=20).unsqueeze(1) * (cond2[16:] - cond1[16:]).unsqueeze(0)
            interpolated_input = torch.cat([fixed_inp_pre, interpolated_input_proprioception], dim=1)

        fps_list = []
        # Going thorugh each interpolated input
        for i, inp in enumerate(interpolated_input):

            # Setup the fixed point finder
            fpf = FixedPointFinder(policy.mrnn, **fpf_hps)

            '''Draw random, noise corrupted samples of those state trajectories
            to use as initial states for the fixed point optimizations.'''

            # Currently using original h for initial states
            initial_states = fpf.sample_states(trial_data1["h"][c:c+1, timesteps:timesteps+1],
                n_inits=N_INITS,
                noise_scale=NOISE_SCALE)

            # Run the fixed point finder
            unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs=inp[None, :])

            # Add fixed points and their info to dict
            fps_list.append(
                {"fps": unique_fps, 
                "interp_point": i 
                }
            )

        cond_fps_list.append(fps_list)

    # Save all information of fps across tasks to pickle file
    save_name = f'interpolated_fps_{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}'
    fname = os.path.join(model_path, save_name + '.pkl')
    print('interpolated fps saved at {:s}'.format(fname))
    with open(fname, 'wb') as f:
        pickle.dump(cond_fps_list, f)





#---------------------------------------------------------------- Subset Pair

# Delay period with different input interpolations
def compute_interpolated_fps_halfreach_fullreach_delay(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay")
# Movement period with different input interpolations
def compute_interpolated_fps_halfreach_fullreach_movement(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", task1_period="all", task2_period="first")

#---------------------------------------------------------------- Extension Pair

# Delay period with different input interpolations
def compute_interpolated_fps_halfcircleclk_sinusoidinv_delay(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "delay")
# Movement period with different input interpolations
def compute_interpolated_fps_halfcircleclk_sinusoidinv_movement(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "movement", task1_period="all", task2_period="all")

#---------------------------------------------------------------- Retraction Pair

# Delay period with different input interpolations
def compute_interpolated_fps_fullcircleclk_figure8_delay(model_name):
    _interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "delay")
# Movement period with different input interpolations
def compute_interpolated_fps_fullcircleclk_figure8_movement(model_name):
    _interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "movement", task1_period="second", task2_period="second")

#---------------------------------------------------------------- Extension-Retraction Pair

# Delay period with different input interpolations
def compute_interpolated_fps_sinusoid_fullreach_delay(model_name):
    _interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "delay")
# Movement period with different input interpolations
def compute_interpolated_fps_sinusoid_fullreach_movement(model_name):
    _interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "movement", task1_period="all", task2_period="second")




def run_all_compute_interpolated_fps(model_name):
    compute_interpolated_fps_halfreach_fullreach_delay(model_name)
    compute_interpolated_fps_halfreach_fullreach_movement(model_name)
    compute_interpolated_fps_halfcircleclk_sinusoidinv_delay(model_name)
    compute_interpolated_fps_halfcircleclk_sinusoidinv_movement(model_name)
    compute_interpolated_fps_fullcircleclk_figure8_delay(model_name)
    compute_interpolated_fps_fullcircleclk_figure8_movement(model_name)
    compute_interpolated_fps_sinusoid_fullreach_delay(model_name)
    compute_interpolated_fps_sinusoid_fullreach_movement(model_name)





def _plot_interpolated_fps(model_name, task1, task2, epoch, task1_period="all", task2_period="all", input_component=None,
        add_new_rule_inputs=False, num_new_inputs=10, save_metrics=False, y_dist=1):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    load_name = os.path.join(model_path, f"interpolated_fps_{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}.pkl")
    exp_path = f"results/{model_name}/compositionality/interpolated_fps"

    fps = load_pickle(load_name)

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 16, "reach_conds": torch.arange(0, 32, 2), "delay_cond": 1, "speed_cond": 5}
    colors_alpha = plt.cm.magma(np.linspace(0, 1, 20)) 

    trial_data1 = _test(model_path, model_file, options, env=env_dict[task1], add_new_rule_inputs=add_new_rule_inputs, num_new_inputs=num_new_inputs)
    trial_data2 = _test(model_path, model_file, options, env=env_dict[task2], add_new_rule_inputs=add_new_rule_inputs, num_new_inputs=num_new_inputs)

    trial_data1_h_epoch = trial_data1["h"][:, trial_data1["epoch_bounds"][epoch][0]:trial_data1["epoch_bounds"][epoch][1]] 
    trial_data2_h_epoch = trial_data2["h"][:, trial_data2["epoch_bounds"][epoch][0]:trial_data2["epoch_bounds"][epoch][1]] 
    halfway_task1 = int(trial_data1_h_epoch.shape[1] / 2)

    two_task_pca = PCA(n_components=2)
    two_task_pca.fit(trial_data1_h_epoch.reshape((-1, trial_data1_h_epoch.shape[-1])))

    fig, ax = ax_3d_no_grid()

    # cond is a list containing the fps for each interpolated rule input for a given condition
    for c, cond in enumerate(fps):
        
        interpolated_fps = [unique_fps["fps"] for unique_fps in cond]

        for i, fps_step in enumerate(interpolated_fps):
            n_inits = fps_step.n
            for init_idx in range(n_inits):
                zstar = plot_utils.plot_fixed_point(
                            fps_step[init_idx],
                            two_task_pca,
                            make_plot=False
                        )
                # Stability of top eigenvalue
                stability = np.abs(fps_step[init_idx].eigval_J_xstar[0, 0])
                if stability > 1:
                    ax.scatter(i/20, zstar[:, 0], zstar[:, 1], marker='.', alpha=0.75, edgecolors=colors_alpha[i], facecolors="w", s=250)
                else:
                    ax.scatter(i/20, zstar[:, 0], zstar[:, 1], marker='.', alpha=0.75, color=colors_alpha[i], s=250)

    save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}"), eps=True)

    trajectory_point = trial_data1_h_epoch[:, halfway_task1]

    fig, ax = standard_2d_ax()

    colors_conds = plt.cm.inferno(np.linspace(0, 1, 16)) 
    # Generate a plot of max eigenvalues
    max_eigs_conds = []
    for c, cond in enumerate(fps):
        interpolated_fps = [unique_fps["fps"] for unique_fps in cond]
        max_eigs = []
        chosen_fps = []
        for i, fps_step in enumerate(interpolated_fps):
            n_inits = fps_step.n
            max_eig = 0
            dist = np.inf

            if i == 0:
                for init_idx in range(n_inits):
                    cur_dist = np.linalg.norm(trajectory_point[c] - fps_step[init_idx].xstar)
                    if cur_dist < dist:
                        dist = cur_dist
                        chosen_fp = fps_step[init_idx]
            else:
                for init_idx in range(n_inits):
                    cur_dist = np.linalg.norm(chosen_fps[i-1].xstar - fps_step[init_idx].xstar)
                    if cur_dist < dist:
                        dist = cur_dist
                        chosen_fp = fps_step[init_idx]
            chosen_fps.append(chosen_fp)

            max_eig = chosen_fp.eigval_J_xstar[0, 0].real
            max_eigs.append(max_eig)

        x = np.arange(1, 21)
        eig_mean = np.mean(max_eigs)
        ax.scatter(x, max_eigs, marker="o", color=colors_conds[c], s=200, alpha=0.75)
        ax.set_ylim([0.5, 1.2])
        max_eigs_conds.append([np.abs(max_eigs[i+1] - max_eigs[i]) for i in range(len(max_eigs)-1)])
    save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}_max_eigs"), eps=True)

    fig, ax = standard_2d_ax()

    # Generate a plot of fp distances
    euc_dists_conds = []
    for c, cond in enumerate(fps):
        interpolated_fps = [unique_fps["fps"] for unique_fps in cond]
        chosen_fps = []
        for i, fps_step in enumerate(interpolated_fps):
            n_inits = fps_step.n
            dist = np.inf
            if i == 0:
                for init_idx in range(n_inits):
                    cur_dist = np.linalg.norm(trajectory_point[c] - fps_step[init_idx].xstar)
                    if cur_dist < dist:
                        dist = cur_dist
                        chosen_fp = fps_step[init_idx].xstar
            else:
                for init_idx in range(n_inits):
                    cur_dist = np.linalg.norm(chosen_fps[i-1] - fps_step[init_idx].xstar)
                    if cur_dist < dist:
                        dist = cur_dist
                        chosen_fp = fps_step[init_idx].xstar
            chosen_fps.append(chosen_fp)
        
        dist_list = [np.linalg.norm(chosen_fps[i+1] - chosen_fps[i]) for i in range(len(chosen_fps)-1)]
        x = np.arange(1, 20)
        ax.scatter(x, dist_list, marker="o", color=colors_conds[c], s=200, alpha=0.75)
        ax.set_ylim([0, y_dist])
        euc_dists_conds.append([np.abs(dist_list[i+1] - dist_list[i]) for i in range(len(dist_list)-1)])
    save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}_dists"), eps=True)
        

    fig, ax = empty_3d()

    colors_traj = plt.cm.inferno(np.linspace(0, 1, 10)) 

    trial_data1_h_epoch = split_movement_epoch(trial_data1, task1_period, "h")
    trial_data2_h_epoch = split_movement_epoch(trial_data2, task2_period, "h")

    # Get trajectories during task period
    combined_tasks = torch.cat([trial_data1_h_epoch, trial_data2_h_epoch], dim=1)

    two_task_pca_3d = PCA(n_components=3)
    two_task_pca_3d.fit(combined_tasks.reshape((-1, combined_tasks.shape[-1])))

    trial_data_1_projected = two_task_pca_3d.transform(trial_data1_h_epoch.reshape((-1, trial_data1_h_epoch.shape[-1])))
    trial_data_2_projected = two_task_pca_3d.transform(trial_data2_h_epoch.reshape((-1, trial_data2_h_epoch.shape[-1])))

    trial_data_1_projected = trial_data_1_projected.reshape((
        trial_data1_h_epoch.shape[0], 
        trial_data1_h_epoch.shape[1],
        3
    ))
    
    trial_data_2_projected = trial_data_2_projected.reshape((
        trial_data2_h_epoch.shape[0], 
        trial_data2_h_epoch.shape[1],
        3
    ))

    for c, condition in enumerate(trial_data_1_projected):
        ax.plot(condition[:, 0], condition[:, 1], condition[:, 2], linewidth=4, color=colors_traj[7], alpha=0.75, zorder=10)
        ax.scatter(condition[0, 0], condition[0, 1], condition[0, 2], s=150, marker="^", color=colors_traj[7])
        ax.scatter(condition[-1, 0], condition[-1, 1], condition[-1, 2], s=150, marker="X", color=colors_traj[7])
    for c, condition in enumerate(trial_data_2_projected):
        ax.plot(condition[:, 0], condition[:, 1], condition[:, 2], linewidth=4, color="black", alpha=0.75, zorder=10)
        ax.scatter(condition[0, 0], condition[0, 1], condition[0, 2], s=150, marker="^", color="black", zorder=20)
        ax.scatter(condition[-1, 0], condition[-1, 1], condition[-1, 2], s=150, marker="X", color="black", zorder=20)

    save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}_pca"), eps=True)

    if save_metrics:
        return max_eigs_conds, euc_dists_conds
    





#---------------------------------------------------------------- Subset Pair
def plot_interpolated_fps_halfreach_fullreach_movement(model_name, save_metrics=False):
    if save_metrics:
        max_eigs, euc_dist = _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", 
            task1_period="all", task2_period="first", save_metrics=save_metrics)
        return max_eigs, euc_dist
    else:
        _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", 
            task1_period="all", task2_period="first", save_metrics=save_metrics)
#---------------------------------------------------------------- Extension Pair
def plot_interpolated_fps_halfcircleclk_sinusoidinv_movement(model_name, save_metrics=False):
    if save_metrics:
        max_eigs, euc_dist = _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "movement", 
            task1_period="all", task2_period="all", save_metrics=save_metrics)
        return max_eigs, euc_dist
    else:
        _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "movement", 
            task1_period="all", task2_period="all", save_metrics=save_metrics)
#---------------------------------------------------------------- Retraction Pair
def plot_interpolated_fps_fullcircleclk_figure8_movement(model_name, save_metrics=False):
    if save_metrics:
        max_eigs, euc_dist = _plot_interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "movement", 
            task1_period="second", task2_period="second", save_metrics=save_metrics)
        return max_eigs, euc_dist
    else:
        _plot_interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "movement", 
            task1_period="second", task2_period="second", save_metrics=save_metrics)
#---------------------------------------------------------------- Extension-Retraction Pair
def plot_interpolated_fps_sinusoid_fullreach_movement(model_name, save_metrics=False):
    if save_metrics:
        max_eigs, euc_dist = _plot_interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "movement", 
            task1_period="all", task2_period="second", save_metrics=save_metrics)
        return max_eigs, euc_dist
    else:
        _plot_interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "movement", 
            task1_period="all", task2_period="second", save_metrics=save_metrics)



def run_all_plot_interpolated_fps(model_name):
    # Add data from each task
    plot_interpolated_fps_halfreach_fullreach_movement(model_name, save_metrics=False)
    plot_interpolated_fps_halfcircleclk_sinusoidinv_movement(model_name, save_metrics=False)
    plot_interpolated_fps_fullcircleclk_figure8_movement(model_name, save_metrics=False)
    plot_interpolated_fps_sinusoid_fullreach_movement(model_name, save_metrics=False)




def _plot_bar(combinations, metric, exp_path, metric_name, combination_labels, combination_colors):

    fig, ax = standard_2d_ax()

    combination_means = []
    combination_stds = []
    combination_data = {}
    for l, combination in enumerate(combinations):
        task_metric = _convert_motif_dict_to_list(combination, metric)

        """
            This is for finding examples for two_task_pcs. Find task pairs with large disparities, etc.
            Delete this once done for sure, change as needed to find different examples
        """

        if combination_labels[l] == "extension_tasks" and metric_name == "muscle_shapes":
            min_val = np.argmax(task_metric)
            condition_val = min_val % 32
            min_val /= 32
            min_val = math.floor(min_val)
            print(f"ext pair with highest muscle shape is: {combination[min_val]}, {condition_val}")

        combination_data[combination_labels[l]] = task_metric
        combination_means.append(sum(task_metric) / len(task_metric))
        combination_stds.append(np.std(task_metric, ddof=1))

    # Convert values to list
    data_values = list(combination_data.values())
    labels = list(combination_data.keys())
    
    ax.axhline(combination_means[-1], color="dimgrey", linestyle="dashed")
    parts = ax.violinplot(data_values[:-1], showmeans=True)

    # Custom colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(combination_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(1.2)
    parts['cbars'].set_edgecolor('black')
    parts['cmins'].set_edgecolor('black')
    parts['cmaxes'].set_edgecolor('black')
    parts['cmeans'].set_color('black')

    if "angles" in metric_name:
        plt.yticks([0, 1.5])
    elif "shapes" in metric_name:
        plt.yticks([0, 1])
    plt.xticks([])
    save_fig(os.path.join(exp_path, "movement", metric_name), eps=True)

    combination_list = list(combination_data.keys())
    pvalues(combination_list, combination_data, metric_name)

    
def _plot_scatter(all_combinations, combinations, combination_colors, metric1, metric2, exp_path, metric1_name, metric2_name):

    fig, ax = standard_2d_ax()

    task_metric1 = _convert_motif_dict_to_list(all_combinations, metric1)
    task_metric2 = _convert_motif_dict_to_list(all_combinations, metric2)

    metric1_list = np.array(task_metric1).reshape((-1, 1))
    metric2_list = np.array(task_metric2).reshape((-1, 1))

    regression = LinearRegression()
    regression.fit(metric1_list, metric2_list)
    print(f"R^2 {metric1_name} to {metric2_name}: ", regression.score(metric1_list, metric2_list))
    x = np.linspace(0, max(metric1_list))
    ax.plot(x, regression.coef_ * x + regression.intercept_, color="black")

    for c, combination in enumerate(combinations[:-1]):
        task_metric1_comb = _convert_motif_dict_to_list(combination, metric1)
        task_metric2_comb = _convert_motif_dict_to_list(combination, metric2)
        ax.scatter(task_metric1_comb, task_metric2_comb, s=100, alpha=0.25, color=combination_colors[c])
    save_fig(os.path.join(exp_path, "movement", f"{metric1_name} vs {metric2_name}"), eps=True)


def _trajectory_alignment(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/alignment"

    options = {"batch_size": 32, "reach_conds": np.tile(np.arange(0, 32, 1), int(32/32)), "speed_cond": 5}

    trial_data_h = {}
    trial_data_muscle = {}
    combinations_h = {}
    combinations_muscle = {}
    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=False)

        if env == "DlyFullReach" or env == "DlyFullCircleClk" or env == "DlyFullCircleCClk" or env == "DlyFigure8" or env == "DlyFigure8Inv":

            halfway = int((trial_data["epoch_bounds"]["movement"][0] + trial_data["epoch_bounds"]["movement"][1]) / 2)

            trial_data_h[env+"1"] = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]
            trial_data_muscle[env+"1"] = trial_data["muscle_acts"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]

            trial_data_h[env+"2"] = trial_data["h"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]
            trial_data_muscle[env+"2"] = trial_data["muscle_acts"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]
        
        else:

            trial_data_h[env] = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            trial_data_muscle[env] = trial_data["muscle_acts"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]

    # Get all unique pairs of unit activity across tasks
    combination_labels = list(itertools.combinations(trial_data_h, 2))
    for combination_label in combination_labels:
        combinations_h[combination_label] = (
            trial_data_h[combination_label[0]],
            trial_data_h[combination_label[1]]
        )
        combinations_muscle[combination_label] = (
            trial_data_muscle[combination_label[0]],
            trial_data_muscle[combination_label[1]]
        )
    
    print("Computing Distances...")
    distances_h = distances_from_combinations(combinations_h, options["batch_size"])
    distances_muscle = distances_from_combinations(combinations_muscle, options["batch_size"])

    print("Computing Angles...")
    angles_h = angles_from_combinations(combinations_h, options["batch_size"])
    angles_muscle = shapes_from_combinations(combinations_muscle, options["batch_size"])

    print("Computing Shapes...")
    shapes_h = shapes_from_combinations(combinations_h, options["batch_size"])
    shapes_muscle = shapes_from_combinations(combinations_muscle, options["batch_size"])

    all_subsets = [
        subset_tasks,
        retraction_tasks,
        extension_tasks, 
        extension_retraction_tasks,
        combination_labels
    ]

    all_subset_labels = [
        "subset_tasks",
        "retraction_tasks",
        "extension_tasks",
        "extension_retraction_tasks",
        "all_tasks"
    ]

    all_subset_colors = ["purple", "pink", "blue", "orange", "grey"]

    all_metrics = {
        "neural_distances": distances_h,
        "muscle_distances": distances_muscle,
        "neural_angles": angles_h,
        "muscle_angles": angles_muscle,
        "neural_shapes": shapes_h,
        "muscle_shapes": shapes_muscle,
    }

    # Make each bar plot
    for metric in all_metrics:
        _plot_bar(all_subsets, all_metrics[metric], exp_path, metric, all_subset_labels, all_subset_colors)

    # -------------------------------------- NEURAL AND MUSCLE SHAPE DISTRIBUTIONS

    fig, ax = standard_2d_ax()
    all_shapes_h = _convert_motif_dict_to_list(combination_labels, shapes_h)
    all_shapes_muscle = _convert_motif_dict_to_list(combination_labels, shapes_muscle)

    bins = np.linspace(0, 1, 15)
    weights_data_h = np.ones_like(all_shapes_h) / len(all_shapes_h)
    weights_data_muscle = np.ones_like(all_shapes_muscle) / len(all_shapes_muscle)
    plt.hist(all_shapes_h, color="blue", alpha=0.5, bins=bins, weights=weights_data_h)
    plt.hist(all_shapes_muscle, color="purple", alpha=0.5, bins=bins, weights=weights_data_muscle)
    plt.axvline(sum(all_shapes_h)/len(all_shapes_h), color="blue", linestyle="dashed", linewidth=2)
    plt.axvline(sum(all_shapes_muscle)/len(all_shapes_muscle), color="purple", linestyle="dashed", linewidth=2)
    plt.xlim([0, 1])
    save_fig(os.path.join(exp_path, "movement", "neural_muscle_shape_dists"), eps=True)

    # ------------------------------------------------------------- ANGLE DISTRIBUTIONS

    fig, ax = standard_2d_ax()
    angle_h_dist = _convert_motif_dict_to_list(combination_labels, angles_h)
    angle_muscle_dist = _convert_motif_dict_to_list(combination_labels, angles_muscle)

    bins = np.linspace(0, 1.5, 15)
    weights_data_h = np.ones_like(angle_h_dist) / len(angle_h_dist)
    weights_data_muscle = np.ones_like(angle_muscle_dist) / len(angle_muscle_dist)
    plt.hist(angle_h_dist, color="blue", alpha=0.5, bins=bins, weights=weights_data_h)
    plt.hist(angle_muscle_dist, color="purple", alpha=0.5, bins=bins, weights=weights_data_muscle)
    plt.axvline(sum(angle_h_dist)/len(angle_h_dist), color="blue", linestyle="dashed", linewidth=2)
    plt.axvline(sum(angle_muscle_dist)/len(angle_muscle_dist), color="purple", linestyle="dashed", linewidth=2)
    plt.xlim([0, 1.5])
    save_fig(os.path.join(exp_path, "movement", "neural_muscle_angle_dists"), eps=True)

    # -------------------------------------- SCATTER PLOTS

    for idx1, metric1 in enumerate(all_metrics):
        for idx2, metric2 in enumerate(all_metrics):
            if idx1 != idx2:
                _plot_scatter(combination_labels, all_subsets, all_subset_colors, all_metrics[metric1], all_metrics[metric2], exp_path, metric1, metric2)





def trajectory_alignment_movement(model_name):
    _trajectory_alignment(model_name)




def _get_vaf_combination(combination_labels, data, mode, comp_range):

    # Initialize the full pc dict
    all_vaf_list_means = []
    all_vaf_list_stds = []

    condition_tuple_dict = {}
    condition_label_dict = {}

    for combination in combination_labels:
        if combination not in data:
            combination = (combination[1], combination[0])
        for c, (task1_condition, task2_condition) in enumerate(zip(data[combination][0], data[combination][1])):
            if c not in condition_tuple_dict:
                condition_tuple_dict[c] = []
                condition_label_dict[c] = []
            # This should return a list of the variance explained for each pc
            condition_tuple_dict[c].append((task1_condition, task2_condition))
            condition_label_dict[c].append(combination)

    for pc in range(1, comp_range):
        condition_vaf_list = []
        for condition in condition_tuple_dict.values():
            # This should return a list of the variance explained for each pc
            ratio_list = vaf_ratio(condition, mode=mode, num_comps=pc, control=False)
            condition_vaf_list.extend(ratio_list)
        all_vaf_list_means.append(np.array(condition_vaf_list).mean())
        all_vaf_list_stds.append(np.array(condition_vaf_list).std())
    
    return np.array(all_vaf_list_means), np.array(all_vaf_list_stds)


def task_vaf_ratio(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/task_vaf_ratio"

    options = {"batch_size": 32, "reach_conds": np.tile(np.arange(0, 32, 1), int(32/32)), "speed_cond": 5}

    trial_data_h = {}
    trial_data_muscle = {}
    combinations_h = {}
    combinations_muscle = {}
    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=False)

        if env == "DlyFullReach" or env == "DlyFullCircleClk" or env == "DlyFullCircleCClk" or env == "DlyFigure8" or env == "DlyFigure8Inv":

            halfway = int((trial_data["epoch_bounds"]["movement"][0] + trial_data["epoch_bounds"]["movement"][1]) / 2)

            trial_data_h[env+"1"] = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]
            trial_data_h[env+"2"] = trial_data["h"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]

            trial_data_muscle[env+"1"] = trial_data["muscle_acts"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]
            trial_data_muscle[env+"2"] = trial_data["muscle_acts"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]
        
        else:

            trial_data_h[env] = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            trial_data_muscle[env] = trial_data["muscle_acts"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
        

    # Get all unique pairs of unit activity across tasks
    combination_labels = list(itertools.combinations(trial_data_h, 2))
    for combination_label in combination_labels:
        combinations_h[combination_label] = (
            trial_data_h[combination_label[0]],
            trial_data_h[combination_label[1]]
        )
        combinations_muscle[combination_label] = (
            trial_data_muscle[combination_label[0]],
            trial_data_muscle[combination_label[1]]
        )

    all_subsets = {
        "extension_tasks": extension_tasks, 
        "retraction_tasks": retraction_tasks, 
        "subset_tasks": subset_tasks, 
        "extension_retraction_tasks": extension_retraction_tasks, 
    }

    all_subsets_colors = {
        "extension_tasks": "blue",
        "retraction_tasks": "pink",
        "subset_tasks": "purple",
        "extension_retraction_tasks": "orange"
    }

    comp_range = 11

    # Plotting vaf for different number of pc components neural

    subset_pc_dict_means = {}
    subset_pc_dict_stds = {}
    for subset in all_subsets:
        subset_pc_dict_means[subset], subset_pc_dict_stds[subset] = _get_vaf_combination(all_subsets[subset], combinations_h, "h", comp_range)
    all_task_pc_means, all_task_pc_stds = _get_vaf_combination(combination_labels, combinations_h, "h", comp_range)
    
    fig, ax = standard_2d_ax()
    x = np.arange(1, comp_range)
    for subset in subset_pc_dict_means:
        ax.plot(x, subset_pc_dict_means[subset], linewidth=4, marker='o', markersize=15, alpha=0.75, color=all_subsets_colors[subset])
        ax.fill_between(
            x, 
            subset_pc_dict_means[subset] - subset_pc_dict_stds[subset], 
            subset_pc_dict_means[subset] + subset_pc_dict_stds[subset], 
            color=all_subsets_colors[subset], 
            alpha=0.25
        )
    ax.plot(x, all_task_pc_means, linewidth=4, marker='o', markersize=15, alpha=0.75, color="grey")
    ax.fill_between(
        x, 
        all_task_pc_means - all_task_pc_stds, 
        all_task_pc_means + all_task_pc_stds, 
        color="grey", 
        alpha=0.25
    )

    comp_range = 7
    ax.set_ylim([0, 1.1])
    save_fig(os.path.join(exp_path, "vaf_ratio_neural"), eps=True)

    # Plotting vaf for different number of pc components muscle
    subset_pc_dict_means = {}
    subset_pc_dict_stds = {}
    for subset in all_subsets:
        subset_pc_dict_means[subset], subset_pc_dict_stds[subset] = _get_vaf_combination(all_subsets[subset], combinations_muscle, "muscle_acts", comp_range)
    all_task_pc_means, all_task_pc_stds = _get_vaf_combination(combination_labels, combinations_muscle, "muscle_acts", comp_range)
    
    fig, ax = standard_2d_ax()
    x = np.arange(1, comp_range)
    for subset in subset_pc_dict_means:
        ax.plot(x, subset_pc_dict_means[subset], linewidth=4, marker='o', markersize=15, alpha=0.75, color=all_subsets_colors[subset])
        ax.fill_between(
            x, 
            subset_pc_dict_means[subset] - subset_pc_dict_stds[subset], 
            subset_pc_dict_means[subset] + subset_pc_dict_stds[subset], 
            color=all_subsets_colors[subset], 
            alpha=0.25
        )
    ax.plot(x, all_task_pc_means, linewidth=4, marker='o', markersize=15, alpha=0.75, color="grey")
    ax.fill_between(
        x, 
        all_task_pc_means - all_task_pc_stds, 
        all_task_pc_means + all_task_pc_stds, 
        color="grey", 
        alpha=0.25
    )

    ax.set_ylim([0, 1.1])
    save_fig(os.path.join(exp_path, "vaf_ratio_muscle"), eps=True)




def dsa_similarity_matrix(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/dsa"

    options = {"batch_size": 32*4, "reach_conds": np.tile(np.arange(0, 32, 1), 4), "speed_cond": 5}

    trial_data_h = []
    trial_data_colors = []

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=True)

        if env == "DlyFullReach" or env == "DlyFullCircleClk" or env == "DlyFullCircleCClk" or env == "DlyFigure8" or env == "DlyFigure8Inv":

            halfway = int((trial_data["epoch_bounds"]["movement"][0] + trial_data["epoch_bounds"]["movement"][1]) / 2)
            extend = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]
            retract = trial_data["h"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]

            pca_extend = PCA(n_components=12)
            extend_reduced = pca_extend.fit_transform(extend.reshape((-1, extend.shape[-1])))
            extend_reduced = extend_reduced.reshape((extend.shape[0], extend.shape[1], 12))

            pca_retract = PCA(n_components=12)
            retract_reduced = pca_retract.fit_transform(retract.reshape((-1, retract.shape[-1])))
            retract_reduced = retract_reduced.reshape((retract.shape[0], retract.shape[1], 12))

            trial_data_h.append(extend_reduced)
            trial_data_colors.append("pink")

            trial_data_h.append(retract_reduced)
            trial_data_colors.append("purple")
        
        else:

            extend = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            pca_extend = PCA(n_components=12)
            extend_reduced = pca_extend.fit_transform(extend.reshape((-1, extend.shape[-1])))
            extend_reduced = extend_reduced.reshape((extend.shape[0], extend.shape[1], 12))

            trial_data_h.append(extend_reduced)
            trial_data_colors.append("blue")

    # TODO play around with hyperparameters
    dsa = DSA(trial_data_h, n_delays=90, rank=150, verbose=True, score_method="euclidean", device="cpu")
    similarities = dsa.fit_score()

    dsa_data = {"similarities": similarities, "colors": trial_data_colors}

    with open(os.path.join(model_path, "dsa_similarity.txt"), 'wb') as f:
        pickle.dump(dsa_data, f)
    
    dsa_scatter(model_name)


def dsa_scatter(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    fig, ax = standard_2d_ax()

    dsa_data = load_pickle(os.path.join(model_path, "dsa_similarity.txt"))
    similarities = dsa_data["similarities"]
    colors = dsa_data["colors"]

    reduced = PCA(n_components=2).fit_transform(similarities)
    ax.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.75, s=250)
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_dsa_scatter"), eps=True)


def dsa_heatmap(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'

    dsa_data = load_pickle(os.path.join(model_path, "dsa_similarity.txt"))
    similarities = dsa_data["similarities"]

    # Reorder indices
    indices_extension = [0, 1, 2, 3, 4]
    indices_extension_long = [5, 7, 9, 11, 13]
    indices_retraction = [6, 8, 10, 12, 14]

    # full reordering index
    new_order = indices_extension + indices_extension_long + indices_retraction

    # reorder rows and columns at once
    re_similarity = similarities[np.ix_(new_order, new_order)]

    sns.heatmap(re_similarity, cmap="Purples")
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_dsa_similarity_vis"), eps=True)




def procrustes_similarity_matrix(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/dsa"

    options = {"batch_size": 32*4, "reach_conds": np.tile(np.arange(0, 32, 1), 4), "speed_cond": 5}

    trial_data_h = []
    trial_data_colors = []

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=True)

        if env == "DlyFullReach" or env == "DlyFullCircleClk" or env == "DlyFullCircleCClk" or env == "DlyFigure8" or env == "DlyFigure8Inv":

            halfway = int((trial_data["epoch_bounds"]["movement"][0] + trial_data["epoch_bounds"]["movement"][1]) / 2)
            extend = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]
            retract = trial_data["h"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]

            pca_extend = PCA(n_components=10)
            extend_reduced = pca_extend.fit_transform(extend.reshape((-1, extend.shape[-1])))
            extend_reduced = extend_reduced.reshape((extend.shape[0], extend.shape[1], 10))

            pca_retract = PCA(n_components=10)
            retract_reduced = pca_retract.fit_transform(retract.reshape((-1, retract.shape[-1])))
            retract_reduced = retract_reduced.reshape((retract.shape[0], retract.shape[1], 10))

            trial_data_h.append(extend_reduced.mean(axis=0))
            trial_data_colors.append("pink")

            trial_data_h.append(retract_reduced.mean(axis=0))
            trial_data_colors.append("purple")
        
        else:

            extend = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            pca_extend = PCA(n_components=10)
            extend_reduced = pca_extend.fit_transform(extend.reshape((-1, extend.shape[-1])))
            extend_reduced = extend_reduced.reshape((extend.shape[0], extend.shape[1], 10))

            trial_data_h.append(extend_reduced.mean(axis=0))
            trial_data_colors.append("blue")

    rows = []
    for task1 in trial_data_h:
        cols = []
        for task2 in trial_data_h:
            _, _, disparity = scipy.spatial.procrustes(task1, task2)
            cols.append(np.sqrt(disparity))
        rows.append(cols)
    similarities = np.array(rows)

    procrustes_data = {"similarities": similarities, "colors": trial_data_colors}

    with open(os.path.join(model_path, "procrustes_similarity.txt"), 'wb') as f:
        pickle.dump(procrustes_data, f)
    
    procrustes_scatter(model_name)


def procrustes_scatter(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    fig, ax = standard_2d_ax()

    procrustes_data = load_pickle(os.path.join(model_path, "procrustes_similarity.txt"))
    similarities = procrustes_data["similarities"]
    colors = procrustes_data["colors"]

    reduced = PCA(n_components=2).fit_transform(similarities)
    ax.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.75, s=250)
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_procrustes_scatter"), eps=True)


def procrustes_heatmap(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'

    procrustes_data = load_pickle(os.path.join(model_path, "procrustes_similarity.txt"))
    similarities = procrustes_data["similarities"]

    # Reorder indices
    indices_extension = [0, 1, 2, 3, 4]
    indices_extension_long = [5, 7, 9, 11, 13]
    indices_retraction = [6, 8, 10, 12, 14]

    # full reordering index
    new_order = indices_extension + indices_extension_long + indices_retraction

    # reorder rows and columns at once
    re_similarity = similarities[np.ix_(new_order, new_order)]

    sns.heatmap(re_similarity, cmap="Purples")
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_procrustes_similarity_vis"), eps=True)




def task_similarity_classification(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    def silhouette_scores(data, labels, num_clusters=3):
        silhouette_values = sklearn.metrics.silhouette_samples(data, labels)
        means_lst = []
        for label in range(num_clusters):
            means_lst.append(silhouette_values[labels == label].mean())
        return means_lst

    dsa_data = load_pickle(os.path.join(model_path, "dsa_similarity.txt"))
    dsa_similarities = dsa_data["similarities"]
    pca = PCA(n_components=2)
    dsa_similarities = pca.fit_transform(dsa_similarities)
    labels_dsa = np.ones(shape=(15,))
    for i, color in enumerate(dsa_data["colors"]):
        if color == "blue":
            labels_dsa[i] = 0
        elif color == "pink":
            labels_dsa[i] = 1
        elif color == "purple":
            labels_dsa[i] = 2
        
    means_dsa = silhouette_scores(dsa_similarities, labels_dsa, 3)

    procrustes_data = load_pickle(os.path.join(model_path, "procrustes_similarity.txt"))
    procrustes_similarities = procrustes_data["similarities"]
    pca = PCA(n_components=2)
    procrustes_similarities = pca.fit_transform(procrustes_similarities)
    labels_procrustes = np.ones(shape=(15,))
    for i, color in enumerate(procrustes_data["colors"]):
        if color == "blue":
            labels_procrustes[i] = 0
        elif color == "pink":
            labels_procrustes[i] = 1
        elif color == "purple":
            labels_procrustes[i] = 2

    means_pro = silhouette_scores(procrustes_similarities, labels_dsa, 3)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    x = ["Ext.", "Ext. (long)", "Ret."]
    plt.bar(x, means_dsa, color=["blue", "pink", "purple"], capsize=10, edgecolor="black", alpha=0.75)
    plt.xticks([])
    save_fig(os.path.join(exp_path, f"dsa_silhouette_bar"), eps=True)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    x = ["Ext.", "Ext. (long)", "Ret."]
    plt.bar(x, means_pro, color=["blue", "pink", "purple"], capsize=10, edgecolor="black", alpha=0.75)
    plt.xticks([])
    save_fig(os.path.join(exp_path, f"procrustes_silhouette_bar"), eps=True)





if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    # --------------------------------------------------------- COMPUTE INTERPOLATED FPS

    if args.experiment == "run_all_compute_interpolated_fps":
        run_all_compute_interpolated_fps(args.model_name)

    # --------------------------------------------------------- PLOT INTERPOLATED FPS

    elif args.experiment == "run_all_plot_interpolated_fps":
        run_all_plot_interpolated_fps(args.model_name)

    elif args.experiment == "trajectory_alignment_movement":
        trajectory_alignment_movement(args.model_name)

    elif args.experiment == "trajectory_alignment_delay":
        trajectory_alignment_delay(args.model_name)

    # Epoch pcs
    elif args.experiment == "stable_pcs_neural":
        stable_pcs_neural(args.model_name)
    elif args.experiment == "delay_pcs_neural":
        delay_pcs_neural(args.model_name)
    elif args.experiment == "movement_pcs_neural":
        movement_pcs_neural(args.model_name)

    elif args.experiment == "stable_pcs_motor":
        stable_pcs_motor(args.model_name)
    elif args.experiment == "delay_pcs_motor":
        delay_pcs_motor(args.model_name)
    elif args.experiment == "movement_pcs_motor":
        movement_pcs_motor(args.model_name)
    

    elif args.experiment == "neural_two_task_pcs_sinusoid_fullcircleclk":
        neural_two_task_pcs_sinusoid_fullcircleclk(args.model_name)
    elif args.experiment == "muscle_two_task_pcs_sinusoid_fullcircleclk":
        muscle_two_task_pcs_sinusoid_fullcircleclk(args.model_name)
    elif args.experiment == "neural_two_task_pcs_halfcircleclk_halfcirclecclk":
        neural_two_task_pcs_halfcircleclk_halfcirclecclk(args.model_name)
    elif args.experiment == "muscle_two_task_pcs_halfcircleclk_halfcirclecclk":
        muscle_two_task_pcs_halfcircleclk_halfcirclecclk(args.model_name)


    elif args.experiment == "task_vaf_ratio":
        task_vaf_ratio(args.model_name)

    elif args.experiment == "extension_similarity_trajectory":
        extension_similarity_trajectory(args.model_name)
    elif args.experiment == "retraction_similarity_trajectory":
        retraction_similarity_trajectory(args.model_name)

    elif args.experiment == "dsa_similarity_matrix":
        dsa_similarity_matrix(args.model_name)
    elif args.experiment == "dsa_scatter":
        dsa_scatter(args.model_name)
    elif args.experiment == "dsa_heatmap":
        dsa_heatmap(args.model_name)

    elif args.experiment == "procrustes_similarity_matrix":
        procrustes_similarity_matrix(args.model_name)
    elif args.experiment == "procrustes_scatter":
        procrustes_scatter(args.model_name)
    elif args.experiment == "procrustes_heatmap":
        procrustes_heatmap(args.model_name)
    
    elif args.experiment == "task_similarity_classification":
        task_similarity_classification(args.model_name)

    else:
        raise ValueError("Experiment not in this file")