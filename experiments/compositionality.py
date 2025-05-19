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
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from exp_utils import _test, env_dict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import seaborn as sns
import scipy
from utils import interpolate_trial
import pandas as pd

plt.rcParams.update({'font.size': 18})  # Sets default font size for all text

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


retraction_full_combinations = list(itertools.combinations(retraction_movements_full, 2))
retraction_tasks = retraction_full_combinations

subset_tasks = [
    ("DlyHalfCircleClk", "DlyFullCircleClk1"),
    ("DlySinusoid", "DlyFigure81"),
    ("DlyHalfReach", "DlyFullReach1"),
    ("DlyHalfCircleCClk", "DlyFullCircleCClk1"),
    ("DlySinusoidInv", "DlyFigure8Inv1"),
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
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(False)
    save_fig(os.path.join(exp_path, f"{system}_{epoch}_pcs"), eps=True)




def _two_task_epoch_pcs(model_name, task1, task2, epoch, system):

    exp_path = f"results/{model_name}/compositionality/pcs"
    pca_3d, env_hs = _get_pcs(model_name, batch_size=256, epoch=epoch, noise=True, system=system)

    colors = plt.cm.tab10(np.linspace(0, 1, len(env_hs))) 

    # Create a figure
    fig = plt.figure()
    fig.set_size_inches(4, 4)
    # Add a 3D subplot
    ax = fig.add_subplot(111, projection="3d")

    handles = []
    task_projections = []
    for i, (env_data, env) in enumerate(zip(env_hs, env_dict)):
        if env == task1 or env == task2:
            # Create patches with no border
            handles.append(mpatches.Patch(color=colors[i], label=env, edgecolor='none'))
            task_projections.append(env_data.squeeze())
            for h in env_data:
                # transform
                h_proj = pca_3d.transform(h)
                ax.scatter(h_proj[-1, 0], h_proj[-1, 1], h_proj[-1, 2], color=colors[i], s=25, alpha=0.25)
    
    centroid_A = task_projections[0].mean(axis=0)
    centroid_B = task_projections[1].mean(axis=0)
    centroid_line = centroid_A.unsqueeze(0) + torch.linspace(0, 1, 100).unsqueeze(1) * (centroid_B - centroid_A).unsqueeze(0)
    projected_line = pca_3d.transform(centroid_line)
    distance = torch.linalg.norm(centroid_A - centroid_B).item()
    
    ax.plot(projected_line[:, 0], projected_line[:, 1], projected_line[:, 2], linewidth=2, alpha=0.5, color="black")

    # Set labels for axes
    ax.text(projected_line[50, 0], projected_line[50, 1], projected_line[50, 2] + 0.5, f"Distance: {round(distance, 2)}", fontsize=10, color='black')
    save_fig(os.path.join(exp_path, f"{system}_{epoch}_{task1}_{task2}_pcs.png"))






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

def neural_epoch_pcs_halfcircleclk_figure8inv(model_name):
    _two_task_epoch_pcs(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "stable", "neural")
def neural_epoch_pcs_halfreach_fullreach(model_name):
    _two_task_epoch_pcs(model_name, "DlyHalfReach", "DlyFullReach", "stable", "neural")




def _interpolated_fps(model_name, task1, task2,  epoch, task1_period="all", task2_period="all", input_component=None):

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
        # Get inputs and x and h from desired timepoint
        inp1 = trial_data1["obs"][:, trial_data1["epoch_bounds"]["delay"][1]-1]
        inp2 = trial_data2["obs"][:, trial_data2["epoch_bounds"]["delay"][1]-1]

    elif epoch == "movement":

        middle_movement1 = int((trial_data1["epoch_bounds"]["movement"][1] + trial_data1["epoch_bounds"]["movement"][0]) / 2)
        middle_movement2 = int((trial_data2["epoch_bounds"]["movement"][1] + trial_data2["epoch_bounds"]["movement"][0]) / 2)

        if task1_period == "first":
            interpolation_point = int((middle_movement1 + trial_data1["epoch_bounds"]["movement"][0]) / 2)
            inp1 = trial_data1["obs"][:, interpolation_point]
        elif task1_period == "second":
            interpolation_point = int((middle_movement1 + trial_data1["epoch_bounds"]["movement"][1]) / 2)
            inp1 = trial_data1["obs"][:, interpolation_point]
        elif task1_period == "all":
            # This option only makes sense if using a half task
            inp1 = trial_data1["obs"][:, middle_movement1]

        if task2_period == "first":
            interpolation_point = int((middle_movement2 + trial_data2["epoch_bounds"]["movement"][0]) / 2)
            inp2 = trial_data2["obs"][:, interpolation_point]
        elif task2_period == "second":
            interpolation_point = int((middle_movement2 + trial_data2["epoch_bounds"]["movement"][1]) / 2)
            inp2 = trial_data2["obs"][:, interpolation_point]
        elif task2_period == "all":
            inp2 = trial_data2["obs"][:, middle_movement2]

    '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
    descriptions of available hyperparameters.'''
    fpf_hps = {
        'max_iters': 250,
        'lr_init': 1.,
        'outlier_distance_scale': 10.0,
        'verbose': False, 
        'super_verbose': False,
        'tol_unique': 1,
        'do_compute_jacobians': False}
        
    cond_fps_list = []
    for c, (cond1, cond2) in enumerate(zip(inp1, inp2)):

        """
            Could either initialize everything in need in the environment to be at the desired timepoint in the trial
            or run the trial again to the desired timpoint. The first is faster but for now more difficult so going with #2
        """

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

            """
                IMPORTANT: For now, treating the continuously interpolated input as a trajectory for the arm and RNN
                Feedback will not match the arm states during this trajectory.
            """

            # Currently using original h for initial states
            initial_states = fpf.sample_states(trial_data1["h"][c:c+1, timesteps:timesteps+1],
                n_inits=N_INITS,
                noise_scale=NOISE_SCALE)

            # Run the fixed point finder
            unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs=inp[None, :])

            # Add fixed points and their info to dict
            fps_list.append(
                {"fps": unique_fps, 
                "interp_point": i, 
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
def compute_interpolated_fps_halfreach_fullreach_delay_rule(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay", input_component="rule")
# Movement period with different input interpolations
def compute_interpolated_fps_halfreach_fullreach_movement_rule(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", task1_period="all", task2_period="first", input_component="rule")

#---------------------------------------------------------------- Extension Pair

# Delay period with different input interpolations
def compute_interpolated_fps_halfcircleclk_sinusoidinv_delay_rule(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "delay", input_component="rule")
# Movement period with different input interpolations
def compute_interpolated_fps_halfcircleclk_sinusoidinv_movement_rule(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "movement", task1_period="all", task2_period="all", input_component="rule")

#---------------------------------------------------------------- Retraction Pair

# Delay period with different input interpolations
def compute_interpolated_fps_fullcircleclk_figure8_delay_rule(model_name):
    _interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "delay", input_component="rule")
# Movement period with different input interpolations
def compute_interpolated_fps_fullcircleclk_figure8_movement_rule(model_name):
    _interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "movement", task1_period="second", task2_period="second", input_component="rule")

#---------------------------------------------------------------- Extension-Retraction Pair

# Delay period with different input interpolations
def compute_interpolated_fps_sinusoid_fullreach_delay_rule(model_name):
    _interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "delay", input_component="rule")
# Movement period with different input interpolations
def compute_interpolated_fps_sinusoid_fullreach_movement_rule(model_name):
    _interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "movement", task1_period="all", task2_period="second", input_component="rule")



def run_all_compute_interpolated_fps(model_name):
    compute_interpolated_fps_halfreach_fullreach_delay_rule(model_name)
    compute_interpolated_fps_halfreach_fullreach_movement_rule(model_name)
    compute_interpolated_fps_halfcircleclk_sinusoidinv_delay_rule(model_name)
    compute_interpolated_fps_halfcircleclk_sinusoidinv_movement_rule(model_name)
    compute_interpolated_fps_fullcircleclk_figure8_delay_rule(model_name)
    compute_interpolated_fps_fullcircleclk_figure8_movement_rule(model_name)
    compute_interpolated_fps_sinusoid_fullreach_delay_rule(model_name)
    compute_interpolated_fps_sinusoid_fullreach_movement_rule(model_name)




def compute_input_switching(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"

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
            device=device
        )
    elif hp["network"] == "gru":
        policy = GRUPolicy(hp["inp_size"], hp["hid_size"], effector.n_muscles, batch_first=True)
    else:
        raise ValueError("Not a valid architecture")

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=torch.device('cpu'))
    policy.load_state_dict(checkpoint['agent_state_dict'])

    combination_labels = list(itertools.combinations(env_dict, 2))

    traj_dict = {}
    for combination in combination_labels:

        task1 = combination[0]
        task2 = combination[1]

        trial_data1 = _test(model_path, model_file, options, env=env_dict[task1])
        trial_data2 = _test(model_path, model_file, options, env=env_dict[task2])

        # Get inputs and x and h from desired timepoint
        trial_data2_feedback = trial_data2["obs"][..., 14:]
        trial_data1_rule = trial_data1["obs"][:, 0:1, :10].repeat(1, trial_data2["obs"].shape[1], 1)
        trial_data1_others = torch.cat([
            trial_data1["obs"][:, :75, 10:14],
            trial_data1["obs"][:, 75:76, 10:14].repeat(1, trial_data2["obs"].shape[1] - 75, 1)
        ], dim=1)

        trial_data1_feedback = trial_data1["obs"][..., 14:]
        trial_data2_rule = trial_data2["obs"][:, 0:1, :10].repeat(1, trial_data1["obs"].shape[1], 1)
        trial_data2_others = torch.cat([
            trial_data2["obs"][:, :75, 10:14],
            trial_data2["obs"][:, 75:76, 10:14].repeat(1, trial_data1["obs"].shape[1] - 75, 1)
        ], dim=1)

        task1_rule_task2_feedback = torch.cat([
            trial_data1_rule,
            trial_data1_others,
            trial_data2_feedback
        ], dim=-1)

        task1_feedback_task2_rule = torch.cat([
            trial_data2_rule,
            trial_data2_others,
            trial_data1_feedback
        ], dim=-1)

        task1_rule_task2_feedback_hs = []
        task1_feedback_task2_rule_hs = []

        x = torch.zeros(size=(1, hp["hid_size"]))
        h = torch.zeros(size=(1, hp["hid_size"]))

        for t in range(task1_rule_task2_feedback.shape[1]):
            with torch.no_grad():
                x, h, _ = policy(task1_rule_task2_feedback[:, t, :], x, h, noise=False)
            task1_rule_task2_feedback_hs.append(h)
        task1_rule_task2_feedback_hs = torch.stack(task1_rule_task2_feedback_hs, dim=1)

        # --------------------------------------

        x = torch.zeros(size=(1, hp["hid_size"]))
        h = torch.zeros(size=(1, hp["hid_size"]))

        for t in range(task1_feedback_task2_rule.shape[1]):
            with torch.no_grad():
                x, h, _ = policy(task1_feedback_task2_rule[:, t, :], x, h, noise=False)
            task1_feedback_task2_rule_hs.append(h)
        task1_feedback_task2_rule_hs = torch.stack(task1_feedback_task2_rule_hs, dim=1)

        traj_dict[combination] = {
            "original_h_task1": trial_data1["h"], 
            "original_h_task2": trial_data2["h"], 
            "task1_epoch_bounds": trial_data1["epoch_bounds"], 
            "task2_epoch_bounds": trial_data2["epoch_bounds"], 
            "task1_rule_task2_feedback": task1_rule_task2_feedback_hs,
            "task1_feedback_task2_rule": task1_feedback_task2_rule_hs
        }

    # Save all information of fps across tasks to pickle file
    save_name = f'input_switching'
    fname = os.path.join(model_path, save_name + '.pkl')
    print('input switching saved at {:s}'.format(fname))
    with open(fname, 'wb') as f:
        pickle.dump(traj_dict, f)





def plot_input_switching(model_name):
    
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    load_name = os.path.join(model_path, f"input_switching.pkl")
    exp_path = f"results/{model_name}/compositionality/alignment"

    # This dict will contain pairs of tasks with different inputs switched
    # structure is a dictionary where key is task pair
    # Inside each dict are the two switched inputs and the original traj for task 1
    input_switching_dict = load_pickle(load_name)

    options = {"batch_size": 16, "reach_conds": torch.arange(0, 32, 2), "delay_cond": 1, "speed_cond": 5}
    combination_labels = list(itertools.combinations(env_dict, 2))

    feedback_switched = {}
    rule_switched = {}
    for combination in input_switching_dict:

        start_task1 = input_switching_dict[combination]["task1_epoch_bounds"]["movement"][0]
        end_task1 = input_switching_dict[combination]["task1_epoch_bounds"]["movement"][1]

        start_task2 = input_switching_dict[combination]["task2_epoch_bounds"]["movement"][0]
        end_task2 = input_switching_dict[combination]["task2_epoch_bounds"]["movement"][1]

        feedback_switched[combination] = (
            input_switching_dict[combination]["task1_rule_task2_feedback"][:, start_task2:end_task2],
            input_switching_dict[combination]["original_h_task1"][:, start_task1:end_task1],
        )
        rule_switched[combination] = (
            input_switching_dict[combination]["task1_feedback_task2_rule"][:, start_task1:end_task1],
            input_switching_dict[combination]["original_h_task1"][:, start_task1:end_task1],
        )
    
    print("Computing Distances...")
    distances_feedback_switched = _gather_distances(feedback_switched, options["batch_size"])
    distances_rule_switched = _gather_distances(rule_switched, options["batch_size"])

    print("Computing Angles...")
    angles_feedback_switched = _gather_angles(feedback_switched, options["batch_size"])
    angles_rule_switched = _gather_angles(rule_switched, options["batch_size"])

    print("Computing Trajectory Disparity...")
    traj_dist_feedback_switched = _gather_trajectory_dist(feedback_switched, options["batch_size"])
    traj_dist_rule_switched = _gather_trajectory_dist(rule_switched, options["batch_size"])

    # ------------------------------------------------------------- Neural Distances

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.set_ylabel("Density")
    ax.set_xlabel("Neural Distance")
    ax.set_xlim(0, 15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _plot_task_subsets(combination_labels, distances_feedback_switched, "feedback switched")
    _plot_task_subsets(combination_labels, distances_rule_switched, "rule switched")
    save_fig(os.path.join(exp_path, "movement", "input_switched_distances"))

    # ------------------------------------------------------------- Neural Angles

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.set_ylabel("Density")
    ax.set_xlabel("Neural Angles")
    ax.set_xlim(0, 90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _plot_task_subsets(combination_labels, angles_feedback_switched, "feedback switched")
    _plot_task_subsets(combination_labels, angles_rule_switched, "rule switched")
    save_fig(os.path.join(exp_path, "movement", "input_switched_angles"))

    # ------------------------------------------------------------- Neural Disparity

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.set_ylabel("Density")
    ax.set_xlabel("Neural Disparity")
    ax.set_xlim(0, 0.65)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _plot_task_subsets(combination_labels, traj_dist_feedback_switched, "feedback switched")
    _plot_task_subsets(combination_labels, traj_dist_rule_switched, "rule switched")
    save_fig(os.path.join(exp_path, "movement", "input_switched_disparity"))






def _plot_interpolated_fps(model_name, task1, task2, epoch, task1_period="all", task2_period="all", input_component=None):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    load_name = os.path.join(model_path, f"interpolated_fps_{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}.pkl")
    exp_path = f"results/{model_name}/compositionality/interpolated_fps"

    fps = load_pickle(load_name)

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 16, "reach_conds": torch.arange(0, 32, 2), "delay_cond": 1, "speed_cond": 5}

    colors_alpha = plt.cm.magma(np.linspace(0, 1, 20)) 

    trial_data1 = _test(model_path, model_file, options, env=env_dict[task1])
    trial_data2 = _test(model_path, model_file, options, env=env_dict[task2])

    trial_data1_h_epoch = trial_data1["h"][:, trial_data1["epoch_bounds"][epoch][0]:trial_data1["epoch_bounds"][epoch][1]] 
    trial_data2_h_epoch = trial_data2["h"][:, trial_data2["epoch_bounds"][epoch][0]:trial_data2["epoch_bounds"][epoch][1]] 

    two_task_pca = PCA(n_components=2)
    two_task_pca.fit(trial_data1_h_epoch.reshape((-1, trial_data1_h_epoch.shape[-1])))

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')  # or projection='3d'

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
                ax.scatter(i/20, zstar[:, 0], zstar[:, 1], marker='.', alpha=0.75, color=colors_alpha[i], s=250)

    ax.grid(False)
    save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", f"{task1}_{task2}_{epoch}_{task1_period}_{task2_period}_{input_component}"), eps=True)

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')  # or projection='3d'

    colors_traj = plt.cm.inferno(np.linspace(0, 1, 10)) 

    middle_movement1 = int((trial_data1["epoch_bounds"]["movement"][1] + trial_data1["epoch_bounds"]["movement"][0]) / 2)
    middle_movement2 = int((trial_data2["epoch_bounds"]["movement"][1] + trial_data2["epoch_bounds"]["movement"][0]) / 2)

    if task1_period == "first":
        trial_data1_h_epoch = trial_data1["h"][:, trial_data1["epoch_bounds"]["movement"][0]:middle_movement1]
    elif task1_period == "second":
        trial_data1_h_epoch = trial_data1["h"][:, middle_movement1:trial_data1["epoch_bounds"]["movement"][1]]
    elif task1_period == "all":
        trial_data1_h_epoch = trial_data1["h"][:, trial_data1["epoch_bounds"]["movement"][0]:trial_data1["epoch_bounds"]["movement"][1]]

    if task2_period == "first":
        trial_data2_h_epoch = trial_data2["h"][:, trial_data2["epoch_bounds"]["movement"][0]:middle_movement2]
    elif task2_period == "second":
        trial_data2_h_epoch = trial_data2["h"][:, middle_movement2:trial_data2["epoch_bounds"]["movement"][1]]
    elif task2_period == "all":
        trial_data2_h_epoch = trial_data2["h"][:, trial_data2["epoch_bounds"]["movement"][0]:trial_data2["epoch_bounds"]["movement"][1]]

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

    min_value = torch.min(combined_tasks)

    for c, condition in enumerate(trial_data_1_projected):
        ax.plot(condition[:, 0], condition[:, 1], condition[:, 2], linewidth=4, color=colors_traj[7], alpha=0.75, zorder=10)
        ax.scatter(condition[0, 0], condition[0, 1], condition[0, 2], s=150, marker="^", color=colors_traj[7])
        ax.scatter(condition[-1, 0], condition[-1, 1], condition[-1, 2], s=150, marker="X", color=colors_traj[7])
    for c, condition in enumerate(trial_data_2_projected):
        ax.plot(condition[:, 0], condition[:, 1], condition[:, 2], linewidth=4, color="black", alpha=0.75, zorder=10)
        ax.scatter(condition[0, 0], condition[0, 1], condition[0, 2], s=150, marker="^", color="black", zorder=20)
        ax.scatter(condition[-1, 0], condition[-1, 1], condition[-1, 2], s=150, marker="X", color="black", zorder=20)

    # Remove everything but the plot line
    ax.set_axis_off()  # hides axes, ticks, labels, etc.
    save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", f"{task1}_{task2}_{epoch}_{input_component}_pca"), eps=True)
    




#---------------------------------------------------------------- Subset Pair

# Delay period with different input interpolations
def plot_interpolated_fps_halfreach_fullreach_delay_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay", input_component="rule")
# Movement period with different input interpolations
def plot_interpolated_fps_halfreach_fullreach_movement_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", task1_period="all", task2_period="first", input_component="rule")

#---------------------------------------------------------------- Extension Pair

# Delay period with different input interpolations
def plot_interpolated_fps_halfcircleclk_sinusoidinv_delay_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "delay", input_component="rule")
# Movement period with different input interpolations
def plot_interpolated_fps_halfcircleclk_sinusoidinv_movement_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "movement", task1_period="all", task2_period="all", input_component="rule")

#---------------------------------------------------------------- Retraction Pair

# Delay period with different input interpolations
def plot_interpolated_fps_fullcircleclk_figure8_delay_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "delay", input_component="rule")
# Movement period with different input interpolations
def plot_interpolated_fps_fullcircleclk_figure8_movement_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "movement", task1_period="second", task2_period="second", input_component="rule")

#---------------------------------------------------------------- Extension-Retraction Pair

# Delay period with different input interpolations
def plot_interpolated_fps_sinusoid_fullreach_delay_rule(model_name):
    _plot_interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "delay", input_component="rule")
# Movement period with different input interpolations
def plot_interpolated_fps_sinusoid_fullreach_movement_rule(model_name):
    _plot_interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "movement", task1_period="all", task2_period="second", input_component="rule")



def run_all_plot_interpolated_fps(model_name):
    plot_interpolated_fps_halfreach_fullreach_delay_rule(model_name)
    plot_interpolated_fps_halfreach_fullreach_movement_rule(model_name)
    plot_interpolated_fps_halfcircleclk_sinusoidinv_delay_rule(model_name)
    plot_interpolated_fps_halfcircleclk_sinusoidinv_movement_rule(model_name)
    plot_interpolated_fps_fullcircleclk_figure8_delay_rule(model_name)
    plot_interpolated_fps_fullcircleclk_figure8_movement_rule(model_name)
    plot_interpolated_fps_sinusoid_fullreach_delay_rule(model_name)
    plot_interpolated_fps_sinusoid_fullreach_movement_rule(model_name)






# TODO not sure if this is debugged but not using it rn
def _two_task_variance_explained(model_name, task1, task2, task1_period, task2_period):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/variance_explained"

    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 1, "speed_cond": 5}

    trial_data_1 = _test(model_path, model_file, options, env=env_dict[task1])
    trial_data_2 = _test(model_path, model_file, options, env=env_dict[task2])

    middle_movement1 = int((trial_data1["epoch_bounds"]["movement"][1] + trial_data1["epoch_bounds"]["movement"][0]) / 2)
    middle_movement2 = int((trial_data2["epoch_bounds"]["movement"][1] + trial_data2["epoch_bounds"]["movement"][0]) / 2)

    if task1_period == "first":
        task_data_1_movement = trial_data1["h"][:, trial_data1["epoch_bounds"]["movement"][0]:middle_movement1]
    elif task1_period == "second":
        task_data_1_movement = trial_data1["h"][:, middle_movement1:trial_data1["epoch_bounds"]["movement"][1]]
    elif task1_period == "all":
        task_data_1_movement = trial_data1["h"][:, trial_data1["epoch_bounds"]["movement"][0]:trial_data1["epoch_bounds"]["movement"][1]]

    if task2_period == "first":
        task_data_2_movement = trial_data2["h"][:, trial_data2["epoch_bounds"]["movement"][0]:middle_movement2]
    elif task2_period == "second":
        task_data_2_movement = trial_data2["h"][:, middle_movement2:trial_data2["epoch_bounds"]["movement"][1]]
    elif task2_period == "all":
        task_data_2_movement = trial_data2["h"][:, trial_data2["epoch_bounds"]["movement"][0]:trial_data2["epoch_bounds"]["movement"][1]]

    for (task_1_condition, task_2_condition) in zip(task_data_1_movement, task_data_2_movement):

        task1_h = task_1_condition
        task2_h = task_2_condition

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
            
        plt.rc('figure', figsize=(3, 6))
        plt.ylim([0, 1])
        plt.plot(variance_task_1, color="black", marker="o", alpha=0.5, label=task1, markersize=15)
        plt.plot(variance_task_2, color="purple", marker="o", alpha=0.5, label=task2, markersize=15)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, f"{task1}_{task2}"), eps=True)




def ve_halfreach_fullreach(model_name):
    _two_task_variance_explained(model_name, "DlyHalfReach", "DlyFullReach", "all", "first")
def ve_halfcircleclk_sinusoidinv(model_name):
    _two_task_variance_explained(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "all", "all")
def ve_fullcircleclk_figure8(model_name):
    _two_task_variance_explained(model_name, "DlyFullCircleClk", "DlyFigure8", "second", "second")




def _gather_distances(combinations, batch_size):

    dist = {}
    for combination in combinations:
        trial_data1 = combinations[combination][0]
        trial_data2 = combinations[combination][1]
        condition_list = []
        for c in range(batch_size):
            averaged_h1 = trial_data1[c]
            averaged_h2 = trial_data2[c]
            condition_dist = torch.linalg.norm(averaged_h1 - averaged_h2)
            condition_list.append(condition_dist.item())
        dist[combination] = condition_list

    return dist

def _gather_angles(combinations, batch_size):

    angles = {}
    for combination in combinations:
        trial_data1 = combinations[combination][0]
        trial_data2 = combinations[combination][1]
        condition_list = []
        for c in range(batch_size):
            averaged_h1 = trial_data1[c]
            averaged_h2 = trial_data2[c]
            condition_angle = torch.arccos(torch.diagonal(averaged_h1 @ averaged_h2.T) / (torch.linalg.norm(averaged_h1, dim=1) * torch.linalg.norm(averaged_h2, dim=1))).mean()
            condition_list.append(condition_angle.item())
        angles[combination] = condition_list

    return angles

def _gather_shapes(combinations, batch_size):

    traj_dists = {}
    for combination in combinations:
        trial_data1 = combinations[combination][0]
        trial_data2 = combinations[combination][1]
        condition_list = []
        for c in range(batch_size):
            # First thing to do is interpolate, then align
            #trial_1 = interpolate_trial(trial_data1[c], trial_data2[c].shape[0]) if trial_data1[c].shape[0] < trial_data2[c].shape[0] else trial_data1[c]
            #trial_2 = interpolate_trial(trial_data2[c], trial_data1[c].shape[0]) if trial_data1[c].shape[0] > trial_data2[c].shape[0] else trial_data2[c]
            _, _, disparity = scipy.spatial.procrustes(trial_data1[c], trial_data2[c])
            condition_list.append(disparity)
        traj_dists[combination] = condition_list

    return traj_dists

def _convert_motif_dict_to_list(target_dict, data):
    target_data = []
    for combination in target_dict:
        if combination in data:
            target_data.extend(data[combination])
        elif (combination[1], combination[0]) in data:
            target_data.extend(data[(combination[1], combination[0])])
    return target_data

def _plot_task_subsets(combinations, metric, combination_labels, combination_colors):

    combination_means = []
    combination_stds = []
    for combination in combinations:
        task_metric = _convert_motif_dict_to_list(combination, metric)
        combination_means.append(sum(task_metric) / len(task_metric))
        combination_stds.append(np.std(task_metric, ddof=1))
    plt.bar(combination_labels, combination_means, yerr=combination_stds, capsize=10, color=combination_colors, edgecolor='black')
    

def _trajectory_alignment(model_name, epoch):

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

        if epoch == "movement":

            if env == "DlyFullReach" or env == "DlyFullCircleClk" or env == "DlyFullCircleCClk" or env == "DlyFigure8" or env == "DlyFigure8Inv":

                halfway = int((trial_data["epoch_bounds"][epoch][0] + trial_data["epoch_bounds"][epoch][1]) / 2)

                trial_data_h[env+"1"] = trial_data["h"][:, trial_data["epoch_bounds"][epoch][0]:halfway]
                trial_data_muscle[env+"1"] = trial_data["muscle_acts"][:, trial_data["epoch_bounds"][epoch][0]:halfway]

                trial_data_h[env+"2"] = trial_data["h"][:, halfway:trial_data["epoch_bounds"][epoch][1]]
                trial_data_muscle[env+"2"] = trial_data["muscle_acts"][:, halfway:trial_data["epoch_bounds"][epoch][1]]
            
            else:

                trial_data_h[env] = trial_data["h"][:, trial_data["epoch_bounds"][epoch][0]:trial_data["epoch_bounds"][epoch][1]]
                trial_data_muscle[env] = trial_data["muscle_acts"][:, trial_data["epoch_bounds"][epoch][0]:trial_data["epoch_bounds"][epoch][1]]
        
        elif epoch == "delay":

            trial_data_h[env] = trial_data["h"][:, trial_data["epoch_bounds"][epoch][0]:trial_data["epoch_bounds"][epoch][1]]
            trial_data_muscle[env] = trial_data["muscle_acts"][:, trial_data["epoch_bounds"][epoch][0]:trial_data["epoch_bounds"][epoch][1]]

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
    distances_h = _gather_distances(combinations_h, options["batch_size"])
    distances_muscle = _gather_distances(combinations_muscle, options["batch_size"])

    print("Computing Angles...")
    angles_h = _gather_angles(combinations_h, options["batch_size"])
    angles_muscle = _gather_angles(combinations_muscle, options["batch_size"])

    print("Computing Shapes...")
    shapes_h = _gather_shapes(combinations_h, options["batch_size"])
    shapes_muscle = _gather_shapes(combinations_muscle, options["batch_size"])

    all_subsets = [
        extension_tasks, 
        retraction_tasks,
        subset_tasks,
        extension_retraction_tasks,
        combination_labels
    ]

    all_subset_labels = [
        "extension_tasks",
        "retraction_tasks",
        "subset_tasks",
        "extension_retraction_tasks",
        "all_tasks"
    ]

    all_subset_colors = [
        "blue",
        "pink",
        "purple",
        "orange",
        "grey"
    ]

    # ------------------------------------------------------------- ANGLE DISTS

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    angle_h_dist = _convert_motif_dict_to_list(combination_labels, angles_h)
    angle_muscle_dist = _convert_motif_dict_to_list(combination_labels, angles_muscle)
    sns.kdeplot(angle_h_dist, color="blue", fill=True, zorder=10, legend=False, linewidth=2, alpha=0.25)
    sns.kdeplot(angle_muscle_dist, color="purple", fill=True, zorder=10, legend=False, linewidth=2, alpha=0.25)
    plt.axvline(sum(angle_h_dist)/len(angle_h_dist), color="blue", linestyle="dashed", linewidth=2)
    plt.axvline(sum(angle_muscle_dist)/len(angle_muscle_dist), color="purple", linestyle="dashed", linewidth=2)
    plt.xlim([0, 1.5])
    save_fig(os.path.join(exp_path, epoch, "neural_muscle_angle_dists"), eps=True)

    # ------------------------------------------------------------- NEURAL DISTANCES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _plot_task_subsets(all_subsets, distances_h, all_subset_labels, all_subset_colors)
    plt.xticks([])
    save_fig(os.path.join(exp_path, epoch, "neural_distances"), eps=True)

    # ------------------------------------------------------------- MUSCLE DISTANCES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _plot_task_subsets(all_subsets, distances_muscle, all_subset_labels, all_subset_colors)
    plt.xticks([])
    save_fig(os.path.join(exp_path, epoch, "muscle_distances"), eps=True)

    # ------------------------------------------------------------- NEURAL ANGLES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _plot_task_subsets(all_subsets, angles_h, all_subset_labels, all_subset_colors)
    plt.ylim([0, 1.5])
    plt.xticks([])
    save_fig(os.path.join(exp_path, epoch, "neural_angles"), eps=True)

    # ------------------------------------------------------------- MUSCLE ANGLES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _plot_task_subsets(all_subsets, angles_muscle, all_subset_labels, all_subset_colors)
    plt.ylim([0, 1.5])
    plt.xticks([])
    save_fig(os.path.join(exp_path, epoch, "muscle_angles"), eps=True)

    # ------------------------------------------------------------- NEURAL SHAPES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _plot_task_subsets(all_subsets, shapes_h, all_subset_labels, all_subset_colors)
    plt.ylim([0, 1])
    plt.xticks([])
    save_fig(os.path.join(exp_path, epoch, "neural_shapes"), eps=True)

    # ------------------------------------------------------------- MUSCLE SHAPES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _plot_task_subsets(all_subsets, shapes_muscle, all_subset_labels, all_subset_colors)
    plt.ylim([0, 1])
    plt.xticks([])
    save_fig(os.path.join(exp_path, epoch, "muscle_shapes"), eps=True)

    all_distances_h = _convert_motif_dict_to_list(combination_labels, distances_h)
    all_distances_muscle = _convert_motif_dict_to_list(combination_labels, distances_muscle)

    all_angles_h = _convert_motif_dict_to_list(combination_labels, angles_h)
    all_angles_muscle = _convert_motif_dict_to_list(combination_labels, angles_muscle)

    all_shapes_h = _convert_motif_dict_to_list(combination_labels, shapes_h)
    all_shapes_muscle = _convert_motif_dict_to_list(combination_labels, shapes_muscle)

    # -------------------------------------- NEURAL AND MUSCLE SHAPE DISTRIBUTIONS

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    neural_shape_mean = sum(all_shapes_h) / len(all_shapes_h)
    muscle_shape_mean = sum(all_shapes_muscle) / len(all_shapes_muscle)

    sns.kdeplot(all_shapes_h, color="blue", fill=True, zorder=10, legend=False, linewidth=2, alpha=0.25)
    sns.kdeplot(all_shapes_muscle, color="purple", fill=True, zorder=10, legend=False, linewidth=2, alpha=0.25)
    plt.axvline(neural_shape_mean, color="blue", linestyle="dashed", linewidth=2)
    plt.axvline(muscle_shape_mean, color="purple", linestyle="dashed", linewidth=2)
    plt.xlim([0, 1])
    save_fig(os.path.join(exp_path, epoch, "neural_muscle_shape_dists"), eps=True)

    # -------------------------------------- NEURAL DISTANCE VS NEURAL ANGLE

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    diff_regress = scipy.stats.linregress(all_distances_h, all_angles_h)
    cc_diff = diff_regress.rvalue
    print("CC neural diff to neural angle: ", cc_diff)
    x = np.linspace(0, max(all_distances_h))
    ax.plot(x, diff_regress.slope * x + diff_regress.intercept, linestyle="dashed", color="grey")
    ax.scatter(all_distances_h, all_angles_h, s=100, alpha=0.5, color="purple")
    save_fig(os.path.join(exp_path, epoch, "neural_diff_vs_neural_angle"), eps=True)

    # -------------------------------------- MUSCLE DISTANCE VS MUSCLE ANGLE

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    diff_regress = scipy.stats.linregress(all_distances_muscle, all_angles_muscle)
    cc_diff = diff_regress.rvalue
    print("CC muscle diff to muscle angle: ", cc_diff)
    x = np.linspace(0, max(all_distances_muscle))
    ax.plot(x, diff_regress.slope * x + diff_regress.intercept, linestyle="dashed", color="grey")
    ax.scatter(all_distances_muscle, all_angles_muscle, s=100, alpha=0.5, color="purple")
    save_fig(os.path.join(exp_path, epoch, "muscle_diff_vs_muscle_angle"), eps=True)

    # -------------------------------------- NEURAL ANGLE VS MUSCLE ANGLE

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    diff_regress = scipy.stats.linregress(all_angles_h, all_angles_muscle)
    cc_diff = diff_regress.rvalue
    print("CC neural angle to muscle angle: ", cc_diff)
    x = np.linspace(0, max(all_angles_h))
    ax.plot(x, diff_regress.slope * x + diff_regress.intercept, color="black")
    ax.scatter(all_angles_h, all_angles_muscle, s=50, alpha=0.25, color="grey")
    save_fig(os.path.join(exp_path, epoch, "neural_angle_vs_muscle_angle"), eps=True)

    # -------------------------------------- NEURAL ANGLE VS MUSCLE SHAPE

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    diff_regress = scipy.stats.linregress(all_angles_h, all_shapes_muscle)
    cc_diff = diff_regress.rvalue
    print("CC neural angle to muscle shape: ", cc_diff)
    x = np.linspace(0, max(all_angles_h))
    ax.plot(x, diff_regress.slope * x + diff_regress.intercept, color="black")
    ax.scatter(all_angles_h, all_shapes_muscle, s=50, alpha=0.25, color="grey")
    save_fig(os.path.join(exp_path, epoch, "neural_angle_vs_muscle_shape"), eps=True)

    # -------------------------------------- MUSCLE SHAPE VS NEURAL SHAPE

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    diff_regress = scipy.stats.linregress(all_shapes_muscle, all_shapes_h)
    cc_diff = diff_regress.rvalue
    print("CC muscle shape to neural shape: ", cc_diff)
    x = np.linspace(0, max(all_shapes_muscle))
    ax.plot(x, diff_regress.slope * x + diff_regress.intercept, color="black")
    ax.scatter(all_shapes_muscle, all_shapes_h, s=50, alpha=0.25, color="grey")
    save_fig(os.path.join(exp_path, epoch, "muscle_shape_vs_neural_shape"), eps=True)

    # -------------------------------------- NEURAL SHAPE VS MUSCLE ANGLE

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    diff_regress = scipy.stats.linregress(all_shapes_h, all_angles_muscle)
    cc_diff = diff_regress.rvalue
    print("CC neural shape to muscle angle: ", cc_diff)
    x = np.linspace(0, max(all_shapes_h))
    ax.plot(x, diff_regress.slope * x + diff_regress.intercept, color="black")
    ax.scatter(all_shapes_h, all_angles_muscle, s=50, alpha=0.25, color="grey")
    save_fig(os.path.join(exp_path, epoch, "neural_shape_vs_muscle_angle"), eps=True)




def trajectory_alignment_movement(model_name):
    _trajectory_alignment(model_name, "movement")
def trajectory_alignment_delay(model_name):
    _trajectory_alignment(model_name, "delay")




def _project_pca_combination(data):

    # Data is currently a tuple containing two tensors for the hs of two tasks
    # across all conditions

    pc_dict = {}

    for (task1_cond, task2_cond) in zip(data[0], data[1]):

        task1_pca = PCA(n_components=5)
        task2_pca = PCA(n_components=5)

        task1_pca.fit(task1_cond)
        task2_pca.fit(task2_cond)

        task1_cond_var = task1_cond.var(dim=0).sum()
        task2_cond_var = task2_cond.var(dim=0).sum()

        for pc in range(5):

            pc_dict[pc] = []
            # Now get variance explained projecting between tasks
            task2_pca_comp = task2_pca.components_[pc]
            var_explained_task1 = (task1_cond @ task2_pca_comp).var(dim=0) / task2_cond_var
            pc_dict[pc].append(var_explained_task1)

            task1_pca_comp = task1_pca.components_[pc]
            var_explained_task2 = (task2_cond @ task1_pca_comp).var(dim=0) / task1_cond_var
            pc_dict[pc].append(var_explained_task2)
    
    return pc_dict


def _get_ve_combination(combinations, data):

    # Initialize the full pc dict
    all_pc_dict = {}
    for i in range(1, 6):
        all_pc_dict[i] = []

    for combination in combinations:
        if combination in data:
            # This should return a list of the variance explained for each pc
            tmp_pc_dict = _project_pca_combination(data[combination])
        elif (combination[1], combination[0]) in data:
            tmp_pc_dict = _project_pca_combination(data[(combination[1]), combination[0]])
        
        # Go through all pcs and add to full pc dict
        for pc in tmp_pc_dict:
            all_pc_dict[pc+1].extend(tmp_pc_dict[pc])
    
    return all_pc_dict


def mode_shift(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/mode_shift"

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
        
        else:
            trial_data_h[env] = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
        

    # Get all unique pairs of unit activity across tasks
    combination_labels = list(itertools.combinations(trial_data_h, 2))
    for combination_label in combination_labels:
        combinations_h[combination_label] = (
            trial_data_h[combination_label[0]],
            trial_data_h[combination_label[1]]
        )

    all_subsets = {
        "inverse_tasks": inverse_tasks, 
        "subset_forward_tasks": subset_forward_tasks, 
        "subset_backward_tasks": subset_backward_tasks, 
    }

    all_subsets_colors = {
        "inverse_tasks": "blue",
        "subset_forward_tasks": "purple",
        "subset_backward_tasks": "orange"
    }

    subset_pc_dict = {}
    for subset in all_subsets:
        subset_pc_dict[subset] = _get_ve_combination(all_subsets[subset], combinations_h)
    
    data = []
    for cond in subset_pc_dict:
        for group in subset_pc_dict[cond]:
            for v in subset_pc_dict[cond][group]:
                data.append({'Group': f"PC {group}", 'Condition': cond, 'Value': v.item()})

    df = pd.DataFrame(data)

    # Aggregate for bar plot (e.g., mean value per group/condition)
    means = bar_df = df.groupby(['Group', 'Condition'], observed=True)["Value"].mean().reset_index()

    # Set the custom order for 'Group' and 'Condition'
    group_order = ['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5']
    condition_order = ['subset_forward_tasks', 'inverse_tasks', 'subset_backward_tasks']

    # Reorder the 'Group' and 'Condition' columns as categorical with a specified order
    means['Group'] = pd.Categorical(means['Group'], categories=group_order, ordered=True)
    means['Condition'] = pd.Categorical(means['Condition'], categories=condition_order, ordered=True)


    # ------------------------------------------------------------- NEURAL DISTANCES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    sns.reset_defaults()

    ax = sns.barplot(data=bar_df, x='Group', y='Value', hue='Condition',
        palette=all_subsets_colors, dodge=True)
    # Remove everything
    #ax.set(xlabel=None, ylabel=None)
    #ax.set_xticks([])
    #ax.get_legend().remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, "pc_shift"), eps=True)





def _get_pa_combination(combination_labels, data):

    # Initialize the full pc dict
    all_pc_list_means = []
    all_pc_list_stds = []

    data_tuples = []
    for combination in combination_labels:
        if combination in data:
            # This should return a list of the variance explained for each pc
            data_tuples.append(data[combination])
        elif (combination[1], combination[0]) in data:
            data_tuples.append(data[(combination[1]), combination[0]])

    for pc in range(1, 13):

        # This should return a list of the variance explained for each pc
        tmp_angles_dict = principal_angles(data_tuples, combination_labels, mode="h", num_comps=pc, control=False)

        combination_list = []
        for combination in tmp_angles_dict:
            # this should go through each combination and get the lowest angle
            combination_list.append(tmp_angles_dict[combination][0])
        all_pc_list_means.append(np.array(combination_list).mean())
        all_pc_list_stds.append(np.array(combination_list).std())
    
    return np.array(all_pc_list_means), np.array(all_pc_list_stds)


def task_principle_angles(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/task_pc_angles"

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
        
        else:
            trial_data_h[env] = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
        

    # Get all unique pairs of unit activity across tasks
    combination_labels = list(itertools.combinations(trial_data_h, 2))
    for combination_label in combination_labels:
        combinations_h[combination_label] = (
            trial_data_h[combination_label[0]],
            trial_data_h[combination_label[1]]
        )

    all_subsets = {
        "inverse_tasks": inverse_tasks, 
        "subset_forward_tasks": subset_forward_tasks, 
        "subset_backward_tasks": subset_backward_tasks, 
    }

    all_subsets_colors = {
        "inverse_tasks": "blue",
        "subset_forward_tasks": "purple",
        "subset_backward_tasks": "orange"
    }

    subset_pc_dict_means = {}
    subset_pc_dict_stds = {}
    for subset in all_subsets:
        subset_pc_dict_means[subset], subset_pc_dict_stds[subset] = _get_pa_combination(all_subsets[subset], combinations_h)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'

    x = np.arange(1, 13)
    for subset in subset_pc_dict_means:
        ax.plot(x, subset_pc_dict_means[subset], linewidth=4, marker='o', markersize=15, alpha=0.5, color=all_subsets_colors[subset])
        ax.fill_between(
            x, 
            subset_pc_dict_means[subset] - subset_pc_dict_stds[subset], 
            subset_pc_dict_means[subset] + subset_pc_dict_stds[subset], 
            color=all_subsets_colors[subset], 
            alpha=0.25
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, "pc_angles"), eps=True)





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

    comp_range = 51

    # Plotting vaf for different number of pc components neural

    subset_pc_dict_means = {}
    subset_pc_dict_stds = {}
    for subset in all_subsets:
        subset_pc_dict_means[subset], subset_pc_dict_stds[subset] = _get_vaf_combination(all_subsets[subset], combinations_h, "h", comp_range)
    all_task_pc_means, all_task_pc_stds = _get_vaf_combination(combination_labels, combinations_h, "h", comp_range)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'

    x = np.arange(1, comp_range)
    for subset in subset_pc_dict_means:
        ax.plot(x, subset_pc_dict_means[subset], linewidth=4, marker='o', markersize=15, alpha=0.5, color=all_subsets_colors[subset])
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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, "vaf_ratio_neural"), eps=True)

    # Plotting vaf for different number of pc components muscle

    subset_pc_dict_means = {}
    subset_pc_dict_stds = {}
    for subset in all_subsets:
        subset_pc_dict_means[subset], subset_pc_dict_stds[subset] = _get_vaf_combination(all_subsets[subset], combinations_muscle, "muscle_acts", comp_range)
    all_task_pc_means, all_task_pc_stds = _get_vaf_combination(combination_labels, combinations_muscle, "muscle_acts", comp_range)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'

    x = np.arange(1, comp_range)
    for subset in subset_pc_dict_means:
        ax.plot(x, subset_pc_dict_means[subset], linewidth=4, marker='o', markersize=15, alpha=0.5, color=all_subsets_colors[subset])
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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, "vaf_ratio_muscle"), eps=True)


if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    # Variance explained between task subspaces
    if args.experiment == "ve_halfcircleclk_figure8inv":
        ve_halfcircleclk_figure8inv(args.model_name)
    elif args.experiment == "ve_halfreach_fullreach":
        ve_halfreach_fullreach(args.model_name)

    # --------------------------------------------------------- COMPUTE INTERPOLATED FPS

    elif args.experiment == "run_all_compute_interpolated_fps":
        run_all_compute_interpolated_fps(args.model_name)

    # --------------------------------------------------------- PLOT INTERPOLATED FPS

    elif args.experiment == "plot_interpolated_fps_halfreach_fullreach_delay":
        plot_interpolated_fps_halfreach_fullreach_delay(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfreach_fullreach_delay_rule":
        plot_interpolated_fps_halfreach_fullreach_delay_rule(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfreach_fullreach_delay_proprioception":
        plot_interpolated_fps_halfreach_fullreach_delay_proprioception(args.model_name)

    elif args.experiment == "plot_interpolated_fps_halfreach_fullreach_movement":
        plot_interpolated_fps_halfreach_fullreach_movement(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfreach_fullreach_movement_rule":
        plot_interpolated_fps_halfreach_fullreach_movement_rule(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfreach_fullreach_movement_proprioception":
        plot_interpolated_fps_halfreach_fullreach_movement_proprioception(args.model_name)

    elif args.experiment == "plot_interpolated_fps_halfcircleclk_figure8inv_delay":
        plot_interpolated_fps_halfcircleclk_figure8inv_delay(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfcircleclk_figure8inv_delay_rule":
        plot_interpolated_fps_halfcircleclk_figure8inv_delay_rule(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfcircleclk_figure8inv_delay_proprioception":
        plot_interpolated_fps_halfcircleclk_figure8inv_delay_proprioception(args.model_name)

    elif args.experiment == "plot_interpolated_fps_halfcircleclk_figure8inv_movement":
        plot_interpolated_fps_halfcircleclk_figure8inv_movement(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfcircleclk_figure8inv_movement_rule":
        plot_interpolated_fps_halfcircleclk_figure8inv_movement_rule(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfcircleclk_figure8inv_movement_proprioception":
        plot_interpolated_fps_halfcircleclk_figure8inv_movement_proprioception(args.model_name)

    elif args.experiment == "run_all_plot_interpolated_fps":
        run_all_plot_interpolated_fps(args.model_name)
    elif args.experiment == "compute_input_switching":
        compute_input_switching(args.model_name)
    elif args.experiment == "plot_input_switching":
        plot_input_switching(args.model_name)

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
    elif args.experiment == "neural_epoch_pcs_halfcircleclk_figure8inv":
        neural_epoch_pcs_halfcircleclk_figure8inv(args.model_name)
    elif args.experiment == "neural_epoch_pcs_halfreach_fullreach":
        neural_epoch_pcs_halfreach_fullreach(args.model_name)

    elif args.experiment == "mode_shift":
        mode_shift(args.model_name)
    elif args.experiment == "task_principal_angles":
        task_principle_angles(args.model_name)
    elif args.experiment == "task_vaf_ratio":
        task_vaf_ratio(args.model_name)

    else:
        raise ValueError("Experiment not in this file")