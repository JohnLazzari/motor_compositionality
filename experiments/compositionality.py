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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import seaborn as sns
import scipy
from utils import interpolate_trial

plt.rcParams.update({'font.size': 18})  # Sets default font size for all text


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




def _interpolated_fps(model_name, task1, task2, epoch, input_component=None):

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
        inp1 = trial_data1["obs"][:, middle_movement1]
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
    save_name = f'interpolated_fps_{task1}_{task2}_{epoch}_{input_component}'
    fname = os.path.join(model_path, save_name + '.pkl')
    print('interpolated fps saved at {:s}'.format(fname))
    with open(fname, 'wb') as f:
        pickle.dump(cond_fps_list, f)





# Similar tasks

# Delay period with different input interpolations
def compute_interpolated_fps_halfreach_fullreach_delay(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay")
def compute_interpolated_fps_halfreach_fullreach_delay_rule(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay", "rule")
def compute_interpolated_fps_halfreach_fullreach_delay_proprioception(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay", "proprioception")

# Movement period with different input interpolations
def compute_interpolated_fps_halfreach_fullreach_movement(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement")
def compute_interpolated_fps_halfreach_fullreach_movement_rule(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", "rule")
def compute_interpolated_fps_halfreach_fullreach_movement_proprioception(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", "proprioception")

# Dissimilar tasks

# Delay period with different input interpolations
def compute_interpolated_fps_halfcircleclk_figure8inv_delay(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay")
def compute_interpolated_fps_halfcircleclk_figure8inv_delay_rule(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay", "rule")
def compute_interpolated_fps_halfcircleclk_figure8inv_delay_proprioception(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay", "proprioception")

# Movement period with different input interpolations
def compute_interpolated_fps_halfcircleclk_figure8inv_movement(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement")
def compute_interpolated_fps_halfcircleclk_figure8inv_movement_rule(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement", "rule")
def compute_interpolated_fps_halfcircleclk_figure8inv_movement_proprioception(model_name):
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement", "proprioception")




def run_all_compute_interpolated_fps(model_name):
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay", "rule")
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay", "proprioception")
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", "rule")
    _interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", "proprioception")
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay", "rule")
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay", "proprioception")
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement", "rule")
    _interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement", "proprioception")




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






def _plot_interpolated_fps(model_name, task1, task2, epoch, input_component=None):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    load_name = os.path.join(model_path, f"interpolated_fps_{task1}_{task2}_{epoch}_{input_component}.pkl")
    exp_path = f"results/{model_name}/compositionality/interpolated_fps"

    fps = load_pickle(load_name)

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 16, "reach_conds": torch.arange(0, 32, 2), "delay_cond": 1, "speed_cond": 5}

    colors_alpha = plt.cm.magma(np.linspace(0, 1, 20)) 
    colors_conditions = plt.cm.inferno(np.linspace(0, 1, 20)) 

    trial_data1 = _test(model_path, model_file, options, env=env_dict[task1])
    trial_data2 = _test(model_path, model_file, options, env=env_dict[task2])

    trial_data1_h_epoch = trial_data1["h"][:, trial_data1["epoch_bounds"][epoch][0]:trial_data1["epoch_bounds"][epoch][1]] 
    trial_data2_h_epoch = trial_data2["h"][:, trial_data2["epoch_bounds"][epoch][0]:trial_data2["epoch_bounds"][epoch][1]] 
    combined_tasks = torch.cat([trial_data1_h_epoch, trial_data2_h_epoch], dim=1)

    task_hs = []
    for env in env_dict:
        trial_data = _test(model_path, model_file, options, env=env_dict[env])
        task_hs.append(trial_data["h"][:, trial_data["epoch_bounds"][epoch][0]:trial_data["epoch_bounds"][epoch][1]])
    all_task_hs = torch.cat(task_hs, dim=1)

    two_task_pca = PCA(n_components=2)
    two_task_pca.fit(combined_tasks.reshape((-1, combined_tasks.shape[-1])))

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
                ax.scatter(i/20, zstar[:, 0], zstar[:, 1], marker='.', alpha=0.75, color=colors_alpha[i], s=200)

    ax.grid(False)
    save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", f"{task1}_{task2}_{epoch}_{input_component}"), eps=True)

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')  # or projection='3d'

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
        ax.plot(condition[:, 0], condition[:, 1], condition[:, 2], linewidth=4, color=colors_conditions[c], alpha=0.75)
        ax.scatter(condition[0, 0], condition[0, 1], condition[0, 2], s=150, marker="^", color=colors_conditions[c])
        ax.scatter(condition[-1, 0], condition[-1, 1], condition[-1, 2], s=150, marker="X", color=colors_conditions[c])
    for c, condition in enumerate(trial_data_2_projected):
        ax.plot(condition[:, 0], condition[:, 1], condition[:, 2], linewidth=4, color=colors_conditions[c], alpha=0.25)
        ax.scatter(condition[0, 0], condition[0, 1], condition[0, 2], s=150, marker="^", color=colors_conditions[c])
        ax.scatter(condition[-1, 0], condition[-1, 1], condition[-1, 2], s=150, marker="X", color=colors_conditions[c])

    ax.grid(False)
    save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", f"{task1}_{task2}_{epoch}_{input_component}_pca"), eps=True)
    



# Similar Tasks
def plot_interpolated_fps_halfreach_fullreach_delay(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay")
def plot_interpolated_fps_halfreach_fullreach_delay_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay", "rule")
def plot_interpolated_fps_halfreach_fullreach_delay_proprioception(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay", "proprioception")

def plot_interpolated_fps_halfreach_fullreach_movement(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement")
def plot_interpolated_fps_halfreach_fullreach_movement_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", "rule")
def plot_interpolated_fps_halfreach_fullreach_movement_proprioception(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", "proprioception")

# Dissimilar Tasks
def plot_interpolated_fps_halfcircleclk_figure8inv_delay(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay")
def plot_interpolated_fps_halfcircleclk_figure8inv_delay_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay", "rule")
def plot_interpolated_fps_halfcircleclk_figure8inv_delay_proprioception(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay", "proprioception")

def plot_interpolated_fps_halfcircleclk_figure8inv_movement(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement")
def plot_interpolated_fps_halfcircleclk_figure8inv_movement_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement", "rule")
def plot_interpolated_fps_halfcircleclk_figure8inv_movement_proprioception(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement", "proprioception")




def run_all_plot_interpolated_fps(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay", "rule")
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay", "proprioception")
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", "rule")
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "movement", "proprioception")
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay", "rule")
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay", "proprioception")
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement", "rule")
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement", "proprioception")



def _manifold_traversal(model_name, task1, task2, epoch, system, input_component=None):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    load_name = os.path.join(model_path, f"interpolated_fps_{task1}_{task2}_{epoch}_{input_component}.pkl")
    exp_path = f"results/{model_name}/compositionality/manifold_traversal"

    fps = load_pickle(load_name)


    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "delay_cond": 1, "speed_cond": 5}

    trial_data1 = _test(model_path, model_file, options, env=env_dict[task1])
    trial_data2 = _test(model_path, model_file, options, env=env_dict[task2])

    # cond is a list containing the fps for each interpolated rule input for a given condition
    condition_task_2_mags = []
    condition_task_1_mags = []
    for c, cond in enumerate(fps):
        
        if system == "neural":
            interpolated_acts = np.concatenate([unique_fps["h_state"] for unique_fps in cond])
        elif system == "muscle":
            interpolated_acts = np.concatenate([unique_fps["muscle_acts"] for unique_fps in cond])

        interpolated_kinematics = np.concatenate([unique_fps["fingertip"] for unique_fps in cond])


        task2_pca = PCA()
        task1_pca = PCA()
        
        if system == "neural":
            # Currently this is specifying the epoch but I didn't do that for principal angles
            task2_pca.fit(trial_data2["h"][c, trial_data2["epoch_bounds"][epoch][0]:trial_data2["epoch_bounds"][epoch][1]])
            task1_pca.fit(trial_data1["h"][c, trial_data1["epoch_bounds"][epoch][0]:trial_data1["epoch_bounds"][epoch][1]])
            n_components = 12
        elif system == "muscle":
            task2_pca.fit(trial_data2["muscle_acts"][c, trial_data2["epoch_bounds"][epoch][0]:trial_data2["epoch_bounds"][epoch][1]])
            task1_pca.fit(trial_data1["muscle_acts"][c, trial_data1["epoch_bounds"][epoch][0]:trial_data1["epoch_bounds"][epoch][1]])
            n_components = 6

        interp_h1_projected = task1_pca.transform(interpolated_acts)
        interp_h2_projected = task2_pca.transform(interpolated_acts)

        # Going through each component
        interp_h1_mags = np.abs(interp_h1_projected[:, :n_components]) - np.abs(interp_h1_projected[0, :n_components])
        interp_h2_mags = np.abs(interp_h2_projected[:, :n_components]) - np.abs(interp_h2_projected[0, :n_components])

        # Each item is [n_interpolated_points, n_components]
        condition_task_2_mags.append(interp_h2_mags)
        condition_task_1_mags.append(interp_h1_mags)
    
    # Convert to numpy array
    condition_task_2_mags = np.stack(condition_task_2_mags)
    condition_task_1_mags = np.stack(condition_task_1_mags)

    # Take the mean and std along the batch dimension
    mean_task_2_mags = condition_task_2_mags.mean(axis=0)
    std_task_2_mags = condition_task_2_mags.std(axis=0)

    mean_task_1_mags = condition_task_1_mags.mean(axis=0)
    std_task_1_mags = condition_task_1_mags.std(axis=0)

    cov_matrix = mean_task_1_mags.T @ mean_task_2_mags

    colors = plt.cm.plasma(np.linspace(0, 1, n_components)) 

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 6))
    ax = fig.add_subplot(111)  # or projection='3d'

    x = np.linspace(0, 20, 20)

    save_name = f"manifold_traversal.png"

    if system == "neural":
        ax.set_ylim([-4, 4])
    elif system == "muscle":
        ax.set_ylim([-.4, .4])
    for comp in range(mean_task_2_mags.shape[1]):
        ax.plot(mean_task_2_mags[:, comp], linewidth=4, alpha=0.5, marker="o", markersize=10, label=f"PC {comp+1}", color=colors[comp])
        ax.errorbar(x, mean_task_2_mags[:, comp], yerr=std_task_2_mags[:, comp], fmt="none", capsize=5, color=colors[comp], alpha=0.25)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", system, save_name))
        
    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    save_name = f"intertask_cov.png"

    im = ax.imshow(cov_matrix, aspect="auto")
    fig.colorbar(im, ax=ax)
    save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", system, save_name))




# ------------------------------------------- NEURAL MANIFOLD TRAVERSAL

# Similar Tasks
def neural_manifold_traversal_halfreach_fullreach_delay(model_name):
    _manifold_traversal(model_name, "DlyHalfReach", "DlyFullReach", "delay", "neural")
def neural_manifold_traversal_halfreach_fullreach_delay_rule(model_name):
    _manifold_traversal(model_name, "DlyHalfReach", "DlyFullReach", "delay", "neural", "rule")
def neural_manifold_traversal_halfreach_fullreach_delay_proprioception(model_name):
    _manifold_traversal(model_name, "DlyHalfReach", "DlyFullReach", "delay", "neural", "proprioception")

def neural_manifold_traversal_halfreach_fullreach_movement(model_name):
    _manifold_traversal(model_name, "DlyHalfReach", "DlyFullReach", "neural", "movement")
def neural_manifold_traversal_halfreach_fullreach_movement_rule(model_name):
    _manifold_traversal(model_name, "DlyHalfReach", "DlyFullReach", "movement", "neural", "rule")
def neural_manifold_traversal_halfreach_fullreach_movement_proprioception(model_name):
    _manifold_traversal(model_name, "DlyHalfReach", "DlyFullReach", "movement", "neural", "proprioception")

# Dissimilar Tasks
def neural_manifold_traversal_halfcircleclk_figure8inv_delay(model_name):
    _manifold_traversal(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "neural", "delay")
def neural_manifold_traversal_halfcircleclk_figure8inv_delay_rule(model_name):
    _manifold_traversal(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay", "neural", "rule")
def neural_manifold_traversal_halfcircleclk_figure8inv_delay_proprioception(model_name):
    _manifold_traversal(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay", "neural", "proprioception")

def neural_manifold_traversal_halfcircleclk_figure8inv_movement(model_name):
    _manifold_traversal(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "neural", "movement")
def neural_manifold_traversal_halfcircleclk_figure8inv_movement_rule(model_name):
    _manifold_traversal(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement", "neural", "rule")
def neural_manifold_traversal_halfcircleclk_figure8inv_movement_proprioception(model_name):
    _manifold_traversal(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement", "neural", "proprioception")


def run_all_manifold_traversal(model_name):
    _manifold_traversal(model_name, "DlyHalfReach", "DlyFullReach", "delay", "rule")
    _manifold_traversal(model_name, "DlyHalfReach", "DlyFullReach", "delay", "proprioception")
    _manifold_traversal(model_name, "DlyHalfReach", "DlyFullReach", "movement", "rule")
    _manifold_traversal(model_name, "DlyHalfReach", "DlyFullReach", "movement", "proprioception")
    _manifold_traversal(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay", "rule")
    _manifold_traversal(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "delay", "proprioception")
    _manifold_traversal(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement", "rule")
    _manifold_traversal(model_name, "DlyHalfCircleClk", "DlyFigure8Inv", "movement", "proprioception")




def _two_task_variance_explained(model_name, task1, task2):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/variance_explained"

    options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "delay_cond": 1, "speed_cond": 5}

    trial_data_1 = _test(model_path, model_file, options, env=env_dict[task1])
    trial_data_2 = _test(model_path, model_file, options, env=env_dict[task2])

    # TODO this is wrong, do it during specific epochs not just everything
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
        
    plt.rc('figure', figsize=(3, 6))
    plt.ylim([0, 1])
    plt.plot(variance_task_1, color="black", marker="o", alpha=0.5, label=task1, markersize=15)
    plt.plot(variance_task_2, color="purple", marker="o", alpha=0.5, label=task2, markersize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    save_fig(os.path.join(exp_path, f"{task1}_{task2}"), eps=True)




def ve_halfreach_fullreach(model_name):
    _two_task_variance_explained(model_name, "DlyHalfReach", "DlyFullReach")
def ve_halfcircleclk_figure8inv(model_name):
    _two_task_variance_explained(model_name, "DlyHalfCircleClk", "DlyFigure8Inv")


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

def _plot_task_subsets(combinations, metric, label, color=None):
    task_metric = _convert_motif_dict_to_list(combinations, metric)
    task_metric_mean = sum(task_metric) / len(task_metric)
    if color:
        sns.kdeplot(task_metric, color=color, label=label, fill=True, zorder=10, legend=False, linewidth=2, alpha=0.25)
        plt.axvline(task_metric_mean, color=color, zorder=20, linestyle="dashed")
    else:
        sns.kdeplot(task_metric, fill=True, label=label, zorder=10, legend=False, linewidth=2, alpha=0.25)
        plt.axvline(task_metric_mean)
    

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

    # Gather task subsets
    inverse_tasks = [
        ("DlyHalfCircleClk", "DlyHalfCircleCClk"),
        ("DlyHalfCircleClk", "DlyFullCircleCClk1"),
        ("DlySinusoid", "DlySinusoidInv"),
        ("DlySinusoid", "DlyFigure8Inv1"),
        ("DlyFullCircleClk1", "DlyFullCircleCClk1"),
        ("DlyFullCircleClk2", "DlyFullCircleCClk2"),
        ("DlyFigure81", "DlyFigure8Inv1"),
        ("DlyFigure82", "DlyFigure8Inv2")
    ]

    subset_forward_tasks = [
        ("DlyHalfCircleClk", "DlyFullCircleClk1"),
        ("DlySinusoid", "DlyFigure81"),
        ("DlyHalfReach", "DlyFullReach1"),
        ("DlyHalfCircleCClk", "DlyFullCircleCClk1"),
        ("DlySinusoidInv", "DlyFigure8Inv1"),
    ]

    subset_backward_tasks = [
        ("DlyHalfCircleClk", "DlyFullCircleClk2"),
        ("DlySinusoid", "DlyFigure82"),
        ("DlyHalfReach", "DlyFullReach2"),
        ("DlyHalfCircleCClk", "DlyFullCircleCClk2"),
        ("DlySinusoidInv", "DlyFigure8Inv2"),
    ]

    """
    large_h_dists = []
    for dist_combination in distances_h:
        mean_conds = sum(distances_h[dist_combination]) / len(distances_h[dist_combination])
        if mean_conds > 75:
            large_h_dists.append(dist_combination)

    close_subsets = []
    for dist_combination in distances_muscle:
        mean_conds = sum(distances_muscle[dist_combination]) / len(distances_muscle[dist_combination])
        if dist_combination in large_h_dists and mean_conds < 3:
            close_subsets.append(dist_combination)
    """

    all_subsets = {
        "inverse_tasks": inverse_tasks, 
        "subset_forward_tasks": subset_forward_tasks, 
        "subset_backward_tasks": subset_backward_tasks, 
    }

    all_subsets_colors = {
        "inverse_tasks": "blue",
        "subset_forward_tasks": "purple",
        "subset_backward_tasks": "orange",
    }

    # ------------------------------------------------------------- NEURAL DISTANCES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for subset in all_subsets:
        _plot_task_subsets(all_subsets[subset], distances_h, subset, color=all_subsets_colors[subset])
    _plot_task_subsets(combination_labels, distances_h, "All Combinations", color="grey")
    save_fig(os.path.join(exp_path, epoch, "neural_distances"), eps=True)

    # ------------------------------------------------------------- MUSCLE DISTANCES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for subset in all_subsets:
        _plot_task_subsets(all_subsets[subset], distances_muscle, subset, color=all_subsets_colors[subset])
    _plot_task_subsets(combination_labels, distances_muscle, "All Combinations", color="grey")
    save_fig(os.path.join(exp_path, epoch, "muscle_distances"), eps=True)

    # ------------------------------------------------------------- NEURAL ANGLES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for subset in all_subsets:
        _plot_task_subsets(all_subsets[subset], angles_h, subset, color=all_subsets_colors[subset])
    _plot_task_subsets(combination_labels, angles_h, "All Combinations", color="grey")
    save_fig(os.path.join(exp_path, epoch, "neural_angles"), eps=True)

    # ------------------------------------------------------------- MUSCLE ANGLES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for subset in all_subsets:
        _plot_task_subsets(all_subsets[subset], angles_muscle, subset, color=all_subsets_colors[subset])
    _plot_task_subsets(combination_labels, angles_muscle, "All Combinations", color="grey")
    save_fig(os.path.join(exp_path, epoch, "muscle_angles"), eps=True)

    # ------------------------------------------------------------- NEURAL SHAPES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.set_xlim(0, 0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for subset in all_subsets:
        _plot_task_subsets(all_subsets[subset], shapes_h, subset, color=all_subsets_colors[subset])
    _plot_task_subsets(combination_labels, shapes_h, "All Combinations", color="grey")
    save_fig(os.path.join(exp_path, epoch, "neural_shapes"), eps=True)

    # ------------------------------------------------------------- MUSCLE SHAPES

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.set_xlim(0, 0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for subset in all_subsets:
        _plot_task_subsets(all_subsets[subset], shapes_muscle, subset, color=all_subsets_colors[subset])
    _plot_task_subsets(combination_labels, shapes_muscle, "All Combinations", color="grey")
    save_fig(os.path.join(exp_path, epoch, "muscle_shapes"), eps=True)

    all_distances_h = _convert_motif_dict_to_list(combination_labels, distances_h)
    all_distances_muscle = _convert_motif_dict_to_list(combination_labels, distances_muscle)

    all_angles_h = _convert_motif_dict_to_list(combination_labels, angles_h)
    all_angles_muscle = _convert_motif_dict_to_list(combination_labels, angles_muscle)

    all_shapes_h = _convert_motif_dict_to_list(combination_labels, shapes_h)
    all_shapes_muscle = _convert_motif_dict_to_list(combination_labels, shapes_muscle)

    # -------------------------------------- NEURAL SHAPE VS MUSCLE SHAPE

    # create figure and 3d axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    shape_regress = scipy.stats.linregress(all_shapes_h, all_shapes_muscle)
    cc_shape = shape_regress.rvalue
    print("cc neural shape to muscle shape: ", cc_shape)
    x = np.linspace(0, max(all_shapes_h))
    ax.plot(x, shape_regress.slope * x + shape_regress.intercept, linestyle="dashed", color="grey")
    ax.scatter(all_shapes_h, all_shapes_muscle, s=100, alpha=0.5, color="purple")
    save_fig(os.path.join(exp_path, epoch, "neural_shape_vs_muscle_shape"), eps=True)

    # -------------------------------------- NEURAL SHAPE VS NEURAL ANGLE

    # create figure and 3d axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    shape_regress = scipy.stats.linregress(all_angles_h, all_shapes_h)
    cc_shape = shape_regress.rvalue
    print("cc neural shape to neural angle: ", cc_shape)
    x = np.linspace(0, max(all_angles_h))
    ax.plot(x, shape_regress.slope * x + shape_regress.intercept, linestyle="dashed", color="grey")
    ax.scatter(all_angles_h, all_shapes_h, s=100, alpha=0.5, color="purple")
    save_fig(os.path.join(exp_path, epoch, "neural_angle_vs_neural_shape"), eps=True)

    # -------------------------------------- NEURAL DISTANCE VS MUSCLE DISTANCE

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    diff_regress = scipy.stats.linregress(all_distances_h, all_distances_muscle)
    cc_diff = diff_regress.rvalue
    print("CC neural diff to muscle diff: ", cc_diff)
    x = np.linspace(0, max(all_distances_h))
    ax.plot(x, diff_regress.slope * x + diff_regress.intercept, linestyle="dashed", color="grey")
    ax.scatter(all_distances_h, all_distances_muscle, s=100, alpha=0.5, color="purple")
    save_fig(os.path.join(exp_path, epoch, "neural_diff_vs_muscle_diff"), eps=True)

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
    ax.plot(x, diff_regress.slope * x + diff_regress.intercept, linestyle="dashed", color="grey")
    ax.scatter(all_angles_h, all_angles_muscle, s=100, alpha=0.5, color="purple")
    save_fig(os.path.join(exp_path, epoch, "neural_angle_vs_muscle_angle"), eps=True)

    # -------------------------------------- NEURAL DISTANCE VS MUSCLE SHAPE

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    diff_regress = scipy.stats.linregress(all_distances_h, all_shapes_muscle)
    cc_diff = diff_regress.rvalue
    print("CC neural diff to muscle shape: ", cc_diff)
    x = np.linspace(0, max(all_distances_h))
    ax.plot(x, diff_regress.slope * x + diff_regress.intercept, linestyle="dashed", color="grey")
    ax.scatter(all_distances_h, all_shapes_muscle, s=100, alpha=0.5, color="purple")
    save_fig(os.path.join(exp_path, epoch, "neural_diff_vs_muscle_shape"), eps=True)

    # -------------------------------------- NEURAL SHAPE VS MUSCLE DISTANCE

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    shape_regress = scipy.stats.linregress(all_shapes_h, all_distances_muscle)
    cc_diff = shape_regress.rvalue
    print("CC neural shape to muscle diff: ", cc_diff)
    x = np.linspace(0, max(all_shapes_h))
    ax.plot(x, shape_regress.slope * x + shape_regress.intercept, linestyle="dashed", color="grey")
    ax.scatter(all_shapes_h, all_distances_muscle, s=100, alpha=0.5, color="purple")
    save_fig(os.path.join(exp_path, epoch, "neural_shape_vs_muscle_diff"), eps=True)




def trajectory_alignment_movement(model_name):
    _trajectory_alignment(model_name, "movement")
def trajectory_alignment_delay(model_name):
    _trajectory_alignment(model_name, "delay")




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

    elif args.experiment == "compute_interpolated_fps_halfreach_fullreach_delay":
        compute_interpolated_fps_halfreach_fullreach_delay(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfreach_fullreach_delay_rule":
        compute_interpolated_fps_halfreach_fullreach_delay_rule(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfreach_fullreach_delay_proprioception":
        compute_interpolated_fps_halfreach_fullreach_delay_proprioception(args.model_name)

    elif args.experiment == "compute_interpolated_fps_halfreach_fullreach_movement":
        compute_interpolated_fps_halfreach_fullreach_movement(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfreach_fullreach_movement_rule":
        compute_interpolated_fps_halfreach_fullreach_movement_rule(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfreach_fullreach_movement_proprioception":
        compute_interpolated_fps_halfreach_fullreach_movement_proprioception(args.model_name)

    elif args.experiment == "compute_interpolated_fps_halfcircleclk_figure8inv_delay":
        compute_interpolated_fps_halfcircleclk_figure8inv_delay(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfcircleclk_figure8inv_delay_rule":
        compute_interpolated_fps_halfcircleclk_figure8inv_delay_rule(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfcircleclk_figure8inv_delay_proprioception":
        compute_interpolated_fps_halfcircleclk_figure8inv_delay_proprioception(args.model_name)

    elif args.experiment == "compute_interpolated_fps_halfcircleclk_figure8inv_movement":
        compute_interpolated_fps_halfcircleclk_figure8inv_movement(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfcircleclk_figure8inv_movement_rule":
        compute_interpolated_fps_halfcircleclk_figure8inv_movement_rule(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfcircleclk_figure8inv_movement_proprioception":
        compute_interpolated_fps_halfcircleclk_figure8inv_movement_proprioception(args.model_name)

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

    # --------------------------------------------------------- NEURAL MANIFOLD TRAVERSAL

    elif args.experiment == "neural_manifold_traversal_halfreach_fullreach_delay":
        neural_manifold_traversal_halfreach_fullreach_delay(args.model_name)
    elif args.experiment == "neural_manifold_traversal_halfreach_fullreach_delay_rule":
        neural_manifold_traversal_halfreach_fullreach_delay_rule(args.model_name)
    elif args.experiment == "neural_manifold_traversal_halfreach_fullreach_delay_proprioception":
        neural_manifold_traversal_halfreach_fullreach_delay_proprioception(args.model_name)

    elif args.experiment == "neural_manifold_traversal_halfreach_fullreach_movement":
        neural_manifold_traversal_halfreach_fullreach_movement(args.model_name)
    elif args.experiment == "neural_manifold_traversal_halfreach_fullreach_movement_rule":
        neural_manifold_traversal_halfreach_fullreach_movement_rule(args.model_name)
    elif args.experiment == "neural_manifold_traversal_halfreach_fullreach_movement_proprioception":
        neural_manifold_traversal_halfreach_fullreach_movement_proprioception(args.model_name)

    elif args.experiment == "neural_manifold_traversal_halfcircleclk_figure8inv_delay":
        neural_manifold_traversal_halfcircleclk_figure8inv_delay(args.model_name)
    elif args.experiment == "neural_manifold_traversal_halfcircleclk_figure8inv_delay_rule":
        neural_manifold_traversal_halfcircleclk_figure8inv_delay_rule(args.model_name)
    elif args.experiment == "neural_manifold_traversal_halfcircleclk_figure8inv_delay_proprioception":
        neural_manifold_traversal_halfcircleclk_figure8inv_delay_proprioception(args.model_name)

    elif args.experiment == "neural_manifold_traversal_halfcircleclk_figure8inv_movement":
        neural_manifold_traversal_halfcircleclk_figure8inv_movement(args.model_name)
    elif args.experiment == "neural_manifold_traversal_halfcircleclk_figure8inv_movement_rule":
        neural_manifold_traversal_halfcircleclk_figure8inv_movement_rule(args.model_name)
    elif args.experiment == "neural_manifold_traversal_halfcircleclk_figure8inv_movement_proprioception":
        neural_manifold_traversal_halfcircleclk_figure8inv_movement_proprioception(args.model_name)

    # --------------------------------------------------------- MOTOR MANIFOLD TRAVERSAL

    elif args.experiment == "motor_manifold_traversal_halfreach_fullreach_delay":
        motor_manifold_traversal_halfreach_fullreach_delay(args.model_name)
    elif args.experiment == "motor_manifold_traversal_halfreach_fullreach_delay_rule":
        motor_manifold_traversal_halfreach_fullreach_delay_rule(args.model_name)
    elif args.experiment == "motor_manifold_traversal_halfreach_fullreach_delay_proprioception":
        motor_manifold_traversal_halfreach_fullreach_delay_proprioception(args.model_name)

    elif args.experiment == "motor_manifold_traversal_halfreach_fullreach_movement":
        motor_manifold_traversal_halfreach_fullreach_movement(args.model_name)
    elif args.experiment == "motor_manifold_traversal_halfreach_fullreach_movement_rule":
        motor_manifold_traversal_halfreach_fullreach_movement_rule(args.model_name)
    elif args.experiment == "motor_manifold_traversal_halfreach_fullreach_movement_proprioception":
        motor_manifold_traversal_halfreach_fullreach_movement_proprioception(args.model_name)

    elif args.experiment == "motor_manifold_traversal_halfcircleclk_figure8inv_delay":
        motor_manifold_traversal_halfcircleclk_figure8inv_delay(args.model_name)
    elif args.experiment == "motor_manifold_traversal_halfcircleclk_figure8inv_delay_rule":
        motor_manifold_traversal_halfcircleclk_figure8inv_delay_rule(args.model_name)
    elif args.experiment == "motor_manifold_traversal_halfcircleclk_figure8inv_delay_proprioception":
        motor_manifold_traversal_halfcircleclk_figure8inv_delay_proprioception(args.model_name)

    elif args.experiment == "motor_manifold_traversal_halfcircleclk_figure8inv_movement":
        motor_manifold_traversal_halfcircleclk_figure8inv_movement(args.model_name)
    elif args.experiment == "motor_manifold_traversal_halfcircleclk_figure8inv_movement_rule":
        motor_manifold_traversal_halfcircleclk_figure8inv_movement_rule(args.model_name)
    elif args.experiment == "motor_manifold_traversal_halfcircleclk_figure8inv_movement_proprioception":
        motor_manifold_traversal_halfcircleclk_figure8inv_movement_proprioception(args.model_name)
    elif args.experiment == "run_all_manifold_traversal":
        run_all_manifold_traversal(args.model_name)
    elif args.experiment == "run_all_plot_interpolated_fps":
        run_all_plot_interpolated_fps(args.model_name)
    elif args.experiment == "run_all_compute_interpolated_fps":
        run_all_compute_interpolated_fps(args.model_name)
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
    else:
        raise ValueError("Experiment not in this file")