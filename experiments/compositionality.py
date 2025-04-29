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


def _get_pcs(model_name, batch_size=8, epoch=None, use_reach_conds=True, speed_cond=5, delay_cond=1, noise=False, system="neural"):

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
            env_hs.append(trial_data[mode][:, trial_data["epoch_bounds"]["movement"][1]-1].unsqueeze(1))
        else:
            raise ValueError("not valid epoch")

    pca_3d = PCA(n_components=3)
    pca_3d.fit(torch.cat(env_hs, dim=1).reshape((-1, size)))

    return pca_3d, env_hs




def _epoch_pcs(model_name, epoch, system):

    exp_path = f"results/{model_name}/compositionality/pcs"
    create_dir(exp_path)

    pca_3d, env_hs = _get_pcs(model_name, batch_size=256, use_reach_conds=False, epoch=epoch, noise=True, system=system)

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
    plt.legend(handles=handles)
    save_fig(os.path.join(exp_path, f"{system}_{epoch}_pcs.png"))




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




def _rule_interpolated_fps(model_name, task1, task2, epoch, input_component=None):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"

    NOISE_SCALE = 1 # Standard deviation of noise added to initial states
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
        'tol_unique': 0.1,
        'do_compute_jacobians': False}
        
    cond_fps_list = []
    for i, (cond1, cond2) in enumerate(zip(inp1, inp2)):

        # Need to reset everything at beginning of next condition
        if epoch == "delay":
            # Get inputs and x and h from desired timepoint
            x_int = trial_data1["x"][i:i+1, trial_data1["epoch_bounds"]["delay"][1]-1]
            h_int = trial_data1["h"][i:i+1, trial_data1["epoch_bounds"]["delay"][1]-1]

        elif epoch == "movement":
            x_int = trial_data1["x"][i:i+1, middle_movement1]
            h_int = trial_data1["h"][i:i+1, middle_movement1]

        # Setup environment and initialize it

        """
            Could either initialize everything in need in the environment to be at the desired timepoint in the trial
            or run the trial again to the desired timpoint. The first is faster but for now more difficult so going with #2
        """
        env = env_dict[task1](effector=mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle()))
        obs, info = env.reset(testing=True, options=options)
        # May need to change in the future if I do middle of movement
        x = torch.zeros(size=(1, hp["hid_size"]))
        h = torch.zeros(size=(1, hp["hid_size"]))

        if epoch == "delay":
            timesteps = env.epoch_bounds["delay"][1]-1
        elif epoch == "movement":
            timesteps = middle_movement1
            
        for t in range(timesteps):
            with torch.no_grad():
                x, h, action = policy(obs[i].unsqueeze(0), x, h, noise=False)
                obs, reward, terminated, info = env.step(t, action=action)
        last_t = t

        # Draw a line from fingertip to goal 
        if input_component == None:
            interpolated_input = cond1.unsqueeze(0) + \
                torch.linspace(0, 1, steps=100).unsqueeze(1) * (cond2 - cond1).unsqueeze(0)

        elif input_component == "rule":
            interpolated_input_rule = cond1[:10].unsqueeze(0) + \
                torch.linspace(0, 1, steps=100).unsqueeze(1) * (cond2[:10] - cond1[:10]).unsqueeze(0)
            fixed_inp = cond1[10:].repeat(100, 1)
            interpolated_input = torch.cat([interpolated_input_rule, fixed_inp], dim=1)

        elif input_component == "speed_scalar":
            fixed_inp_pre = cond1[10:].repeat(100, 1)
            interpolated_input_speed = cond1[10:11].unsqueeze(0) + \
                torch.linspace(0, 1, steps=100).unsqueeze(1) * (cond2[10:11] - cond1[10:11]).unsqueeze(0)
            fixed_inp_post = cond1[11:].repeat(100, 1)
            interpolated_input = torch.cat([fixed_inp_pre, interpolated_input_speed, fixed_inp_post], dim=1)

        elif input_component == "go_cue":
            fixed_inp_pre = cond1[11:].repeat(100, 1)
            interpolated_input_go_cue = cond1[11:12].unsqueeze(0) + \
                torch.linspace(0, 1, steps=100).unsqueeze(1) * (cond2[11:12] - cond1[11:12]).unsqueeze(0)
            fixed_inp_post = cond1[12:].repeat(100, 1)
            interpolated_input = torch.cat([fixed_inp_pre, interpolated_input_go_cue, fixed_inp_post], dim=1)

        elif input_component == "vis_inp":
            fixed_inp_pre = cond1[12:].repeat(100, 1)
            interpolated_input_vis_inp = cond1[12:14].unsqueeze(0) + \
                torch.linspace(0, 1, steps=100).unsqueeze(1) * (cond2[12:14] - cond1[12:14]).unsqueeze(0)
            fixed_inp_post = cond1[14:].repeat(100, 1)
            interpolated_input = torch.cat([fixed_inp_pre, interpolated_input_vis_inp, fixed_inp_post], dim=1)

        elif input_component == "visual_feedback":
            fixed_inp_pre = cond1[14:].repeat(100, 1)
            interpolated_input_vis_feedback = cond1[14:16].unsqueeze(0) + \
                torch.linspace(0, 1, steps=100).unsqueeze(1) * (cond2[14:16] - cond1[14:16]).unsqueeze(0)
            fixed_inp_post = cond1[16:].repeat(100, 1)
            interpolated_input = torch.cat([fixed_inp_pre, interpolated_input_vis_feedback, fixed_inp_post], dim=1)

        elif input_component == "proprioception":
            fixed_inp_pre = cond1[:16].repeat(100, 1)
            interpolated_input_proprioception = cond1[16:].unsqueeze(0) + \
                torch.linspace(0, 1, steps=100).unsqueeze(1) * (cond2[16:] - cond1[16:]).unsqueeze(0)
            interpolated_input = torch.cat([fixed_inp_pre, interpolated_input_proprioception], dim=1)

        h0 = h_int.clone()
        fps_list = []
        # Going thorugh each interpolated input
        for j, inp in enumerate(interpolated_input):

            # Setup the fixed point finder
            fpf = FixedPointFinder(policy.mrnn, **fpf_hps)

            '''Draw random, noise corrupted samples of those state trajectories
            to use as initial states for the fixed point optimizations.'''

            """
                IMPORTANT: For now, treating the continuously interpolated input as a trajectory for the arm and RNN
                Feedback will not match the arm states during this trajectory.
            """

            # Get next hidden state from interpolated input trajectory
            with torch.no_grad():
                x_int, h_int, action_int = policy(inp[None, :], x_int, h_int)
                # t doesnt matter in this case since the input only comes from inp
                _, _, _, info = env.step(last_t, action=action_int)

            # Use these hidden states for initialization
            initial_states = fpf.sample_states(h0[None],
                n_inits=N_INITS,
                noise_scale=NOISE_SCALE)

            # Run the fixed point finder
            unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs=inp[None, :])

            # TODO eventually add joint angles as well to do "motor space"
            # Add fixed points and their info to dict
            fps_list.append(
                {"fps": unique_fps, 
                "interp_point": j, 
                "x_state": x_int, 
                "h_state": h_int,
                "muscle_acts": info["states"]["muscle"][:, 0].unsqueeze(1),
                "fingertip": info["states"]["fingertip"][:, None, :]
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
def compute_interpolated_fps_halfcircleclk_halfcirclecclk_delay(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "delay")
def compute_interpolated_fps_halfcircleclk_halfcirclecclk_delay_rule(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "delay", "rule")
def compute_interpolated_fps_halfcircleclk_halfcirclecclk_delay_proprioception(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "delay", "proprioception")

# Movement period with different input interpolations
def compute_interpolated_fps_halfcircleclk_halfcirclecclk_movement(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "movement")
def compute_interpolated_fps_halfcircleclk_halfcirclecclk_movement_rule(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "movement", "rule")
def compute_interpolated_fps_halfcircleclk_halfcirclecclk_movement_proprioception(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "movement", "proprioception")

# Dissimilar tasks

# Delay period with different input interpolations
def compute_interpolated_fps_halfreach_figure8inv_delay(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "delay")
def compute_interpolated_fps_halfreach_figure8inv_delay_rule(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "delay", "rule")
def compute_interpolated_fps_halfreach_figure8inv_delay_proprioception(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "delay", "proprioception")

# Movement period with different input interpolations
def compute_interpolated_fps_halfreach_figure8inv_movement(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "movement")
def compute_interpolated_fps_halfreach_figure8inv_movement_rule(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "movement", "rule")
def compute_interpolated_fps_halfreach_figure8inv_movement_proprioception(model_name):
    _rule_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "movement", "proprioception")




def _plot_interpolated_fps(model_name, task1, task2, epoch, input_component=None):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    load_name = os.path.join(model_path, f"interpolated_fps_{task1}_{task2}_{epoch}_{input_component}.pkl")
    exp_path = f"results/{model_name}/compositionality/interpolated_fps"

    fps = load_pickle(load_name)

    colors = plt.cm.inferno(np.linspace(0, 1, 100)) 

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
        task1_pca.fit(trial_data1["h"][i, trial_data1["epoch_bounds"][epoch][0]:trial_data1["epoch_bounds"][epoch][1]])

        # Create figure and 3D axes
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')  # or projection='3d'

        task1_h_reduced = task1_pca.transform(trial_data1["h"][i, trial_data1["epoch_bounds"][epoch][0]:trial_data1["epoch_bounds"][epoch][1]])
        task2_h_reduced = task1_pca.transform(trial_data2["h"][i, trial_data2["epoch_bounds"][epoch][0]:trial_data2["epoch_bounds"][epoch][1]])

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
                ax.plot((j/100)*np.ones_like(zstar)[:, 0], zstar[:, 0], zstar[:, 1], marker='.', alpha=0.5, color=colors[j], markersize=12)

        ax.grid(False)
        save_fig(os.path.join(exp_path, f"{task1}_{task2}", f"{epoch}", f"{input_component}", save_name))




# Similar Tasks
def plot_interpolated_fps_halfcircleclk_halfcirclecclk_delay(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "delay")
def plot_interpolated_fps_halfcircleclk_halfcirclecclk_delay_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "delay", "rule")
def plot_interpolated_fps_halfcircleclk_halfcirclecclk_delay_proprioception(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "delay", "proprioception")

def plot_interpolated_fps_halfcircleclk_halfcirclecclk_movement(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "movement")
def plot_interpolated_fps_halfcircleclk_halfcirclecclk_movement_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "movement", "rule")
def plot_interpolated_fps_halfcircleclk_halfcirclecclk_movement_proprioception(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfCircleClk", "DlyHalfCircleCClk", "movement", "proprioception")

# Dissimilar Tasks
def plot_interpolated_fps_halfreach_figure8inv_delay(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "delay")
def plot_interpolated_fps_halfreach_figure8inv_delay_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "delay", "rule")
def plot_interpolated_fps_halfreach_figure8inv_delay_proprioception(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "delay", "proprioception")

def plot_interpolated_fps_halfreach_figure8inv_movement(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "movement")
def plot_interpolated_fps_halfreach_figure8inv_movement_rule(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "movement", "rule")
def plot_interpolated_fps_halfreach_figure8inv_movement_proprioception(model_name):
    _plot_interpolated_fps(model_name, "DlyHalfReach", "DlyFigure8Inv", "movement", "proprioception")




def _manifold_traversal(model_name, task1, task2, epoch):
    pass



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

    # --------------------------------------------------------- COMPUTE INTERPOLATED FPS
    elif args.experiment == "compute_interpolated_fps_halfcircleclk_halfcirclecclk_delay":
        compute_interpolated_fps_halfcircleclk_halfcirclecclk_delay(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfcircleclk_halfcirclecclk_delay_rule":
        compute_interpolated_fps_halfcircleclk_halfcirclecclk_delay_rule(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfcircleclk_halfcirclecclk_delay_proprioception":
        compute_interpolated_fps_halfcircleclk_halfcirclecclk_delay_proprioception(args.model_name)

    elif args.experiment == "compute_interpolated_fps_halfcircleclk_halfcirclecclk_movement":
        compute_interpolated_fps_halfcircleclk_halfcirclecclk_movement(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfcircleclk_halfcirclecclk_movement_rule":
        compute_interpolated_fps_halfcircleclk_halfcirclecclk_movement_rule(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfcircleclk_halfcirclecclk_movement_proprioception":
        compute_interpolated_fps_halfcircleclk_halfcirclecclk_movement_proprioception(args.model_name)

    elif args.experiment == "compute_interpolated_fps_halfreach_figure8inv_delay":
        compute_interpolated_fps_halfreach_figure8inv_delay(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfreach_figure8inv_delay_rule":
        compute_interpolated_fps_halfreach_figure8inv_delay_rule(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfreach_figure8inv_delay_proprioception":
        compute_interpolated_fps_halfreach_figure8inv_delay_proprioception(args.model_name)

    elif args.experiment == "compute_interpolated_fps_halfreach_figure8inv_movement":
        compute_interpolated_fps_halfreach_figure8inv_movement(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfreach_figure8inv_movement_rule":
        compute_interpolated_fps_halfreach_figure8inv_movement_rule(args.model_name)
    elif args.experiment == "compute_interpolated_fps_halfreach_figure8inv_movement_proprioception":
        compute_interpolated_fps_halfreach_figure8inv_movement_proprioception(args.model_name)

    # --------------------------------------------------------- PLOT INTERPOLATED FPS
    elif args.experiment == "plot_interpolated_fps_halfcircleclk_halfcirclecclk_delay":
        plot_interpolated_fps_halfcircleclk_halfcirclecclk_delay(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfcircleclk_halfcirclecclk_delay_rule":
        plot_interpolated_fps_halfcircleclk_halfcirclecclk_delay_rule(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfcircleclk_halfcirclecclk_delay_proprioception":
        plot_interpolated_fps_halfcircleclk_halfcirclecclk_delay_proprioception(args.model_name)

    elif args.experiment == "plot_interpolated_fps_halfcircleclk_halfcirclecclk_movement":
        plot_interpolated_fps_halfcircleclk_halfcirclecclk_movement(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfcircleclk_halfcirclecclk_movement_rule":
        plot_interpolated_fps_halfcircleclk_halfcirclecclk_movement_rule(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfcircleclk_halfcirclecclk_movement_proprioception":
        plot_interpolated_fps_halfcircleclk_halfcirclecclk_movement_proprioception(args.model_name)

    elif args.experiment == "plot_interpolated_fps_halfreach_figure8inv_delay":
        plot_interpolated_fps_halfreach_figure8inv_delay(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfreach_figure8inv_delay_rule":
        plot_interpolated_fps_halfreach_figure8inv_delay_rule(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfreach_figure8inv_delay_proprioception":
        plot_interpolated_fps_halfreach_figure8inv_delay_proprioception(args.model_name)

    elif args.experiment == "plot_interpolated_fps_halfreach_figure8inv_movement":
        plot_interpolated_fps_halfreach_figure8inv_movement(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfreach_figure8inv_movement_rule":
        plot_interpolated_fps_halfreach_figure8inv_movement_rule(args.model_name)
    elif args.experiment == "plot_interpolated_fps_halfreach_figure8inv_movement_proprioception":
        plot_interpolated_fps_halfreach_figure8inv_movement_proprioception(args.model_name)

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
    else:
        raise ValueError("Experiment not in this file")