import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp

import warnings
warnings.filterwarnings("ignore")

import torch
from utils import load_hp, save_fig 
import matplotlib.pyplot as plt
import numpy as np
import config
from analysis.clustering import Analysis
import tqdm as tqdm
from losses import l1_dist
from exp_utils import _test, env_dict


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
    feedback_masks["go_cue"] = go_mask

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
    plt.xticks(ticks=np.arange(0, 7), labels=[mask for mask in feedback_masks], rotation=45)
    plt.yticks(ticks=np.arange(0, 10), labels=[env for env in env_dict], rotation=45)
    cbar = plt.colorbar(img)
    cbar.set_label('Change in Loss')
    save_fig(exp_path)




if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "module_silencing":
        module_silencing(args.model_name)
    elif args.experiment == "feedback_ablation":
        feedback_ablation(args.model_name)
    else:
        raise ValueError("Experiment not in this file")
