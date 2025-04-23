import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp, interpolate_trial

import warnings
warnings.filterwarnings("ignore")

import torch
import os
from utils import load_hp, interpolate_trial
import config
from analysis.clustering import Analysis
import pickle
import tqdm as tqdm
from exp_utils import _test, env_dict

def variance_by_rule(model_name):

    # Get variance of units across tasks and save to pickle file in model directory
    # Doing so across rules only
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    hp = load_hp(model_path)

    env_var_dict = {}
    var_dir_tensor = torch.empty(size=(hp["hid_size"], len(env_dict)))
    task_list = []

    for i, env in enumerate(env_dict):
        dir_var_list = []
        for speed in range(10):

            options = {"batch_size": 32, "reach_conds": torch.arange(0, 32, 1), "speed_cond": speed, "delay_cond": 0}
            trial_data = _test(model_path, model_file, options, env=env_dict[env])
            delay_time = trial_data["epoch_bounds"]["delay"][0]

            # Should be of shape batch, time, neurons
            h = trial_data["h"][:, delay_time:]
            dir_var_list.append(h)

        # Get the max timesteps to interpolate
        max_dim1 = max(tensor.shape[1] for tensor in dir_var_list)

        # interpolate timesteps over speed conditions
        interpolated_trials = torch.stack([interpolate_trial(direction, max_dim1) for h in dir_var_list for direction in h])

        # Variance along batch dimension, average along time
        var_dir = interpolated_trials.var(dim=0).mean(dim=0)
        # shape for clustering will be samples x features, samples is neurons features is tasks
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




if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "variance_by_rule":
        variance_by_rule(args.model_name) 
    elif args.experiment == "variance_by_epoch":
        variance_by_epoch(args.model_name) 
    elif args.experiment == "plot_variance_by_rule":
        plot_variance_by_rule(args.model_name) 
    elif args.experiment == "plot_variance_by_epoch":
        plot_variance_by_epoch(args.model_name) 
    else:
        raise ValueError("Experiment not in this file")