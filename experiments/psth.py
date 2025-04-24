import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings("ignore")

import torch
import os
from utils import create_dir, save_fig
import matplotlib.pyplot as plt
import numpy as np
import config
import tqdm as tqdm
from exp_utils import _test, env_dict

def plot_psth(model_name):
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
    exp_path = f"results/{model_name}/psth"

    create_dir(exp_path)

    for env in env_dict:
        for speed in range(10):

            options = {"batch_size": 8, "reach_conds": torch.arange(0, 32, 4), "speed_cond": speed, "delay_cond": 1}

            trial_data = _test(model_path, model_file, options, env=env_dict[env])
        
            # Get kinematics and activity in a center out setting
            # On random and delay
            colors = plt.cm.inferno(np.linspace(0, 1, trial_data["h"].shape[0])) 

            delay = trial_data["epoch_bounds"]["delay"][0]
            mov = trial_data["epoch_bounds"]["movement"][0]
            hold = trial_data["epoch_bounds"]["hold"][0]

            for i, h in enumerate(trial_data["h"]):
                plt.plot(torch.mean(h, dim=-1)[delay:], color=colors[i], linewidth=4)
                plt.axvline(mov-delay, linestyle="dashed", color="grey")
                plt.axvline(hold-delay, linestyle="dashed", color="grey")
            # Access current axes and hide top/right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            save_fig(os.path.join(exp_path, f"{env}_speed{speed}_tg_trajectory.png"))




if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "plot_psth":
        plot_psth(args.model_name) 
    else:
        raise ValueError("Experiment not in this file")