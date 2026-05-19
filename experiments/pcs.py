import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import config
import tqdm as tqdm
from utils.reduction_utils import plot_pca3d

""" The functions here are currently doing pca on each environment then plotting, this may change
"""

plt.rcParams.update({"font.size": 18})  # Sets default font size for all text


def plot_neural_pca_delay(model_name):
    plot_pca3d(model_name, "delay", "neural", "direction")


def plot_neural_pca_movement(model_name):
    plot_pca3d(model_name, "movement", "neural", "direction")


def plot_motor_pca_delay(model_name):
    plot_pca3d(model_name, "delay", "muscle", "direction")


def plot_motor_pca_movement(model_name):
    plot_pca3d(model_name, "movement", "muscle", "direction")


def plot_neural_pca_speeds_delay(model_name):
    plot_pca3d(model_name, "delay", "neural", "speed")


def plot_neural_pca_speeds_movement(model_name):
    plot_pca3d(model_name, "movement", "neural", "speed")


def plot_motor_pca_speeds_delay(model_name):
    plot_pca3d(model_name, "delay", "muscle", "speed")


def plot_motor_pca_speeds_movement(model_name):
    plot_pca3d(model_name, "movement", "muscle", "speed")


def plot_jpcs_delay(model_name):
    _plot_jpcs(model_name, "delay")


def plot_jpcs_movement(model_name):
    _plot_jpcs(model_name, "movement")


def plot_all_task_trajectories_neural(model_name):
    _plot_all_task_trajectories(model_name, "h")


def plot_all_task_trajectories_muscle(model_name):
    _plot_all_task_trajectories(model_name, "muscle_acts")


if __name__ == "__main__":
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "plot_neural_pca_delay":
        plot_neural_pca_delay(args.model_name)
    elif args.experiment == "plot_neural_pca_movement":
        plot_neural_pca_movement(args.model_name)
    elif args.experiment == "plot_motor_pca_delay":
        plot_motor_pca_delay(args.model_name)
    elif args.experiment == "plot_motor_pca_movement":
        plot_motor_pca_movement(args.model_name)

    elif args.experiment == "plot_neural_pca_speeds_delay":
        plot_neural_pca_speeds_delay(args.model_name)
    elif args.experiment == "plot_neural_pca_speeds_movement":
        plot_neural_pca_speeds_movement(args.model_name)
    elif args.experiment == "plot_motor_pca_speeds_delay":
        plot_motor_pca_speeds_delay(args.model_name)
    elif args.experiment == "plot_motor_pca_speeds_movement":
        plot_motor_pca_speeds_movement(args.model_name)

    elif args.experiment == "plot_jpcs_delay":
        plot_jpcs_delay(args.model_name)
    elif args.experiment == "plot_jpcs_movement":
        plot_jpcs_movement(args.model_name)

    elif args.experiment == "plot_all_task_trajectories_neural":
        plot_all_task_trajectories_neural(args.model_name)
    elif args.experiment == "plot_all_task_trajectories_muscle":
        plot_all_task_trajectories_muscle(args.model_name)

    else:
        raise ValueError("Experiment not in this file")
