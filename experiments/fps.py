import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import config
import tqdm as tqdm
from utils.fp_utils import interpolated_fps, plot_interpolated_fps

plt.rcParams.update({"font.size": 18})  # Sets default font size for all text


# ---------------------------------------------------------------- Subset Pair


# Delay period with different input interpolations
def compute_interpolated_fps_halfreach_fullreach_delay(model_name):
    interpolated_fps(model_name, "DlyHalfReach", "DlyFullReach", "delay")


# Movement period with different input interpolations
def compute_interpolated_fps_halfreach_fullreach_movement(model_name):
    interpolated_fps(
        model_name,
        "DlyHalfReach",
        "DlyFullReach",
        "movement",
        task1_period="all",
        task2_period="first",
    )


# ---------------------------------------------------------------- Extension Pair


# Delay period with different input interpolations
def compute_interpolated_fps_halfcircleclk_sinusoidinv_delay(model_name):
    interpolated_fps(model_name, "DlyHalfCircleClk", "DlySinusoidInv", "delay")


# Movement period with different input interpolations
def compute_interpolated_fps_halfcircleclk_sinusoidinv_movement(model_name):
    interpolated_fps(
        model_name,
        "DlyHalfCircleClk",
        "DlySinusoidInv",
        "movement",
        task1_period="all",
        task2_period="all",
    )


# ---------------------------------------------------------------- Retraction Pair


# Delay period with different input interpolations
def compute_interpolated_fps_fullcircleclk_figure8_delay(model_name):
    interpolated_fps(model_name, "DlyFullCircleClk", "DlyFigure8", "delay")


# Movement period with different input interpolations
def compute_interpolated_fps_fullcircleclk_figure8_movement(model_name):
    interpolated_fps(
        model_name,
        "DlyFullCircleClk",
        "DlyFigure8",
        "movement",
        task1_period="second",
        task2_period="second",
    )


# ---------------------------------------------------------------- Extension-Retraction Pair


# Delay period with different input interpolations
def compute_interpolated_fps_sinusoid_fullreach_delay(model_name):
    interpolated_fps(model_name, "DlySinusoid", "DlyFullReach", "delay")


# Movement period with different input interpolations
def compute_interpolated_fps_sinusoid_fullreach_movement(model_name):
    interpolated_fps(
        model_name,
        "DlySinusoid",
        "DlyFullReach",
        "movement",
        task1_period="all",
        task2_period="second",
    )


def run_all_compute_interpolated_fps(model_name):
    compute_interpolated_fps_halfreach_fullreach_delay(model_name)
    compute_interpolated_fps_halfreach_fullreach_movement(model_name)
    compute_interpolated_fps_halfcircleclk_sinusoidinv_delay(model_name)
    compute_interpolated_fps_halfcircleclk_sinusoidinv_movement(model_name)
    compute_interpolated_fps_fullcircleclk_figure8_delay(model_name)
    compute_interpolated_fps_fullcircleclk_figure8_movement(model_name)
    compute_interpolated_fps_sinusoid_fullreach_delay(model_name)
    compute_interpolated_fps_sinusoid_fullreach_movement(model_name)


# ---------------------------------------------------------------- Subset Pair
def plot_interpolated_fps_halfreach_fullreach_movement(model_name, save_metrics=False):
    plot_interpolated_fps(
        model_name,
        "DlyHalfReach",
        "DlyFullReach",
        "movement",
        task1_period="all",
        task2_period="first",
        save_metrics=save_metrics,
    )


# ---------------------------------------------------------------- Extension Pair
def plot_interpolated_fps_halfcircleclk_sinusoidinv_movement(
    model_name, save_metrics=False
):
    plot_interpolated_fps(
        model_name,
        "DlyHalfCircleClk",
        "DlySinusoidInv",
        "movement",
        task1_period="all",
        task2_period="all",
        save_metrics=save_metrics,
    )


# ---------------------------------------------------------------- Retraction Pair
def plot_interpolated_fps_fullcircleclk_figure8_movement(
    model_name, save_metrics=False
):
    plot_interpolated_fps(
        model_name,
        "DlyFullCircleClk",
        "DlyFigure8",
        "movement",
        task1_period="second",
        task2_period="second",
        save_metrics=save_metrics,
    )


# ---------------------------------------------------------------- Extension-Retraction Pair
def plot_interpolated_fps_sinusoid_fullreach_movement(model_name, save_metrics=False):
    plot_interpolated_fps(
        model_name,
        "DlySinusoid",
        "DlyFullReach",
        "movement",
        task1_period="all",
        task2_period="second",
        save_metrics=save_metrics,
    )


def run_all_plot_interpolated_fps(model_name):
    # Add data from each task
    plot_interpolated_fps_halfreach_fullreach_movement(model_name, save_metrics=False)
    plot_interpolated_fps_halfcircleclk_sinusoidinv_movement(
        model_name, save_metrics=False
    )
    plot_interpolated_fps_fullcircleclk_figure8_movement(model_name, save_metrics=False)
    plot_interpolated_fps_sinusoid_fullreach_movement(model_name, save_metrics=False)


if __name__ == "__main__":
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "run_all_compute_interpolated_fps":
        run_all_compute_interpolated_fps(args.model_name)
    elif args.experiment == "run_all_plot_interpolated_fps":
        run_all_plot_interpolated_fps(args.model_name)
    else:
        raise ValueError("Experiment not in this file")
