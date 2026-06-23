import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import config
from utils.reduction_utils import plot_mean_trial_activity

plt.rcParams.update({"font.size": 18})


def plot_psth(model_name):
    plot_mean_trial_activity(model_name)


if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "plot_psth":
        plot_psth(args.model_name)
    else:
        raise ValueError("Experiment not in this file")
