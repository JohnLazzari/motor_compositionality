import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import config
import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm

from modules.test import Test
from utils.exp_utils import env_dict, load_pickle, save_pickle
from utils.plot_utils import save_fig


INDIVIDUAL_MODEL_ENV = {
    "Reach": "rnn256_softplus_reach",
    "ClkCurvedReach": "rnn256_softplus_clkcurvedreach",
    "CClkCurvedReach": "rnn256_softplus_cclkcurvedreach",
    "Sinusoid": "rnn256_softplus_sinusoid",
    "InvSinusoid": "rnn256_softplus_invsinusoid",
    "ReachBack": "rnn256_softplus_reachback",
    "ClkCycle": "rnn256_softplus_clkcycle",
    "CClkCycle": "rnn256_softplus_cclkcycle",
    "Figure8": "rnn256_softplus_figure8",
    "InvFigure8": "rnn256_softplus_invfigure8",
}

INDIVIDUAL_MODEL_ENV_NO_FEEDBACK = {
    "Reach": "rnn256_softplus_reach_nofeedback",
    "ClkCurvedReach": "rnn256_softplus_clkcurvedreach_nofeedback",
    "CClkCurvedReach": "rnn256_softplus_cclkcurvedreach_nofeedback",
    "Sinusoid": "rnn256_softplus_sinusoid_nofeedback",
    "InvSinusoid": "rnn256_softplus_invsinusoid_nofeedback",
    "ReachBack": "rnn256_softplus_reachback_nofeedback",
    "ClkCycle": "rnn256_softplus_clkcycle_nofeedback",
    "CClkCycle": "rnn256_softplus_cclkcycle_nofeedback",
    "Figure8": "rnn256_softplus_figure8_nofeedback",
    "InvFigure8": "rnn256_softplus_invfigure8_nofeedback",
}


def collect_muscle_data(model_dict, save_name):
    """Collect muscle data from one task-specific RNN per matching environment."""
    reach_conds = np.arange(0, 32)
    speed_conds = np.arange(0, 10)
    delay_conds = np.arange(0, 3)

    muscle_data = {
        "model_name": "individual_rnns",
        "reach_conds": reach_conds,
        "speed_conds": speed_conds,
        "delay_conds": delay_conds,
        "tasks": {},
    }

    for env_name, model_name in tqdm.tqdm(
        model_dict.items(), desc="Collecting individual RNN muscle data"
    ):
        model_path = os.path.join("checkpoints", model_name)
        test = Test(model_path, model_name)
        env = env_dict[env_name]
        muscle_data["tasks"][env_name] = {}

        for speed in speed_conds:
            muscle_data["tasks"][env_name][int(speed)] = {}
            for delay_cond in delay_conds:
                options = {
                    "batch_size": len(reach_conds),
                    "reach_conds": reach_conds,
                    "speed_cond": int(speed),
                    "delay_cond": int(delay_cond),
                    "deterministic": True,
                }
                trial_data = test.trial(options, env)
                muscle_data["tasks"][env_name][int(speed)][int(delay_cond)] = {
                    "muscle_acts": trial_data["muscle_acts"],
                    "action": trial_data["action"],
                }

    save_dir = f"checkpoints/{save_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "muscle_act_data.pkl")
    save_pickle(save_path, muscle_data)
    return muscle_data


def collect_muscle_data_feedback():
    collect_muscle_data(INDIVIDUAL_MODEL_ENV, "individual_rnns")


def collect_muscle_data_nofeedback():
    collect_muscle_data(INDIVIDUAL_MODEL_ENV_NO_FEEDBACK, "individual_rnns_nofeedback")


def _to_numpy(data):
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def iter_muscle_conditions(muscle_data):
    for env_name, speed_data in muscle_data["tasks"].items():
        for speed, condition_data in speed_data.items():
            if "obs" in condition_data:
                yield env_name, int(speed), None, condition_data
            else:
                for delay_cond, delay_data in condition_data.items():
                    yield env_name, int(speed), int(delay_cond), delay_data


def plot_muscle_data(model_name):
    """Plot saved muscle activity for each task and speed."""
    model_path = f"checkpoints/{model_name}"
    muscle_data = load_pickle(os.path.join(model_path, "muscle_act_data.pkl"))
    save_path = os.path.join("results", "supervised_data", model_name)

    for env_name, speed, delay_cond, data in tqdm.tqdm(
        iter_muscle_conditions(muscle_data), desc="Plotting muscle data"
    ):
        muscle_acts = _to_numpy(data["muscle_acts"])

        fig, muscle_ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, muscle_acts.shape[-1]))
        for muscle_idx, color in enumerate(colors):
            muscle_ax.plot(
                muscle_acts[:, :, muscle_idx].T,
                color=color,
                linewidth=0.6,
                alpha=0.12,
            )
            muscle_ax.plot(
                muscle_acts[:, :, muscle_idx].mean(axis=0),
                color=color,
                linewidth=2,
                label=f"muscle {muscle_idx}",
            )
        muscle_ax.set_ylabel("muscle", rotation=0, labelpad=25, va="center")
        muscle_ax.set_xlabel("timestep")
        muscle_ax.spines["top"].set_visible(False)
        muscle_ax.spines["right"].set_visible(False)
        muscle_ax.legend(frameon=False, ncol=3, fontsize=8)

        suffix = f"{env_name}_speed{speed}"
        title = f"{env_name} speed {speed}"
        if delay_cond is not None:
            suffix = f"{suffix}_delay{delay_cond}"
            title = f"{title} delay {delay_cond}"
        fig.suptitle(title)
        save_fig(os.path.join(save_path, suffix))


if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "collect_muscle_data_feedback":
        collect_muscle_data_feedback()
    elif args.experiment == "collect_muscle_data_nofeedback":
        collect_muscle_data_nofeedback()
    elif args.experiment == "plot_muscle_data":
        plot_muscle_data(args.model_name)
    else:
        raise ValueError("Experiment not in this file")
