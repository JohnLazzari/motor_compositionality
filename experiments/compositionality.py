import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

from utils.compositionality_utils import (
    dsa_similarity_matrix,
    epoch_pcs,
    gather_all_traj_metrics,
    trajectory_metric_bar_plot,
    convert_motif_dict_to_list,
    plot_metric_scatter,
    composite_input_optimization,
)
from utils.plot_utils import save_fig, standard_2d_ax
from utils.exp_utils import load_pickle, env_dict, extension_dict, retraction_dict
import matplotlib.pyplot as plt
import numpy as np
import config
import pickle

plt.rcParams.update({"font.size": 18})  # Sets default font size for all text

################### PCA Axis Experiments ################################


def stable_pcs_extension(model_name, add_new_rule_inputs=False, plot_3d=False):
    epoch_pcs(
        model_name,
        "stable",
        "extension",
        add_new_rule_inputs=add_new_rule_inputs,
        plot_3d=plot_3d,
    )


def delay_pcs_extension(model_name, add_new_rule_inputs=False, plot_3d=False):
    epoch_pcs(
        model_name,
        "delay",
        "extension",
        add_new_rule_inputs=add_new_rule_inputs,
        plot_3d=plot_3d,
    )


def movement_pcs_extension(model_name, add_new_rule_inputs=False, plot_3d=False):
    epoch_pcs(
        model_name,
        "extension",
        "extension",
        add_new_rule_inputs=add_new_rule_inputs,
        plot_3d=plot_3d,
    )


def hold_pcs_extension(model_name, add_new_rule_inputs=False, plot_3d=False):
    epoch_pcs(
        model_name,
        "hold",
        "extension",
        add_new_rule_inputs=add_new_rule_inputs,
        plot_3d=plot_3d,
    )


def stable_pcs_extension_retraction(
    model_name, add_new_rule_inputs=False, plot_3d=False
):
    epoch_pcs(
        model_name,
        "stable",
        "extension_retraction",
        add_new_rule_inputs=add_new_rule_inputs,
        plot_3d=plot_3d,
    )


def delay_pcs_extension_retraction(
    model_name, add_new_rule_inputs=False, plot_3d=False
):
    epoch_pcs(
        model_name,
        "delay",
        "extension_retraction",
        add_new_rule_inputs=add_new_rule_inputs,
        plot_3d=plot_3d,
    )


def extension_pcs_extension_retraction(
    model_name, add_new_rule_inputs=False, plot_3d=False
):
    epoch_pcs(
        model_name,
        "extension",
        "extension_retraction",
        add_new_rule_inputs=add_new_rule_inputs,
        plot_3d=plot_3d,
    )


def retraction_pcs_extension_retraction(
    model_name, add_new_rule_inputs=False, plot_3d=False
):
    epoch_pcs(
        model_name,
        "retraction",
        "extension_retraction",
        add_new_rule_inputs=add_new_rule_inputs,
        plot_3d=plot_3d,
    )


def hold_pcs_extension_retraction(model_name, add_new_rule_inputs=False, plot_3d=False):
    epoch_pcs(
        model_name,
        "hold",
        "extension_retraction",
        add_new_rule_inputs=add_new_rule_inputs,
        plot_3d=plot_3d,
    )


def run_all_epoch_pcs(model_name):
    stable_pcs_extension(model_name)
    delay_pcs_extension(model_name)
    movement_pcs_extension(model_name)
    hold_pcs_extension(model_name)
    stable_pcs_extension_retraction(model_name)
    delay_pcs_extension_retraction(model_name)
    extension_pcs_extension_retraction(model_name)
    retraction_pcs_extension_retraction(model_name)
    hold_pcs_extension_retraction(model_name)


def run_all_epoch_pcs_transfer(model_name):
    stable_pcs_extension(model_name, add_new_rule_inputs=True)
    delay_pcs_extension(model_name, add_new_rule_inputs=True)
    movement_pcs_extension(model_name, add_new_rule_inputs=True)
    hold_pcs_extension(model_name, add_new_rule_inputs=True)
    stable_pcs_extension_retraction(model_name, add_new_rule_inputs=True)
    delay_pcs_extension_retraction(model_name, add_new_rule_inputs=True)
    extension_pcs_extension_retraction(model_name, add_new_rule_inputs=True)
    retraction_pcs_extension_retraction(model_name, add_new_rule_inputs=True)
    hold_pcs_extension_retraction(model_name, add_new_rule_inputs=True)


################### Task Metric Experiments ################################


def plot_trajectory_metric_bar_h(model_name):
    trajectory_metric_bar_plot(model_name, "h")


def plot_trajectory_metric_bar_m(model_name):
    trajectory_metric_bar_plot(model_name, "muscle")


def plot_shape_dists(model_name):
    exp_path = f"results/{model_name}/compositionality/alignment"

    all_subsets_h, _, all_metrics_h = gather_all_traj_metrics(model_name, "h")
    all_subsets_m, _, all_metrics_m = gather_all_traj_metrics(model_name, "muscle")

    all_shapes_h = convert_motif_dict_to_list(
        [key for key in all_subsets_h.keys()], all_metrics_h["shapes"]
    )
    all_shapes_muscle = convert_motif_dict_to_list(
        [key for key in all_subsets_m.keys()], all_metrics_m["shapes"]
    )

    bins = tuple(np.linspace(0, 1, 15))
    weights_data_h = np.ones_like(all_shapes_h) / len(all_shapes_h)
    weights_data_muscle = np.ones_like(all_shapes_muscle) / len(all_shapes_muscle)
    plt.hist(all_shapes_h, color="blue", alpha=0.5, bins=bins, weights=weights_data_h)
    plt.hist(
        all_shapes_muscle,
        color="purple",
        alpha=0.5,
        bins=bins,
        weights=weights_data_muscle,
    )
    plt.axvline(
        sum(all_shapes_h) / len(all_shapes_h),
        color="blue",
        linestyle="dashed",
        linewidth=2,
    )
    plt.axvline(
        sum(all_shapes_muscle) / len(all_shapes_muscle),
        color="purple",
        linestyle="dashed",
        linewidth=2,
    )
    plt.xlim([0, 1])
    save_fig(os.path.join(exp_path, "movement", "neural_muscle_shape_dists"), eps=True)


def plot_angle_dists(model_name):
    exp_path = f"results/{model_name}/compositionality/alignment"

    all_subsets_h, _, all_metrics_h = gather_all_traj_metrics(model_name, "h")
    all_subsets_m, _, all_metrics_m = gather_all_traj_metrics(model_name, "muscle")

    angle_h_dist = convert_motif_dict_to_list(
        [key for key in all_subsets_h.keys()], all_metrics_h["angles"]
    )
    angle_muscle_dist = convert_motif_dict_to_list(
        [key for key in all_subsets_m.keys()], all_metrics_m["angles"]
    )

    bins = tuple(np.linspace(0, 1.5, 15))
    weights_data_h = np.ones_like(angle_h_dist) / len(angle_h_dist)
    weights_data_muscle = np.ones_like(angle_muscle_dist) / len(angle_muscle_dist)
    plt.hist(angle_h_dist, color="blue", alpha=0.5, bins=bins, weights=weights_data_h)
    plt.hist(
        angle_muscle_dist,
        color="purple",
        alpha=0.5,
        bins=bins,
        weights=weights_data_muscle,
    )
    plt.axvline(
        sum(angle_h_dist) / len(angle_h_dist),
        color="blue",
        linestyle="dashed",
        linewidth=2,
    )
    plt.axvline(
        sum(angle_muscle_dist) / len(angle_muscle_dist),
        color="purple",
        linestyle="dashed",
        linewidth=2,
    )
    plt.xlim([0, 1.5])
    save_fig(os.path.join(exp_path, "movement", "neural_muscle_angle_dists"), eps=True)


def plot_metric_scatter_h(model_name):
    plot_metric_scatter(model_name, "h")


def plot_metric_scatter_m(model_name):
    plot_metric_scatter(model_name, "muscle")


#################### Composite Input Experiments ###########################3


def composite_input_loss(model_name):
    exp_path = f"results/{model_name}/compositionality/composite_rule_inputs/losses"
    load_name = f"checkpoints/{model_name}/composite_rule_inputs.pkl"
    trial_data = load_pickle(load_name)
    colors_envs = plt.cm.tab10(np.linspace(0, 1, len(env_dict)))

    _, ax = standard_2d_ax()
    for e, env in enumerate(trial_data):
        loss = trial_data[env]["test_loss"]
        ax.plot(loss, linewidth=4, color=colors_envs[e], alpha=0.75)
    save_fig(os.path.join(exp_path, "optimization_losses"), eps=True)


def run_composite_input_optimization(model_name):
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    options = {
        "batch_size": 8,
        "reach_conds": np.arange(0, 32, 4),
        "speed_cond": 9,
        "custom_delay": 150,
    }

    all_trial_data = {}
    for env in extension_dict:
        trial_data = composite_input_optimization(
            model_path, model_file, options, extension_dict[env], env
        )
        all_trial_data[env] = trial_data

    # Save all information of inputs across envs
    save_name = "composite_rule_inputs.pkl"
    fname = os.path.join(model_path, save_name)
    with open(fname, "wb") as f:
        pickle.dump(all_trial_data, f)


# Get the loss from every composite input on each environment and get the heat map
def composite_rule_input_heat_map(model_name):
    exp_path = f"results/{model_name}/compositionality/composite_rule_inputs/heat_map"
    load_name = f"checkpoints/{model_name}/composite_rule_inputs.pkl"
    trial_data = load_pickle(load_name)

    for env in extension_dict:
        fig, ax = no_ticks_2d_ax()
        rule_input = trial_data[env]["rule_input"][:, :5].numpy()
        im = ax.imshow(rule_input, cmap="RdBu", vmin=2, vmax=-2)
        _ = fig.colorbar(im, ax=ax, fraction=0.07, pad=0.04)
        fig.tight_layout()
        save_fig(os.path.join(exp_path, f"extension_heat_map_{env}"), eps=True)


# Get the loss from every composite input on each environment and get the heat map
def composite_rule_input_kinematics(model_name):
    def plot_env_kinematics(xy):
        _, ax = empty_2d_ax()
        for i, batch in enumerate(xy):
            ax.plot(batch[:, 0], batch[:, 1], linewidth=4, color=colors[i])
            ax.scatter(batch[0, 0], batch[0, 1], s=250, marker="^", color=colors[i])
            ax.scatter(batch[-1, 0], batch[-1, 1], s=250, marker="X", color=colors[i])

    exp_path = f"results/{model_name}/compositionality/composite_rule_inputs/kinematics"
    load_name = f"checkpoints/{model_name}/composite_rule_inputs.pkl"
    trial_data = load_pickle(load_name)
    colors = plt.cm.inferno(np.linspace(0, 1, 8))

    for env in env_dict_ext:
        plot_env_kinematics(trial_data[env]["xy"])
        save_fig(os.path.join(exp_path, f"extension_kinematics_{env}"), eps=True)


def composite_input_init(model_name):
    exp_path = f"results/{model_name}/compositionality/composite_rule_inputs/init_cond"
    load_name = f"checkpoints/{model_name}/composite_rule_inputs.pkl"
    trial_data = load_pickle(load_name)
    colors_envs = plt.cm.tab10(np.linspace(0, 1, len(env_dict)))

    env_hs, _ = _get_mean_act(
        model_name, "delay", "extension", delay_cond=2, batch_size=32
    )
    env_hs = np.concatenate(env_hs)
    epoch_pca = PCA(n_components=3)
    epoch_pca.fit(env_hs)

    _, ax = ax_3d_no_grid()
    for e, env in enumerate(env_dict_ext):
        delay_start, delay_end = delay_bounds(trial_data)

        composite_h = trial_data[env]["h"][:, delay_start:delay_end]
        composite_h = composite_h.mean(dim=0)

        all_data_for_min = np.concatenate([composite_h, env_hs])
        red_all_data_for_min = epoch_pca.transform(all_data_for_min)
        min_val = np.min(red_all_data_for_min)

        reduced_baseline = epoch_pca.transform(env_hs)
        ax.scatter(
            reduced_baseline[e, 0],
            reduced_baseline[e, 1],
            reduced_baseline[e, 2],
            s=200,
            marker="o",
            color=colors_envs[e],
        )
        ax.scatter(
            reduced_baseline[e, 0],
            reduced_baseline[e, 1],
            min_val,
            s=200,
            marker="o",
            color=colors_envs[e],
            alpha=0.10,
        )

        reduced_composite = epoch_pca.transform(composite_h)
        ax.scatter(
            reduced_composite[-1, 0],
            reduced_composite[-1, 1],
            min_val,
            s=200,
            marker="X",
            color=colors_envs[e],
            alpha=0.10,
        )
        ax.scatter(
            reduced_composite[-1, 0],
            reduced_composite[-1, 1],
            reduced_composite[-1, 2],
            s=200,
            marker="X",
            color=colors_envs[e],
        )
    save_fig(os.path.join(exp_path, "extension_init_all"), eps=True)


def run_all_sequential_rule_inputs(model_name):
    for ext in extension_dict:
        for ret in retraction_dict:
            _sequential_rule_inputs(model_name, ext, ret)


if __name__ == "__main__":
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    # Epoch pcs
    if args.experiment == "stable_pcs_extension":
        stable_pcs_extension(args.model_name)
    elif args.experiment == "delay_pcs_extension":
        delay_pcs_extension(args.model_name)
    elif args.experiment == "movement_pcs_extension":
        movement_pcs_extension(args.model_name)
    elif args.experiment == "hold_pcs_extension":
        hold_pcs_extension(args.model_name)

    elif args.experiment == "stable_pcs_extension_retraction":
        stable_pcs_extension_retraction(args.model_name)
    elif args.experiment == "delay_pcs_extension_retraction":
        delay_pcs_extension_retraction(args.model_name)
    elif args.experiment == "extension_pcs_extension_retraction":
        extension_pcs_extension_retraction(args.model_name)
    elif args.experiment == "retraction_pcs_extension_retraction":
        retraction_pcs_extension_retraction(args.model_name)
    elif args.experiment == "hold_pcs_extension_retraction":
        hold_pcs_extension_retraction(args.model_name)

    elif args.experiment == "neural_two_task_pcs_sinusoid_fullcircleclk":
        neural_two_task_pcs_sinusoid_fullcircleclk(args.model_name)
    elif args.experiment == "muscle_two_task_pcs_sinusoid_fullcircleclk":
        muscle_two_task_pcs_sinusoid_fullcircleclk(args.model_name)
    elif args.experiment == "neural_two_task_pcs_halfcircleclk_halfcirclecclk":
        neural_two_task_pcs_halfcircleclk_halfcirclecclk(args.model_name)
    elif args.experiment == "muscle_two_task_pcs_halfcircleclk_halfcirclecclk":
        muscle_two_task_pcs_halfcircleclk_halfcirclecclk(args.model_name)

    elif args.experiment == "run_all_epoch_pcs":
        run_all_epoch_pcs(args.model_name)
    elif args.experiment == "run_all_epoch_pcs_transfer":
        run_all_epoch_pcs_transfer(args.model_name)
    elif args.experiment == "run_all_epoch_lda":
        run_all_epoch_lda(args.model_name)
    elif args.experiment == "run_all_epoch_lda_transfer":
        run_all_epoch_lda_transfer(args.model_name)

    elif args.experiment == "task_vaf_ratio":
        task_vaf_ratio(args.model_name)

    elif args.experiment == "dsa_similarity_matrix":
        dsa_similarity_matrix(args.model_name)
    elif args.experiment == "dsa_scatter":
        dsa_scatter(args.model_name)
    elif args.experiment == "dsa_heatmap":
        dsa_heatmap(args.model_name)

    elif args.experiment == "procrustes_similarity_matrix":
        procrustes_similarity_matrix(args.model_name)
    elif args.experiment == "procrustes_scatter":
        procrustes_scatter(args.model_name)
    elif args.experiment == "procrustes_heatmap":
        procrustes_heatmap(args.model_name)

    elif args.experiment == "task_similarity_classification":
        task_similarity_classification(args.model_name)

    elif args.experiment == "composite_rule_input_heat_map":
        composite_rule_input_heat_map(args.model_name)
    elif args.experiment == "composite_rule_input_kinematics":
        composite_rule_input_kinematics(args.model_name)
    elif args.experiment == "composite_input_init":
        composite_input_init(args.model_name)
    elif args.experiment == "composite_input_loss":
        composite_input_loss(args.model_name)
    elif args.experiment == "run_composite_input_optimization":
        run_composite_input_optimization(args.model_name)

    elif args.experiment == "run_all_sequential_rule_inputs":
        run_all_sequential_rule_inputs(args.model_name)

    else:
        raise ValueError("Experiment not in this file")
