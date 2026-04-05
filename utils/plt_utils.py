import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib import rcParams


def create_dir(save_path):
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def save_fig(save_path, eps=False):
    # Tell matplotlib to embed fonts as text, not outlines
    rcParams["pdf.fonttype"] = 42  # 42 = TrueType (editable in Illustrator)
    rcParams["ps.fonttype"] = 42
    # Simple function to save figure while creating dir and closing
    dir = os.path.dirname(save_path)
    create_dir(dir)
    plt.tight_layout()
    if eps:
        plt.savefig(save_path + ".pdf", format="pdf")
    else:
        plt.savefig(save_path + ".png")
    plt.close()


def standard_2d_ax(w=4, h=4):
    # Create figure and 3D axes
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def no_ticks_2d_ax():
    fig, ax = standard_2d_ax()
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def empty_2d_ax():
    fig, ax = standard_2d_ax()
    ax.set_axis_off()  # hides axes, ticks, labels, etc.
    return fig, ax


def ax_3d_no_grid():
    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return fig, ax


def empty_3d():
    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()  # hides axes, ticks, labels, etc.
    return fig, ax
