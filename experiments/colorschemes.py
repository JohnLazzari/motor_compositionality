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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import seaborn as sns
import scipy
from utils import interpolate_trial
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def directions():
    exp_path = "results/colorschemes"

    colors = plt.cm.inferno(np.linspace(0, 1, 8)) 

    # Set up figure
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Draw 8 lines at 45° intervals (0 to 360°)
    for i in range(8):
        angle = i * (np.pi / 4)  # 45° = π/4 radians
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot([0, x], [0, y], linewidth=6, color=colors[i], alpha=0.75)

    # Set limits and grid
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(False)
    ax.axis('off')  # removes ticks, labels, and spines
    save_fig(os.path.join(exp_path, "direction"), eps=True)
    

def speeds():

    exp_path = "results/colorschemes"

    colors = plt.cm.plasma(np.linspace(0, 1, 5)) 

    # Set up figure
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Draw 10 horizontal lines at y = 0.1, 0.2, ..., 1.0
    for i in range(5):
        y = 0.1 * (i + 1)
        ax.plot([0, 1], [y, y], linewidth=6, color=colors[4-i], alpha=0.75)

    # Set limits and grid
    ax.grid(False)
    ax.axis('off')  # removes ticks, labels, and spines

    save_fig(os.path.join(exp_path, "speed"), eps=True)

def interpolation():

    exp_path = "results/colorschemes"

    # Create figure
    fig = plt.figure(figsize=(1.5, 5))

    # Add a dedicated axis for the colorbar
    cbar_ax = fig.add_axes([0.4, 0.05, 0.2, 0.9])  # [left, bottom, width, height]

    # Create a ScalarMappable for the colorbar
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap='magma')
    sm.set_array([])

    # Create colorbar on specified axis
    cbar = fig.colorbar(sm, cax=cbar_ax)

    # Remove ticks and outline
    cbar.ax.tick_params(left=False, right=False, labelleft=False)
    cbar.outline.set_visible(False)

    save_fig(os.path.join(exp_path, "interpolation"), eps=True)

if __name__ == "__main__":
    directions()
    speeds()
    interpolation()