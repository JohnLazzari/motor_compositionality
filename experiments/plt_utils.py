import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils import load_hp, interpolate_trial

import warnings
warnings.filterwarnings("ignore")

import math
import motornet as mn
from model import RNNPolicy, GRUPolicy
import torch
import os
from utils import load_hp, create_dir, save_fig, load_pickle, standard_2d_ax, pvalues
import matplotlib.pyplot as plt
import numpy as np
import config
import pickle
from analysis.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
import analysis.plot_utils as plot_utils
from analysis.manifold import principal_angles, vaf_ratio
import tqdm as tqdm
from sklearn.decomposition import PCA, NMF
import sklearn 
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.patches as mpatches
from exp_utils import _test, env_dict, split_movement_epoch, get_interpolation_input
from exp_utils import distances_from_combinations, angles_from_combinations, shapes_from_combinations, convert_motif_dict_to_list
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import seaborn as sns
from DSA import DSA
import scipy
from utils import interpolate_trial
import pandas as pd

def standard_2d_ax():
    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
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