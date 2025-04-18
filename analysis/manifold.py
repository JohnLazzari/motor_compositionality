from train import train_2link
import motornet as mn
from model import RNNPolicy, GRUPolicy
import torch
import os
from utils import load_hp, create_dir, save_fig, load_pickle, interpolate_trial, random_orthonormal_basis
from envs import DlyHalfReach, DlyHalfCircleClk, DlyHalfCircleCClk, DlySinusoid, DlySinusoidInv
from envs import DlyFullReach, DlyFullCircleClk, DlyFullCircleCClk, DlyFigure8, DlyFigure8Inv
import matplotlib.pyplot as plt
import numpy as np
import config
from analysis.clustering import Analysis
import pickle
from analysis.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
import analysis.plot_utils as plot_utils
#from analysis.dPCA import dPCA
import tqdm as tqdm
import itertools
from sklearn.decomposition import PCA
from losses import l1_dist
import scipy
from mRNNTorch.analysis import flow_field
import warnings
warnings.filterwarnings("ignore")




def principal_angles(combinations):
    """
        Perform manifold analysis (principle angles and VAF)

        params:
            system: "neural" or "muscle"
            epoch: "delay" or "movement"
    """
    angles_dict = {}

    for combination in combinations:
        
        # ------------------------------------ GET PRINCIPLE ANGLES

        pca1 = PCA()
        pca2 = PCA()

        task1_data = trial_data_envs[combination[0]].reshape((-1, trial_data_envs[combination[0]].shape[-1])).numpy()
        task2_data = trial_data_envs[combination[1]].reshape((-1, trial_data_envs[combination[1]].shape[-1])).numpy()

        pca1.fit(task1_data)
        pca2.fit(task2_data)

        pca1_comps = pca1.components_[:12]
        pca2_comps = pca2.components_[:12]

        # Get principle angles
        inner_prod_mat = pca1_comps @ pca2_comps.T # Should be m x m
        U, s, Vh = np.linalg.svd(inner_prod_mat)
        angles = np.degrees(np.arccos(s))
        angles_dict[combination] = angles

    # ------------------------------------ PLOTTING

    return angles_dict    



def vaf_ratio(combinations):

    vaf_ratio_list = []
    vaf_ratio_list_control = []

    # Create a random manifold as a control
    random_bases = np.empty(shape=(5000, 256, 256))
    for basis in range(5000):
        random_bases[basis] = random_orthonormal_basis(hp["hid_size"]).T

    for combination in combinations:
        
        pca1 = PCA()
        pca2 = PCA()

        task1_data = trial_data_envs[combination[0]].reshape((-1, trial_data_envs[combination[0]].shape[-1])).numpy()
        task2_data = trial_data_envs[combination[1]].reshape((-1, trial_data_envs[combination[1]].shape[-1])).numpy()

        pca1.fit(task1_data)
        pca2.fit(task2_data)

        pca1_comps = pca1.components_[:12]
        pca2_comps = pca2.components_[:12]

        # ------------------------------------ TRUE ACROSS AND WITHIN TASK VAFs

        # Get VAF
        across_task_vaf_task1 = (pca2_comps @ task1_data.T).T.var() / task1_data.var()
        within_task_vaf_task1 = (pca1_comps @ task1_data.T).T.var() / task1_data.var()
        ratio_task1 = across_task_vaf_task1 / within_task_vaf_task1
        vaf_ratio_list.append(ratio_task1)

        across_task_vaf_task2 = (pca1_comps @ task2_data.T).T.var() / task2_data.var()
        within_task_vaf_task2 = (pca2_comps @ task2_data.T).T.var() / task2_data.var()
        ratio_task2 = across_task_vaf_task2 / within_task_vaf_task2
        vaf_ratio_list.append(ratio_task2)

        # ------------------------------------ CONTROL ACROSS TASK VAFs

        task1_controls = []
        # Get random VAFs
        across_task_vaf = (random_bases[:, :12, :] @ task1_data.T).var(axis=(1, 2)) / task1_data.var()
        task1_controls.append(across_task_vaf)
        vaf_ratio_list_control.append(np.percentile(task1_controls, 90) / within_task_vaf_task1)

        task2_controls = []
        # Get random VAFs
        across_task_vaf = (random_bases[:, :12, :] @ task2_data.T).var(axis=(1, 2)) / task2_data.var()
        task2_controls.append(across_task_vaf)
        vaf_ratio_list_control.append(np.percentile(task2_controls, 90) / within_task_vaf_task2)
    
    return vaf_ratio_list, vaf_ratio_list_control