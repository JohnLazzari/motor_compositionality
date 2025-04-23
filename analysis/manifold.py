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




def principal_angles(combinations, combination_labels):
    """
        Perform manifold analysis (principle angles and VAF)

        params:
            system: "neural" or "muscle"
            epoch: "delay" or "movement"
    """
    angles_dict = {}

    for i, combination in enumerate(combinations):
        
        # ------------------------------------ GET PRINCIPLE ANGLES

        pca1 = PCA()
        pca2 = PCA()

        task1_data = combination[0].reshape((-1, combination[0].shape[-1])).numpy()
        task2_data = combination[1].reshape((-1, combination[1].shape[-1])).numpy()

        pca1.fit(task1_data)
        pca2.fit(task2_data)

        pca1_comps = pca1.components_[:12]
        pca2_comps = pca2.components_[:12]

        # Get principle angles
        inner_prod_mat = pca1_comps @ pca2_comps.T # Should be m x m
        U, s, Vh = np.linalg.svd(inner_prod_mat)
        angles = np.degrees(np.arccos(s))
        angles_dict[combination_labels[i]] = angles

    # ------------------------------------ PLOTTING

    return angles_dict    



def vaf_ratio(combinations, hid_size):

    vaf_ratio_list = []
    vaf_ratio_list_control = []

    """
    # Create a random manifold as a control
    random_bases = np.empty(shape=(5000, 256, 256))
    for basis in range(5000):
        random_bases[basis] = random_orthonormal_basis(hid_size).T
    """

    for combination in combinations:
        
        pca1 = PCA()
        pca2 = PCA()

        task1_data = combination[0].reshape((-1, combination[0].shape[-1])).numpy()
        task2_data = combination[1].reshape((-1, combination[1].shape[-1])).numpy()

        pca1.fit(task1_data)
        pca2.fit(task2_data)

        pca1_comps = pca1.components_[:12]
        pca2_comps = pca2.components_[:12]

        #------------------------------------- PUT TO DEVICES 

        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        task1_data = torch.tensor(task1_data, dtype=torch.float32).to(device)
        task2_data = torch.tensor(task2_data, dtype=torch.float32).to(device)

        pca1_comps = torch.tensor(pca1_comps, dtype=torch.float32).to(device)
        pca2_comps = torch.tensor(pca2_comps, dtype=torch.float32).to(device)
        """

        #random_bases = torch.tensor(random_bases, dtype=torch.float32).to(device)

        # ------------------------------------ TRUE ACROSS AND WITHIN TASK VAFs

        # Get VAF
        across_task_vaf_task1 = (pca2_comps @ task1_data.T).T.var(axis=0).sum() / task1_data.var(axis=0).sum()
        within_task_vaf_task1 = (pca1_comps @ task1_data.T).T.var(axis=0).sum() / task1_data.var(axis=0).sum()
        ratio_task1 = across_task_vaf_task1 / within_task_vaf_task1
        vaf_ratio_list.append(ratio_task1)

        across_task_vaf_task2 = (pca1_comps @ task2_data.T).T.var(axis=0).sum() / task2_data.var(axis=0).sum()
        within_task_vaf_task2 = (pca2_comps @ task2_data.T).T.var(axis=0).sum() / task2_data.var(axis=0).sum()
        ratio_task2 = across_task_vaf_task2 / within_task_vaf_task2
        vaf_ratio_list.append(ratio_task2)

        # ------------------------------------ CONTROL ACROSS TASK VAFs

        """
        # Get random VAFs
        across_task_vaf = (random_bases[:, :12, :] @ task1_data.T).var(axis=2).sum(axis=1) / task1_data.var(axis=0).sum()
        vaf_ratio_list_control.append(np.percentile(across_task_vaf, 90) / within_task_vaf_task1)

        # Get random VAFs
        across_task_vaf = (random_bases[:, :12, :] @ task2_data.T).var(axis=2).sum(axis=1) / task2_data.var(axis=0).sum()
        vaf_ratio_list_control.append(np.percentile(across_task_vaf, 90) / within_task_vaf_task2)
        """

    return vaf_ratio_list