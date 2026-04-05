'''
plot_utils.py
Supports FixedPointFinder
Written for Python 3.8.17
@ Matt Golub, October 2018
Please direct correspondence to mgolub@cs.washington.edu
'''

import numpy as np
import pdb

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_fps(fps,
    dims=2,
    pca_traj=None,
    state_traj=None,
    plot_batch_idx=None,
    plot_start_time=0,
    plot_stop_time=None,
    mode_scale=0.25,
    traj_color="black",
    stable_color="black",
    marker="o",
    fig=None):

    '''Plots a visualization and analysis of the unique fixed points.

    1) Finds a low-dimensional subspace for visualization via PCA. If
    state_traj is provided, PCA is fit to [all of] those RNN state
    trajectories. Otherwise, PCA is fit to the identified unique fixed
    points. This subspace is 3-dimensional if the RNN state dimensionality
    is >= 3.

    2) Plots the PCA representation of the stable unique fixed points as
    black dots.

    3) Plots the PCA representation of the unstable unique fixed points as
    red dots.

    4) Plots the PCA representation of the modes of the Jacobian at each
    fixed point. By default, only unstable modes are plotted.

    5) (optional) Plots example RNN state trajectories as blue lines.

    Args:
        fps: a FixedPoints object. See FixedPoints.py.

        state_traj (optional): [n_batch x n_time x n_states] numpy
        array or LSTMStateTuple with .c and .h as
        [n_batch x n_time x n_states/2] numpy arrays. Contains example
        trials of RNN state trajectories.

        plot_batch_idx (optional): Indices specifying which trials in
        state_traj to plot on top of the fixed points. Default: plot all
        trials.

        plot_start_time (optional): int specifying the first timestep to
        plot in the example trials of state_traj. Default: 0.

        plot_stop_time (optional): int specifying the last timestep to
        plot in the example trials of stat_traj. Default: n_time.

        stop_time (optional):

        mode_scale (optional): Non-negative float specifying the scaling
        of the plotted eigenmodes. A value of 1.0 results in each mode
        plotted as a set of diametrically opposed line segments
        originating at a fixed point, with each segment's length specified
        by the magnitude of the corresponding eigenvalue.

        fig (optional): Matplotlib figure upon which to plot.

    Returns:
        None.
    '''

    FONT_WEIGHT = 'bold'
    if fig is None:
        FIG_WIDTH = 6 # inches
        FIG_HEIGHT = 6 # inches
        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT),
            tight_layout=True)

    if pca_traj is not None:
        
        pca_traj_bxtxd = pca_traj
        [n_batch, n_time, n_states] = pca_traj_bxtxd.shape

        # Ensure plot_start_time >= 0
        plot_start_time = np.max([plot_start_time, 0])

        if plot_stop_time is None:
            plot_stop_time = n_time
        else:
            # Ensure plot_stop_time <= n_time
            plot_stop_time = np.min([plot_stop_time, n_time])

        plot_time_idx = list(range(plot_start_time, plot_stop_time))

    n_inits = fps.n
    n_states = fps.n_states

    if n_states >= 3:

        pca = PCA(n_components=dims)

        if pca_traj is not None:
            pca_traj_btxd = np.reshape(pca_traj_bxtxd,
                (n_batch*n_time, n_states))
            pca.fit(pca_traj_btxd)
        else:
            pca.fit(fps.xstar)

        # For generating figure in paper.md
        #ax.set_xticks([-2, -1, 0, 1, 2])
        #ax.set_yticks([-1, 0, 1])
    else:
        # For 1D or 0D networks (i.e., never)
        pca = None

    if pca_traj is not None and state_traj is not None:

        state_traj_bxtxd = state_traj
        [n_batch_s, n_time_s, n_states_s] = state_traj_bxtxd.shape

        if plot_batch_idx is None:
            plot_batch_idx = list(range(n_batch_s))

        for batch_idx in plot_batch_idx:
            x_idx = state_traj[batch_idx]

            if n_states >= 3:
                z_idx = pca.transform(x_idx[plot_time_idx, :])
            else:
                z_idx = x_idx[plot_time_idx, :]
            
            plot_123d(z_idx, color=traj_color, linewidth=4)
            plt.scatter(z_idx[0, 0], z_idx[0, 1], marker="^", color=traj_color, s=250, zorder=10)
            plt.scatter(z_idx[-1, 0], z_idx[-1, 1], marker="X", color=traj_color, s=250, zorder=10)

    for init_idx in range(n_inits):
        plot_fixed_point(
            fps[init_idx],
            pca,
            stable_marker=marker,
            stable_color=stable_color,
            scale=mode_scale,
            alpha=0.5)

    return fig

def plot_fixed_point(fp, pca,
    scale=1.0,
    max_n_modes=3,
    do_plot_unstable_fps=True,
    do_plot_stable_modes=False, # (for unstable FPs)
    stable_color='k',
    stable_marker='.',
    unstable_color='w',
    unstable_marker=None,
    make_plot=True,
    **kwargs):
    '''Plots a single fixed point and its dominant eigenmodes.

    Args:
        ax: Matplotlib figure axis on which to plot everything.

        fp: a FixedPoints object containing a single fixed point
        (i.e., fp.n == 1),

        pca: PCA object as returned by sklearn.decomposition.PCA. This
        is used to transform the high-d state space representations
        into 3-d for visualization.

        scale (optional): Scale factor for stretching (>1) or shrinking
        (<1) lines representing eigenmodes of the Jacobian. Default:
        1.0 (unity).

        max_n_modes (optional): Maximum number of eigenmodes to plot.
        Default: 3.

        do_plot_stable_modes (optional): bool indicating whether or
        not to plot lines representing stable modes (i.e.,
        eigenvectors of the Jacobian whose eigenvalue magnitude is
        less than one).

    Returns:
        None.
    '''

    xstar = fp.xstar
    J = fp.J_xstar
    n_states = fp.n_states

    has_J = J is not None

    do_plot = (not has_J) or do_plot_unstable_fps

    if do_plot:

        if n_states >= 3 and pca is not None:
            zstar = pca.transform(xstar)
        else:
            zstar = xstar

        if make_plot:
            plot_123d(zstar,
                    color=color,
                    marker=marker,
                    markersize=12,
                    **kwargs)
        else:
            return zstar

def plot_123d(z, **kwargs):
    '''Plots in 1D, 2D, or 3D.

    Args:
        ax: Matplotlib figure axis on which to plot everything.

        z: [n x n_states] numpy array containing data to be plotted,
        where n_states is 1, 2, or 3.

        any keyword arguments that can be passed to ax.plot(...).

    Returns:
        None.
    '''
    n_states = z.shape[1]
    if n_states ==3:
        plt.plot(z[:, 0], z[:, 1], z[:, 2], **kwargs)
    elif n_states == 2:
        plt.plot(z[:, 0], z[:, 1], **kwargs)
    elif n_states == 1:
        plt.plot(z, **kwargs)