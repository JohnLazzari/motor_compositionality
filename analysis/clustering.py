"""
Clustering analysis
Analyze how units are involved in various tasks
"""

from __future__ import division

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import utils

os.environ["OMP_NUM_THREADS"] = '1'

# Colors used for clusters
kelly_colors = \
[np.array([ 0.94901961,  0.95294118,  0.95686275]),
np.array([ 0.13333333,  0.13333333,  0.13333333]),
np.array([ 0.95294118,  0.76470588,  0.        ]),
np.array([ 0.52941176,  0.3372549 ,  0.57254902]),
np.array([ 0.95294118,  0.51764706,  0.        ]),
np.array([ 0.63137255,  0.79215686,  0.94509804]),
np.array([ 0.74509804,  0.        ,  0.19607843]),
np.array([ 0.76078431,  0.69803922,  0.50196078]),
np.array([ 0.51764706,  0.51764706,  0.50980392]),
np.array([ 0.        ,  0.53333333,  0.3372549 ]),
np.array([ 0.90196078,  0.56078431,  0.6745098 ]),
np.array([ 0.        ,  0.40392157,  0.64705882]),
np.array([ 0.97647059,  0.57647059,  0.4745098 ]),
np.array([ 0.37647059,  0.30588235,  0.59215686]),
np.array([ 0.96470588,  0.65098039,  0.        ]),
np.array([ 0.70196078,  0.26666667,  0.42352941]),
np.array([ 0.8627451 ,  0.82745098,  0.        ]),
np.array([ 0.53333333,  0.17647059,  0.09019608]),
np.array([ 0.55294118,  0.71372549,  0.        ]),
np.array([ 0.39607843,  0.27058824,  0.13333333])]


save = True


class Analysis(object):
    def __init__(self, model_dir, data_type, normalization_method='max'):
        hp = utils.load_hp(model_dir)

        # If not computed, use variance.py
        fname = os.path.join(model_dir, 'variance_' + data_type + '.pkl')
        res = utils.load_pickle(fname)
        h_var_all_ = res['h_var_all']
        self.keys  = res['keys']

        # First only get active units. Total variance across tasks larger than 1e-3
        # ind_active = np.where(h_var_all_.sum(axis=1) > 1e-2)[0]
        ind_active = np.where(h_var_all_.sum(axis=1) > 1e-3)[0]
        h_var_all  = h_var_all_[ind_active, :]
        print(h_var_all.shape)

        # Normalize by the total variance across tasks
        if normalization_method == 'sum':
            h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T
        elif normalization_method == 'max':
            h_normvar_all = (h_var_all.T/np.max(h_var_all, axis=1)).T
        elif normalization_method == 'none':
            h_normvar_all = h_var_all
        else:
            raise NotImplementedError()

        ################################## Clustering ################################
        from sklearn import metrics
        X = h_normvar_all

        # Clustering
        from sklearn.cluster import AgglomerativeClustering, KMeans

        # Choose number of clusters that maximize silhouette score
        n_clusters = range(2, 20)
        scores = list()
        labels_list = list()
        for n_cluster in n_clusters:
            # clustering = AgglomerativeClustering(n_cluster, affinity='cosine', linkage='average')
            clustering = KMeans(n_cluster, n_init=100, max_iter=1000, random_state=0)
            clustering.fit(X) # n_samples, n_features = n_units, n_rules/n_epochs
            labels = clustering.labels_ # cluster labels

            score = metrics.silhouette_score(X, labels)

            scores.append(score)
            labels_list.append(labels)

        scores = np.array(scores)

        # Heuristic elbow method
        # Choose the number of cluster when Silhouette score first falls
        # Choose the number of cluster when Silhouette score is maximum
        if data_type == 'rule':
            #i = np.where((scores[1:]-scores[:-1])<0)[0][0]
            i = np.argmax(scores)
        else:
            # The more rigorous method doesn't work well in this case
            i = n_clusters.index(10)
            #i = np.argmax(scores)

        labels = labels_list[i]
        n_cluster = n_clusters[i]
        print('Choosing {:d} clusters'.format(n_cluster))

        # Sort clusters by its task preference (important for consistency across nets)
        if data_type == 'rule':
            label_prefs = [np.argmax(h_normvar_all[labels==l].sum(axis=0)) for l in set(labels)]
        elif data_type == 'epoch':
            ## TODO: this may no longer work!
            label_prefs = [self.keys[np.argmax(h_normvar_all[labels==l].sum(axis=0))][0] for l in set(labels)]

        ind_label_sort = np.argsort(label_prefs)
        label_prefs = np.array(label_prefs)[ind_label_sort]
        # Relabel
        labels2 = np.zeros_like(labels)
        for i, ind in enumerate(ind_label_sort):
            labels2[labels==ind] = i
        labels = labels2

        ind_sort = np.argsort(labels)

        labels          = labels[ind_sort]
        self.h_normvar_all   = h_normvar_all[ind_sort, :]
        self.ind_active      = ind_active[ind_sort]

        self.n_clusters = n_clusters
        self.scores = scores
        self.n_cluster = n_cluster

        self.h_var_all = h_var_all
        self.normalization_method = normalization_method
        self.labels = labels
        self.unique_labels = np.unique(labels)

        self.model_dir = model_dir
        self.hp = hp
        self.data_type = data_type

    def plot_cluster_score(self, save_name):
        """Plot the score by the number of clusters."""
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0.3, 0.3, 0.55, 0.55])
        ax.plot(self.n_clusters, self.scores, 'o-', ms=3)
        ax.set_xlabel('Number of clusters', fontsize=7)
        ax.set_ylabel('Silhouette score', fontsize=7)
        ax.set_title('Chosen number of clusters: {:d}'.format(self.n_cluster),
                     fontsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_ylim([0, 0.32])
        if save:
            fig_name = 'cluster_score'
            if save_name is None:
                save_name = self.hp['activation']
            fig_name = fig_name + save_name
            plt.savefig(save_name)
            plt.close()

    def plot_variance(self, save_path):
        labels = self.labels
        ######################### Plotting Variance ###################################
        # Plot Normalized Variance
        if self.data_type == 'rule':
            figsize = (3.5,2.5)
            rect = [0.25, 0.2, 0.6, 0.7]
            rect_color = [0.25, 0.15, 0.6, 0.05]
            rect_cb = [0.87, 0.2, 0.03, 0.7]
            tick_names = [key for key in self.keys]
            fs = 6
            labelpad = 13
        elif self.data_type == 'epoch':
            figsize = (3.5,4.5)
            rect = [0.25, 0.1, 0.6, 0.85]
            rect_color = [0.25, 0.05, 0.6, 0.05]
            rect_cb = [0.87, 0.1, 0.03, 0.85]
            tick_names = [key for key in self.keys]
            fs = 5
            labelpad = 20
        else:
            raise ValueError

        h_plot  = self.h_normvar_all.T
        vmin, vmax = 0, 1
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        im = ax.imshow(h_plot, cmap='hot',
                       aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

        plt.yticks(range(len(tick_names)), tick_names,
                   rotation=0, va='center', fontsize=fs)
        plt.xticks([])
        plt.title('Units', fontsize=7, y=0.97)
        plt.xlabel('Clusters', fontsize=7, labelpad=labelpad)
        ax.tick_params('both', length=0)
        for loc in ['bottom','top','left','right']:
            ax.spines[loc].set_visible(False)
        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(im, cax=ax, ticks=[vmin,vmax])
        cb.outline.set_linewidth(0.5)
        if self.normalization_method == 'sum':
            clabel = 'Normalized Task Variance'
        elif self.normalization_method == 'max':
            clabel = 'Normalized Task Variance'
        elif self.normalization_method == 'none':
            clabel = 'Variance'

        cb.set_label(clabel, fontsize=7, labelpad=0)
        plt.tick_params(axis='both', which='major', labelsize=7)
        

        # Plot color bars indicating clustering
        if True:
            ax = fig.add_axes(rect_color)
            for il, l in enumerate(self.unique_labels):
                ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
                ax.plot(ind_l, [0,0], linewidth=4, solid_capstyle='butt',
                        color=kelly_colors[il+1])
                ax.text(np.mean(ind_l), -0.5, str(il+1), fontsize=6,
                        ha='center', va='top', color=kelly_colors[il+1])
            ax.set_xlim([0, len(labels)])
            ax.set_ylim([-1, 1])
            ax.axis('off')

        # Get the directory part of the save path
        directory = os.path.dirname(save_path)
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path)

    def plot_similarity_matrix(self):
        labels = self.labels
        ######################### Plotting Similarity Matrix ##########################

        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(self.h_normvar_all) # TODO: check
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
        im = ax.imshow(similarity, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        ax.axis('off')

        ax = fig.add_axes([0.87, 0.25, 0.03, 0.6])
        cb = plt.colorbar(im, cax=ax, ticks=[0,1])
        cb.outline.set_linewidth(0.5)
        cb.set_label('Similarity', fontsize=7, labelpad=0)
        plt.tick_params(axis='both', which='major', labelsize=7)

        ax1 = fig.add_axes([0.25, 0.85, 0.6, 0.05])
        ax2 = fig.add_axes([0.2, 0.25, 0.05, 0.6])
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
            ax1.plot(ind_l, [0,0], linewidth=2, solid_capstyle='butt',
                    color=kelly_colors[il+1])
            ax2.plot([0,0], len(labels)-ind_l, linewidth=2, solid_capstyle='butt',
                    color=kelly_colors[il+1])
        ax1.set_xlim([0, len(labels)])
        ax2.set_ylim([0, len(labels)])
        ax1.axis('off')
        ax2.axis('off')
        if save:
            plt.savefig('figure/feature_similarity_by'+self.data_type+'.pdf', transparent=True)
        plt.show()

    def plot_2Dvisualization(self, save_path, method='tSNE'):
        labels = self.labels
        ######################## Plotting 2-D visualization of variance map ###########
        from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
        from sklearn.decomposition import PCA

        # model = LocallyLinearEmbedding()
        if method == 'PCA':
            model = PCA(n_components=2, whiten=False)
        elif method == 'MDS':
            model = MDS(metric=True, n_components=2, n_init=10, max_iter=1000)
        elif method == 'tSNE':
            model = TSNE(n_components=2, random_state=0, init='pca',
                            verbose=1, method='exact',
                            learning_rate=100, perplexity=30)
        else:
            raise NotImplementedError

        print(self.h_normvar_all.shape)
        Y = model.fit_transform(self.h_normvar_all)

        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels==l)[0]
            ax.scatter(Y[ind_l, 0], Y[ind_l, 1], color=kelly_colors[il+1], s=10)
        ax.axis('off')
        plt.title(method, fontsize=7)
        # Get the directory part of the save path
        directory = os.path.dirname(save_path)
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path)

        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.scatter(Y[:,0], Y[:,1], color='black')
        ax.axis('off')