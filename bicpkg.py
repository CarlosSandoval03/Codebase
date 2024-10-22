import os
from pathlib import Path
import numpy as np
from scipy.spatial import distance
from sklearn import cluster
import plotly.graph_objects as go


class BICAnalysis():
    def __init__(self, data):
        print(" --- CREATING INSTANCE OF BIC ANALYSIS FOR CLUSTERING --- ")
        self.data = data

    def __compute_bic(self, kmeans, X):
        """
        Computes the BIC metric for a given clusters. This is from StackOverFlow, need to find the link again.
        Parameters:
        -----------------------------------------
        kmeans:  List of clustering object from scikit learn
        X     :  multidimension np array of data points
        Returns:
        -----------------------------------------
        BIC value
        """
        centers = [kmeans.cluster_centers_]
        labels = kmeans.labels_
        # number of clusters
        m = kmeans.n_clusters
        # size of the clusters
        n = np.bincount(labels)
        # size of data set
        N, d = X.shape

        # compute variance for all clusters beforehand
        cl_var = (1.0 / (N - m) / d) * sum(
            [sum(distance.cdist(X.iloc[labels == i], [centers[0][i]], 'euclidean')) for i in range(m)])

        const_term = 0.5 * m * np.log(N) * (d + 1)

        BIC = np.sum([n[i] * np.log(n[i]) -
                      n[i] * np.log(N) -
                      ((n[i] * d) / 2) * np.log(2 * np.pi * cl_var) -
                      ((n[i] - 1) * d / 2) for i in range(m)]) - const_term

        return (BIC)

    def __compute_bic_rss(self, kmeans, X):
        k = kmeans.n_clusters
        n = len(X)
        m = X.shape[1]

        # Compute RSS (Residual Sum of Squares)
        rss = np.sum(np.min(kmeans.transform(X) ** 2, axis=1))
        # BIC formula
        bic = n * np.log(rss / n) + k * np.log(n) * m
        return bic

    def create_line_plots(self, max_num_clusters, results_folder, add_name):
        """
        CREATES A PLOT OF THE BIC PROGRESSION WITH THE NUMBER OF CLUSTERS
        Args:
            max_num_clusters: Maximum number of clusters to be tested
            results_folder: Directory where the plot will be saved
            add_name: Name of the plot

        Returns: 0

        """
        BIC_val = self.return_bic_list(max_num_clusters)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(1, max_num_clusters + 1),
            y=BIC_val,
            mode='lines+markers',
            marker=dict(size=8, color='blue'),
            line=dict(color='blue'),
            name='BIC'
        ))
        fig.update_layout(
            title='BIC by Number of Clusters',
            xaxis_title='Number of Clusters',
            yaxis_title='BIC',
            # template='plotly_white',
            xaxis=dict(tickmode='linear'),
            yaxis=dict(showgrid=True)
        )
        figure_name = os.path.join(results_folder, "BIC", f"{add_name}.png")
        Path(os.path.join(results_folder, "BIC")).mkdir(parents=True, exist_ok=True)
        fig.write_image(figure_name)

        return 0

    def return_bic_list(self, max_num_clusters):
        """
        COMPUTE THE BIC VALUE FOR EVERY NUMBER OF CLUSTERS UNTIL THE MAXIMUM SPECIFIED.
        Args:
            max_num_clusters: Maximum number of clusters

        Returns: List of BIC values.

        """
        num_clusters = range(1, max_num_clusters + 1)
        KMeans = [cluster.KMeans(n_clusters=i, init="k-means++", random_state=42).fit(self.data) for i in num_clusters]

        # now run for each cluster the BIC computation
        BIC_val = [self.__compute_bic_rss(kmeansi, self.data) for kmeansi in KMeans]

        return BIC_val

    def return_optimal_clusters(self, max_num_clusters):
        """
        RETURN THE OPTIMAL NUMBER OF CLUSTERS
        Args:
            max_num_clusters: Maximum number of clusters to be tested.

        Returns: Optimal number of clusters

        """
        BIC_val = self.return_bic_list(max_num_clusters)
        min_index = BIC_val.index(min(BIC_val))

        return min_index