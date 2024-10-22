import os
import sys
import copy
from pathlib import Path
import numpy as np
from numpy import unique
from numpy import where
import pandas as pd
from scipy import stats
from scipy.stats import t
from scipy.special import gammaln
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
import matplotlib
# matplotlib.use('TkAgg')  # or 'Agg'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utilitiespkg import DatabaseStructure, DataHandling, SearchDirectories


class Clustering(DataHandling):
    """
    CLASS THAT PROVIDES THE TOOLS FOR DOING CLUSTERING ANALYSIS IN UPDRS DATABASES FROM DRDR AND PPP.
    """
    def __init__(self, data, database_v, trem_filtered = True):
        print(" --- CREATING INSTANCE OF CLUSTERING CLASS --- ")
        super().__init__(data, database_v, trem_filtered=trem_filtered)

        if self.database == "ppp":
            self._column_model = "Labels_Model15_3clustb"
        elif self.database == "drdr":
            self._column_model = "Model6"
        else:
            raise ValueError("Database version is not valid. Select:\n"
                             "ppp - Personalized Parkinson Project\n"
                             "drdr - Dopamine Resistant vs Dopamine Responsive\n")

        self.results_clustering = os.path.join(self.results_folder, "Clustering")
        Path(self.results_clustering).mkdir(parents=True, exist_ok=True)


    def return_labels(self, method, model="", path_to_labels="", sheet_idx=0, model_name=""):
        """
        RETRIEVE THE PRE-OBTAINED CLASSIFICATION LABELS.
        Args:
            method: String indicating the method from which the labels will be retrieved
            model: String with the name of the column that contains the labels. Default to best models for each clustering method.
            path_to_labels: If method is 'other' then it needs to be provided
            sheet_idx: used only in special case, when method is 'other' and using uncommon response key.

        Returns: A list with the classification label per subject, and a dictionary with the names of each label.

        """
        self.clustering_method = method

        if model != "" and method == "two-steps":
            if model[-1] != "b" and f"{model}b" in self.data[self.sheet_names[0]].columns.tolist():
                # self._column_model = f"{model}b"
                model_specifics = f"{model}b"
            model_specifics = model
        elif model != "" and method == "arbitrary-updrs":
            model_specifics = model
        elif model == "" and method == "two-steps":
            if self.database == "ppp":
                model_specifics = self._column_model
            elif self.database == "drdr":
                model_specifics = self._column_model
        elif model == "" and method == "arbitrary-updrs":
            model_specifics = "Final_Responsiveness"
        else:
            model_specifics = ""

        if self.clustering_method == "two-steps":
            profilesSub = pd.read_excel(
                f"/project/3024023.01/{self._folder_root}/updrs_analysis/Two-Steps_clustering_labels.xlsx",
                sheet_name="LongitudinalSubjectProfile")
            mergedLabels = pd.merge(self.data[self.sheet_names[0]]["Subject"],
                                    profilesSub[['Subject', f'{model_specifics}']], on='Subject', how='left')
            cluster_labels = mergedLabels[f"{model_specifics}"]
            cluster_names = {1: "Responsive", 2: "Resistant", 3: "Intermediate"}
            if model_specifics == "Labels_Model12_3clust":
                cluster_names = {1: "Intermediate", 2: "Responsive", 3: "Resistant"}
        elif self.clustering_method == "arbitrary-updrs":
            if model_name == "":
                path_to_arbitrary_resp = self.path_to_arbitrary_responsiveness_profile
            else:
                search_Obj = SearchDirectories()
                root_directory_resp, _ = os.path.split(self.path_to_arbitrary_responsiveness_profile)
                path_to_arbitrary_resp = search_Obj.search_name_patterns(root_directory_resp, [self.name_pattern_responsiveness_file, model_name])
                path_to_arbitrary_resp = path_to_arbitrary_resp[0]
            profilesSub = pd.read_excel(path_to_arbitrary_resp,
                                        sheet_name="LongitudinalSubjectProfile")
            mergedLabels = pd.merge(self.data[self.sheet_names[0]]["Subject"],
                                    profilesSub[['Subject', f'{model_specifics}']], on='Subject', how='left')
            cluster_labels = mergedLabels[f"{model_specifics}"] # 1=Resi and 2=Resp
            cluster_names = {0: "Intermediate", 1: "Resistant", 2: "Responsive", 3: "Excluded"}
        elif self.clustering_method == "consistent-subjects":
            cluster_labels = pd.Series([1] * len(self.data[self.sheet_names[0]]["Subject"]))
            cluster_names = {1: "Consistent subjects"}
        elif self.clustering_method == "other":
            assert path_to_labels!="", ("If you select OTHER option as clustering method, you have to provide a path to "
                                        "the list of labels through the variable 'path_to_labels'")
            labelsO = pd.read_csv(path_to_labels)

            mergedLabels = pd.merge(self.data[self.sheet_names[0]]["Subject"], labelsO[['Subject', f'Labels']], on='Subject', how='left')
            cluster_labels = mergedLabels[f"Labels"]

            uniqueValues = np.unique(cluster_labels)

            try:
                key_resp = "ResponseRestTrem"
                mean_per_cluster = self.data[self.sheet_names[sheet_idx]].groupby(cluster_labels)[key_resp].mean()
            except KeyError:
                if sheet_idx == 0:
                    key_resp = f"ResponseRestTrem_1"
                elif sheet_idx == 2:
                    key_resp = f"ResponseRestTrem_2"
                elif sheet_idx == 4:
                    key_resp = f"ResponseRestTrem_3"
                mean_per_cluster = self.data[self.sheet_names[0]].groupby(cluster_labels)[key_resp].mean()

            sorted_means = mean_per_cluster.sort_values()
            cluster_names = {
                sorted_means.index[0]: "Resistant",  # Cluster with the smallest mean
                sorted_means.index[1]: "Responsive"  # Cluster with the largest mean
            }

            if "spectral" in path_to_labels:
                method_name = "spectral"
            elif "gaussian" in path_to_labels:
                method_name = "gaussian"
            elif "kmeans" in path_to_labels:
                method_name = "kmeans"
            self.clustering_method = method_name
        else:
            raise ValueError("METHOD NOT IMPLEMENTED YET. SELECT FROM: \n"
                             "two-steps \n"
                             "arbitrary-updrs \n"
                             "consistent-subjects.\n")

        return (cluster_labels, cluster_names)

    def apply_clustering(self, clust_type, n_clusters=2, sheet_idx=0,
            predictors=["ResponseTremorUPDRS_1", "ResponseLimbsRestTrem_1"],
            model_name=""
    ):
        """
        PERFORM SPECTRAL CLUSTERING USING SKLEARN LIBRARY.
        The function will create a scatter figure and a csv file with the relation of observation ID and labels.
        Args:
            clust_type: Type of clustering: spectral, kmeans, gaussian, or dbscan.
            n_clusters: Number of clusters. Default to 2.
            sheet_idx: Index of the sheet in the database from where data will be taken. Default to 0.
            predictors: List of names od the columns that will be used as predictors. Dafault to Tremor predictors.
            model_name: ID name added to the image name. If empty it will be automatically generated.

        Returns: Labels per observation

        """
        drdr_reduced = False
        if self.database == "drdr":
            dopamine_resistant = [30, 8, 11, 28, 27, 42, 50, 72, 75, 74, 73, 78, 81, 83]
            dopamine_responsive = [2, 18, 60, 59, 38, 49, 40, 19, 29, 36, 33, 71, 21, 70, 64, 56, 48, 43, 76, 77]
            reduced_subjects = dopamine_resistant + dopamine_responsive
            reduced_subjects = [i - 1 for i in reduced_subjects]
            drdr_reduced = True
        
        if model_name == "":
            model_name = '-'.join(predictors)
        
        model = {
            "spectral": SpectralClustering(n_clusters=n_clusters),
            "gaussian": GaussianMixture(n_components=n_clusters),
            "kmeans": KMeans(n_clusters=n_clusters),
            "dbscan": DBSCAN(eps=0.30, min_samples=9)
        }.get(clust_type)
        
        data_filt = self.data[self.sheet_names[sheet_idx]][predictors]
        if clust_type == "spectral" or clust_type == "dbscan":
            yhat = model.fit_predict(data_filt)
        elif clust_type == "kmeans" or clust_type == "gaussian":
            model.fit(data_filt)
            yhat = model.predict(data_filt)
        else:
            raise ValueError("Clustering method not implemented yet. Check function description for valid clustering methods.")
        
        clusters = unique(yhat)
        for cluster in clusters:
            row_ix = where(yhat == cluster)
            if not drdr_reduced:
                plotx = [data_filt[predictors[0]][i] for i in row_ix]
                ploty = [data_filt[predictors[1]][i] for i in row_ix]
                reduced_str = ""
            else:
                plotx = [data_filt[predictors[0]][i] for i in row_ix[0] if i in reduced_subjects]
                ploty = [data_filt[predictors[1]][i] for i in row_ix[0] if i in reduced_subjects]
                reduced_str = "34participants"
            plt.scatter(plotx, ploty)
        figure_name = os.path.join(self.results_clustering, f"{clust_type}_{model_name}_clusters={n_clusters}_{reduced_str}.png")
        plt.xlabel(predictors[0])
        plt.ylabel(predictors[1])
        plt.title(f"{clust_type} clustering")
        plt.savefig(figure_name, bbox_inches='tight', dpi=600)
        plt.clf()
        
        # Save CSV file
        csv_name = os.path.join(self.results_clustering, f"{clust_type}_{model_name}_clusters={n_clusters}_{reduced_str}.csv")
        results = pd.DataFrame()
        if self.database == "ppp":
            results["Subject"] = subjects = self.data[self.sheet_names[sheet_idx]]["Subject"]
        else:
            results["Subject"] = self.data[self.sheet_names[sheet_idx]]["PatCode"]
        results["Labels"] = yhat
        
        results.to_csv(csv_name, index=False)
        
        return yhat

    def plot_subject_wise_cluster_comparison(self, methods_to_compare):
        """
        PLOTS A GRID WITH ONE SQUARE PER SUBJECT. THE COLOR WILL BE GREEN IF THE SUBJECT WAS CLASSIFIED IN THE SAME
        GROUP BY BOTH METHODS, RED IF IT IS IN DIFFERENT GROUPS, BLUE IF IN AT LEAST ONE METHOD IT IS INDETERMINED,
        AND WHITE IF MISSING DATA.
        Args:
            methods_to_compare: List of two strings with the names of the two methods to compare.

        Returns: 0. Saves the figures to the clustering results folder.

        """
        assert len(methods_to_compare) == 2, "You have to provide 2 methods."

        (labels_1, clust_names_1) = self.return_labels(methods_to_compare[0])
        (labels_2, clust_names_2) = self.return_labels(methods_to_compare[1])

        assert len(labels_1) == len(labels_2), "Series must be of the same length"
        length = len(labels_1)
        size = int(np.ceil(np.sqrt(length)))
        padded_length = size * size
        padded_labels_1 = pd.concat([labels_1, pd.Series([np.nan] * (padded_length - length))], ignore_index=True)
        padded_labels_2 = pd.concat([labels_2, pd.Series([np.nan] * (padded_length - length))], ignore_index=True)

        # Create a grid of size x size
        grid1 = padded_labels_1.values.reshape((size, size))
        grid2 = padded_labels_2.values.reshape((size, size))

        keyResi = next((keyResi for keyResi, value in clust_names_1.items() if value == "Resistant"), None)
        keyResp = next((keyResp for keyResp, value in clust_names_2.items() if value == "Resistant"), None)

        # Initialize a color map
        if keyResi == keyResp:
            colors = ['green', 'red', 'blue', 'white']
        else:
            colors = ['red', 'green', 'blue', 'white']
        cmap = ListedColormap(colors)

        # Define grid colors based on the conditions
        grid_colors = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                v1 = grid1[i, j]
                v2 = grid2[i, j]
                if v1 == 3:
                    v1 = 0
                if pd.isna(v1) or pd.isna(v2):
                    grid_colors[i, j] = 3  # Handle NaN values if any
                elif (v1 == 0 and v2 != 0) or (v2 == 0 and v1 != 0):
                    grid_colors[i, j] = 2  # Blue if one value is 0
                elif v1 == v2:
                    grid_colors[i, j] = 0
                else:
                    grid_colors[i, j] = 1

        # Plotting
        fig, ax = plt.subplots(figsize=(size, size))
        cax = ax.matshow(grid_colors, cmap=cmap, vmin=0, vmax=3)

        # Adding a horizontal color bar
        cbar = plt.colorbar(cax, ticks=[0, 1, 2, 3], orientation='horizontal')
        cbar.set_label('Grid Color Legend')
        cbar.ax.set_xticklabels(['Error', 'Match', 'Zero(no-valid)', 'NaNs'])

        # Adjust layout to fit color bar
        plt.subplots_adjust(bottom=0.15)  # Adjust this value as needed

        # Adding labels
        ax.set_xticks(np.arange(size))
        ax.set_yticks(np.arange(size))
        ax.set_xticklabels(np.arange(1, size + 1))
        ax.set_yticklabels(np.arange(1, size + 1))

        plt.title('Comparison of Cluster per Subject', size=16)

        figure_name = os.path.join(self.results_clustering, f"{methods_to_compare[0]}_vs_{methods_to_compare[1]}_results_comparisons.png")
        plt.savefig(figure_name, bbox_inches='tight', dpi=600)

        return 0

    def plot_participant_clusters_bars(self):
        """
        PLOT A GRID OF ONE SQUARE PER SUBJECT. EACH SQUARE HAS 3 BARS SIGNALING THE CLUSTER THAT WAS ASSIGNED IN EACH OF
        THE 3 VISITS. BASED ON THE arbitrary-updrs METHOD FOR CLASSIFICATION.
        Returns: 0. Saves the figure to the clustering results folder.

        """
        assert self.database == "ppp", "This function only works for PPP database."

        profilesSub = pd.read_excel(self.path_to_arbitrary_responsiveness_profile, sheet_name="LongitudinalSubjectProfile")
        visitsnames = ["Responsiveness_Baseline", "Responsiveness_Year 1", "Responsiveness_Year 2"]
        df = profilesSub[visitsnames]

        colors = {0: 'blue', 1: 'green', 2: 'red'}
        num_participants = df.shape[0]
        # num_cols = 20
        num_cols = int(np.ceil(np.sqrt(num_participants)))
        num_rows = num_cols
        # num_rows = int(np.ceil(num_participants / num_cols))
        figsize = (num_cols * 2, num_rows * 2)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True)
        axes = axes.flatten()
        for i, row in df.iterrows():
            visits = [row[visitsnames[0]], row[visitsnames[1]], row[visitsnames[2]]]
            colors_for_visits = [colors[val] for val in visits]
            ax = axes[i]
            bars = ax.bar([visitsnames[0], visitsnames[1], visitsnames[2]], visits, color=colors_for_visits)
            ax.set_ylim(0, 2.5)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

        for j in range(num_participants, len(axes)):
            axes[j].axis('off')

        fig.suptitle("Clustering Distribution by Visit", fontsize=20)
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        figure_name = os.path.join(self.results_clustering, "arbitrary-updrs_clustering-by-visit.png")
        plt.savefig(figure_name, bbox_inches='tight', dpi=600)

        return 0

    def plot_participant_clusters_stability(self):
        """
        ANALYZE CLASSIFICATION STABILITY ACROSS VISITS. GREEN SQUARES REPRESENT SUBJECTS THAT DID NOT CHANGE GROUPS
        ACROSS VISITS, RED THE ONES THAT DID, AND BLUE REPRESENT THE ONES THAT CHANGED TO AN INTERMEDIATE GROUP.
        Returns: 0. It creates the figure and saves it to the clustering results folder.
        """
        assert self.database == "ppp", "This function only works for PPP database."

        (labels_baseline, _) = self.return_labels(method="arbitrary-updrs", model="Responsiveness_Baseline")
        (labels_year1, _) = self.return_labels(method="arbitrary-updrs", model="Responsiveness_Year 1")
        (labels_year2, _) = self.return_labels(method="arbitrary-updrs", model="Responsiveness_Year 2")
        (labels_final, labels_dict) = self.return_labels(method="arbitrary-updrs", model="Final_Responsiveness")

        group_subjects = {
            'Resistant': [],
            'Intermediate': [],
            'Responsive': []
        }
        for i, label in enumerate(labels_final):
            group = labels_dict[label]
            group_subjects[group].append(i)

        label_color_map = {
            0: 'yellow',  # Intermediate
            1: 'red',  # Resistant
            2: 'green'  # Responsive
        }

        # Increase the size of the figure to make room for the rectangles (change the width and height accordingly)
        fig, axs = plt.subplots(1, 3, figsize=(24, 15))  # Adjust height based on the number of subjects

        for ax, (group_name, subject_indices) in zip(axs, group_subjects.items()):
            ax.set_title(f"{group_name}, n={len(subject_indices)}", fontsize=20)

            for row, subject_idx in enumerate(subject_indices):
                subject_baseline = labels_baseline[subject_idx]
                subject_year1 = labels_year1[subject_idx]
                subject_year2 = labels_year2[subject_idx]

                colors = [
                    label_color_map[subject_baseline],
                    label_color_map[subject_year1],
                    label_color_map[subject_year2]
                ]

                for col, color in enumerate(colors):
                    # Adjust the rectangle width and height to make the rectangles fill the space
                    ax.add_patch(plt.Rectangle((col, row), 2, 1, facecolor=color))  # The (1, 1) makes them square-like

            # Set limits and adjust aspect ratio for rectangle-shaped patches
            ax.set_xlim(0, 3)
            ax.set_ylim(0, len(subject_indices))
            ax.set_xticks([0.5, 1.5, 2.5])
            ax.set_xticklabels(['Baseline', 'Year 1', 'Year 2'])
            ax.set_yticks([])
            ax.invert_yaxis()  # Flip to have the first subject at the top

        legend_elements = [
            mpatches.Patch(color='green', label='Responsive'),
            mpatches.Patch(color='yellow', label='Intermediate'),
            mpatches.Patch(color='red', label='Resistant')
        ]

        # Place the legend at the bottom center of the figure
        fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()

        # fig.suptitle('Participant Clusters Stability', fontsize=28)

        if not os.path.exists(self.results_clustering):
            os.makedirs(self.results_clustering)

        plt.savefig(os.path.join(self.results_clustering, 'participants_arbitrary-clustering_by_group_by_visit.png'),
                    bbox_inches='tight', dpi=900)
        plt.clf()
        plt.close()

        return 0

    def plot_participant_clusters_stability_old(self):
        """
        ANALYZE CLASSIFICATION STABILITY ACROSS VISITS. GREEN SQUARES REPRESENT SUBJECTS THAT DIDNOT CHANGE GROUPS
        ACROSS VISITS, RED THE ONES THAT DID, AND BLUE REPRESENTS THE ONES THAT CHANGED TO AN INTERMEDIATE GROUP.
        Returns: 0. It creates the figure and saves it to the clustering results folder.

        """
        assert self.database == "ppp", "This function only works for PPP database."

        profilesSub = pd.read_excel(self.path_to_arbitrary_responsiveness_profile, sheet_name="LongitudinalSubjectProfile")
        visits_names = ["Responsiveness_Baseline", "Responsiveness_Year 1", "Responsiveness_Year 2"]
        df_clusters = profilesSub[visits_names]

        n_participants = len(df_clusters)
        colors = []
        for i, row in df_clusters.iterrows():
            values = row[visits_names].values
            unique_values = set(values)
            if 1 in unique_values and 2 in unique_values:
                colors.append(2)  # Condition 1: Contains 1 and 2 -> Red
            elif (1 in unique_values and 0 in unique_values) or (2 in unique_values and 0 in unique_values):
                colors.append(1)  # Condition 2: Contains 1 and 0, or 2 and 0 -> Blue
            elif (1 in unique_values and 3 in unique_values) or (2 in unique_values and 3 in unique_values):
                colors.append(1)
            elif len(unique_values) == 1:
                colors.append(0)  # Condition 3: Contains only 1 unique number -> Green
            else:
                colors.append(3)  # Fallback color (in case of an unhandled case)

        num_cols = int(np.ceil(np.sqrt(n_participants)))
        total_size = num_cols * num_cols
        if len(colors) < total_size:
            colors.extend([3] * (total_size - len(colors)))  # Pad with 'gray' color
        else:
            colors = colors[:total_size]

        color_grid = np.array(colors).reshape(num_cols, num_cols)
        color_map = ListedColormap(['green', 'blue', 'red', 'white'])

        fig, ax = plt.subplots(figsize=(num_cols, num_cols))
        ax.matshow(color_grid, cmap=color_map, vmin=0, vmax=3)
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        fig.suptitle('Participant Clusters Stability')

        if not os.path.exists(self.results_clustering):
            os.makedirs(self.results_clustering)

        plt.savefig(os.path.join(self.results_clustering, 'participant_clusters_stability.png'), bbox_inches='tight',
                    dpi=600)
        plt.clf()
        plt.close()
        return 0

    def clusters_scatter_plot(self, plotting_keys, clustering_method, visit_name="Baseline", path_to_labels="", model_name=""):
        """

        Args:
            plotting_keys:
            clustering_method:
            visit_name:
            path_to_labels:
            model_name:

        Returns:

        """
        sheet_idx = {
            "Baseline": 0,
            "Year 1": 2,
            "Year 2": 4
        }.get(visit_name, 0)

        (labels, labels_dict) = self.return_labels(method=clustering_method, path_to_labels=path_to_labels, model=model_name, sheet_idx=sheet_idx)
        # data = self.data[self.sheet_names[sheet_idx]][plotting_keys]
        data, plotting_keys = self.__get_data_with_corrected_keys(sheet_idx, plotting_keys)

        labels = np.array(labels)
        fig, ax = plt.subplots()
        unique_labels = np.unique(labels)  # [np.unique(labels) != 0]
        colors = plt.cm.get_cmap('tab10', len(unique_labels))

        for i, label in enumerate(unique_labels):
            cluster_data = data[labels == label]
            ax.scatter(cluster_data[plotting_keys[0]], cluster_data[plotting_keys[1]],
                       color=colors(i), label=f'{labels_dict[label]}')

        # Set plot labels and title
        ax.set_xlabel(plotting_keys[0])
        ax.set_ylabel(plotting_keys[1])
        ax.set_title(f"Clustering with {clustering_method}")
        ax.legend()

        figure_name = os.path.join(self.results_clustering, f"clusters_separation_{self.clustering_method}{model_name}_{visit_name}.png")
        plt.savefig(figure_name, bbox_inches='tight', dpi=600)
        plt.clf()
        return 0

    def __get_data_with_corrected_keys(self, sheet_idx, plotting_keys):
        try:
            corrected_keys = plotting_keys
            data = self.data[self.sheet_names[sheet_idx]][plotting_keys]
        except KeyError:
            corrected_keys = []
            for key in plotting_keys:
                if key in self.data[self.sheet_names[sheet_idx]].columns:
                    corrected_keys.append(key)
                else:
                    if sheet_idx == 0:
                        corrected_keys.append(f"{key}_1")
                    elif sheet_idx == 2:
                        corrected_keys.append(f"{key}_2")
                    elif sheet_idx == 4:
                        corrected_keys.append(f"{key}_3")

            data = self.data[self.sheet_names[sheet_idx]][corrected_keys]

        return data, corrected_keys


