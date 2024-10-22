import os
import sys
import copy
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from scipy.special import gammaln
import statsmodels.formula.api as smf
from scipy.spatial import distance
from sklearn import cluster
from statsmodels.stats.anova import AnovaRM
from pathlib import Path
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap

# My Library: Codebase
from utilitiespkg import DataHandling
from clusteringpkg import Clustering
from plottingpkg import UPDRSPlotting

class StatisticsHelper(UPDRSPlotting):
    """
    CLASS THAT PROVIDES TOOLS FOR STATISTICAL ANALYSIS OF UPDRS SCORES.
    """
    def __init__(self, data, database_v, trem_filtered=True):
        print(" --- INITIALIZING INSTANCE OF STATISTICS HELPER CLASS --- ")
        super().__init__(data, database_v, trem_filtered = trem_filtered)

        if self.trem_filtered_flag:
            analysis_name = f"Stats_{self.filtered_string}"
        else:
            analysis_name = f"Stats"
        self.results_stats = os.path.join(self.results_folder, analysis_name)
        Path(os.path.join(self.results_stats)).mkdir(parents=True, exist_ok=True)


    def repeated_measures_anova(self, keys_to_analyze, model_name=""):
        """
        PERFORMS REPEATED MEASURES ANOVA ANALYSIS OF THE VARIABLES SPECIFIED IN THE keys_to_analyze VARIABLE.
        Args:
            keys_to_analyze: List of the column names of the variables that will be analyzed using RMAnova
            model_name: String that will identify the model that was run, for example, for variation of keys. Default to empty string.

        Returns: 0. The function creates an excell file with the results in the results folder directory.

        """
        repeated_measures = dict()
        df = pd.DataFrame()

        visits = [0, 1, 2]
        sessions = [0, 1] # 0 = OFF and 1 = ON

        visitTime = "Visit"
        condition = "Condition"
        subjects = "Subject"
        scores = "UPDRS_scores"

        visit_labels_map = {0: "Baseline", 1: "Year 1", 2: "Year 2"}
        condition_labels_map = {0: "OFF", 1: "ON"}

        for key in keys_to_analyze:
            visit_labels = []
            condition_labels = []
            subject_labels = []
            score_values = []
            for visit in visits:
                for ses in sessions:
                    updrs_scores = self.data[self.sheet_names[2 * visit + ses]][key]
                    lengthData = len(updrs_scores)

                    visit_label = visit_labels_map.get(visit, "Unknown")
                    ses_label = condition_labels_map.get(ses, "Unknown")

                    visit_labels.extend([visit_label] * lengthData)
                    condition_labels.extend([ses_label] * lengthData)
                    subject_labels.extend(self.data[self.sheet_names[2 * visit + ses]]["Subject"].tolist())
                    score_values.extend(updrs_scores.tolist())

            df = pd.DataFrame({
                visitTime: visit_labels,
                condition: condition_labels,
                subjects: subject_labels,
                scores: score_values
            })
            # anova_results = pg.rm_anova(dv=scores, within=[visitTime, condition], subject=subjects, data=df, detailed=True)
            anova_results = AnovaRM(data=df, depvar=scores, subject=subjects, within=[visitTime, condition]).fit()
            # print(anova_results)
            repeated_measures[key] = anova_results.anova_table

        with pd.ExcelWriter(os.path.join(self.results_stats, f"{model_name}rm_anova_off-on.xlsx"), engine='openpyxl') as writer:
            for sheet_name, data in repeated_measures.items():
                data.to_excel(writer, sheet_name=sheet_name, index=True)

        return 0

    def mixed_effects(self, keys_to_analyze, classification_type="two-steps", model_name="", use_covariates=True, path_to_labels="", model_labels=""):
        """
        PERFORMS MIXED-EFFECTS LINEAR MODELS ANALYSIS COMPARING THE DIFFERENT CLUSTERS GROUPS.
        Args:
            keys_to_analyze: List of UPDRS column names that will be used as the measurement scores. Each element will be analized separately.
            classification_type: Type of classification that led to the clusters.
            model_name: Identifier string for the analysis.
            use_covariates: Flag that indicates whether covariates will be used in the analysis.
            path_to_labels: If classification_type is 'other' then it needs to point to the file containing the labels

        Returns: 0. It creates Excel files in the Stats results folder.

        """
        Path(os.path.join(self.results_stats, "Design_2_factors")).mkdir(parents=True, exist_ok=True)
        dataFrames_toSave = dict()
        (cluster_labels, condition_labels_map) = self.return_labels(method=classification_type, path_to_labels=path_to_labels, model=model_labels, model_name=model_name)

        repeated_measures = dict()
        pairwise_results_compilation = dict()
        df = pd.DataFrame()

        visits = [0, 1, 2]

        visitTime = "Visit"
        condition = "Group"
        subjects = "Subject"
        scores = "UPDRS_scores"
        LED_key = "LED"

        visit_labels_map = {0: "Baseline", 1: "Year 1", 2: "Year 2"}

        unique_labels = cluster_labels.unique()  # [cluster_labels.unique() != 0]
        if condition_labels_map == {}:
            for clust in unique_labels:
                condition_labels_map[clust] = f"Cluster{clust}"

        for key in keys_to_analyze:
            visit_labels = []
            condition_labels = []
            subject_labels = []
            score_values = []
            LED_values = []
            for visit in visits:
                for ses in unique_labels:
                    updrs_scores = self.data[self.sheet_names[2 * visit]][key][cluster_labels == ses]
                    lengthData = len(updrs_scores)

                    LED = self.data[self.sheet_names[2 * visit + 1]]["LEDD"][cluster_labels == ses]
                    LED.fillna(LED.mean(), inplace=True)

                    visit_label = visit_labels_map.get(visit, "Unknown")
                    ses_label = condition_labels_map.get(ses, "Unknown")

                    visit_labels.extend([visit_label] * lengthData)
                    condition_labels.extend([ses_label] * lengthData)
                    subject_labels.extend(self.data[self.sheet_names[2 * visit]]["Subject"][cluster_labels == ses].tolist())
                    score_values.extend(updrs_scores.tolist())
                    LED_values.extend(LED.tolist())

            if use_covariates:
                LED_values = pd.Series(LED_values) + 1e-6
                LED_values = LED_values.tolist()

            df = pd.DataFrame({
                visitTime: visit_labels,
                condition: condition_labels,
                subjects: subject_labels,
                scores: score_values,
                LED_key: LED_values
            })
            dataFrames_toSave[key] = df

            if use_covariates:
                model = smf.mixedlm(
                    f"{scores} ~ C({condition}, Treatment(reference='Responsive')) * {visitTime} + {LED_key}", df,
                    groups=df[subjects])
            else:
                model = smf.mixedlm(f"{scores} ~ C({condition}, Treatment(reference='Responsive')) * {visitTime}", df, groups=df[subjects])

            mixedlm_results = model.fit()

            summary_df = mixedlm_results.summary().tables[1]
            repeated_measures[key] = pd.DataFrame(summary_df)

            pairwise_results = self.__pairwise_ttests_posthoc(df, scores, visitTime, condition, subjects)
            pairwise_results_compilation[key] = pairwise_results

        if use_covariates:
            if model_name == "":
                model_name = "LED-Cov"
            else:
                model_name = f"LED-Cov_{model_name}"

        with pd.ExcelWriter(os.path.join(self.results_stats, "Design_2_factors", f"{self.clustering_method}_{model_name}_pre-data-frames.xlsx"),
                            engine='openpyxl') as writer:
            for sheet_name, data in dataFrames_toSave.items():
                data.to_excel(writer, sheet_name=sheet_name, index=True)

        with pd.ExcelWriter(os.path.join(self.results_stats, "Design_2_factors", f"{self.clustering_method}_{model_name}_mixed_effects.xlsx"),
                            engine='openpyxl') as writer:
            for sheet_name, data in repeated_measures.items():
                data.to_excel(writer, sheet_name=sheet_name, index=True)

        with pd.ExcelWriter(os.path.join(self.results_stats, "Design_2_factors", f"{self.clustering_method}_{model_name}_posthoc_paired_T-Test.xlsx"),
                            engine='openpyxl') as writer:
            for sheet_name, data in pairwise_results_compilation.items():
                data.to_excel(writer, sheet_name=sheet_name, index=True)
        return 0

    def mixed_effects_3factors_design(self, keys_to_analyze, classification_type="two-steps", model_name="", use_covariates=True, path_to_labels="", model_labels=""):
        """
        PERFORMS MIXED-EFFECTS LINEAR MODELS ANALYSIS COMPARING THE DIFFERENT CLUSTERS GROUPS USING A 3 X 2 X 2 DESIGN. 3 VISITS, 2 GROUPS (RESP/RESI), 2 SESSIONS (OFF/ON).
        Args:
            keys_to_analyze: List of UPDRS column names that will be used as the measurement scores. Each element will be analized separately.
            classification_type: Type of classification that led to the clusters.
            model_name: Identifier string for the analysis.
            use_covariates: Flag that indicates whether covariates will be used in the analysis.
            path_to_labels: If classification_type is 'other' then it needs to point to the file containing the labels

        Returns: 0. It creates Excel files in the Stats results folder.

        """
        Path(os.path.join(self.results_stats, "Design_3_factors")).mkdir(parents=True, exist_ok=True)
        dataFrames_toSave = dict()
        (cluster_labels, condition_labels_map) = self.return_labels(method=classification_type, path_to_labels=path_to_labels, model=model_labels, model_name=model_name)

        repeated_measures = dict()
        pairwise_results_compilation = dict()
        df = pd.DataFrame()

        visits = [0, 1, 2]

        visitTime = "Visit"
        condition = "Group"
        subjects = "Subject"
        scores = "UPDRS_scores"
        sessions = "Session"
        LED_key = "LED"

        visit_labels_map = {0: "Baseline", 1: "Year 1", 2: "Year 2"}

        unique_labels = cluster_labels.unique()  # [cluster_labels.unique() != 0]
        if condition_labels_map == {}:
            for clust in unique_labels:
                condition_labels_map[clust] = f"Cluster{clust}"

        for key in keys_to_analyze:
            visit_labels = []
            condition_labels = []
            subject_labels = []
            score_values = []
            LED_values = []
            sessions_labels = []
            for visit in self.sheet_names:
                for label in unique_labels:
                    updrs_scores = self.data[visit][key][cluster_labels == label]
                    lengthData = len(updrs_scores)

                    LED = self.data[visit]["LEDD"][cluster_labels == label]
                    LED.fillna(LED.mean(), inplace=True)

                    visit_idx =  0 if "1" in visit else 1 if "2" in visit else 2
                    visit_label = visit_labels_map.get(visit_idx, "Unknown")
                    group_label = condition_labels_map.get(label, "Unknown")
                    ses_label = "off" if "OFF" in visit else "on"

                    visit_labels.extend([visit_label] * lengthData)
                    condition_labels.extend([group_label] * lengthData)
                    subject_labels.extend(self.data[visit]["Subject"][cluster_labels == label].tolist())
                    score_values.extend(updrs_scores.tolist())
                    LED_values.extend(LED.tolist())
                    sessions_labels.extend([ses_label] * lengthData)

            if use_covariates:
                # LED_values = pd.Series(LED_values) + 1e-6
                LED_values = pd.Series(LED_values)
                LED_values = LED_values.tolist()

            df = pd.DataFrame({
                subjects: subject_labels,
                visitTime: visit_labels,
                condition: condition_labels,
                sessions: sessions_labels,
                scores: score_values,
                LED_key: LED_values
            })
            dataFrames_toSave[key] = df

            if use_covariates:
                model = smf.mixedlm(
                    f"{scores} ~ C({condition}, Treatment(reference='Responsive')) * C({sessions}, Treatment(reference='off')) * {visitTime} + {LED_key}",
                    df,
                    groups=df[subjects]
                )
            else:
                model = smf.mixedlm(
                    f"{scores} ~ C({condition}, Treatment(reference='Responsive')) * C({sessions}, Treatment(reference='off')) * {visitTime}",
                    df,
                    groups=df[subjects]
                )

            mixedlm_results = model.fit()

            summary_df = mixedlm_results.summary().tables[1]
            repeated_measures[key] = pd.DataFrame(summary_df)

            pairwise_results = self.__pairwise_ttests_posthoc_3factors(df, scores, visitTime, condition, subjects, sessions)
            pairwise_results_compilation[key] = pairwise_results

        if use_covariates:
            if model_name == "":
                model_name = "LED-Cov"
            else:
                model_name = f"LED-Cov_{model_name}"

        with pd.ExcelWriter(os.path.join(self.results_stats, "Design_3_factors", f"{self.clustering_method}_{model_name}_3factors_pre-data-frames.xlsx"),
                            engine='openpyxl') as writer:
            for sheet_name, data in dataFrames_toSave.items():
                data.to_excel(writer, sheet_name=sheet_name, index=True)

        with pd.ExcelWriter(os.path.join(self.results_stats, "Design_3_factors", f"{self.clustering_method}_{model_name}_3factors_mixed_effects.xlsx"),
                            engine='openpyxl') as writer:
            for sheet_name, data in repeated_measures.items():
                data.to_excel(writer, sheet_name=sheet_name, index=True)

        with pd.ExcelWriter(os.path.join(self.results_stats, "Design_3_factors", f"{self.clustering_method}_{model_name}_3factors_posthoc_paired_T-Test.xlsx"),
                            engine='openpyxl') as writer:
            for sheet_name, data in pairwise_results_compilation.items():
                data.to_excel(writer, sheet_name=sheet_name, index=True)
        return 0

    def create_clinical_metrics_table(self, clustering_method="arbitrary-updrs"):
        percentageChangeFunction = self.percentage_change_basic
        if self.trem_filtered_flag:
            analysis_name = f"StatsByGroup_{self.filtered_string}"
        else:
            analysis_name = f"StatsByGroup"
        results_folder = os.path.join(self.results_stats, analysis_name)
        Path(os.path.join(results_folder)).mkdir(parents=True, exist_ok=True)

        database = copy.deepcopy(self.data)

        clinical_conf = {
            'updrs': [
                {'off': 'Age', 'on': 'Age'},
                {'off': 'Gender', 'on': 'Gender'},
                {'off': 'HoeYah', 'on': 'HoeYah'},
                {'off': 'MonthSinceDiag', 'on': 'MonthSinceDiag'},
            ],
            'subplots': [
                {'groups': ['Resistant', 'Intermediate', 'Responsive'], 'visits': ['Baseline', 'Year 1', 'Year 2'], 'sessions': ['off']},
            ]}
        dataframes_clinical = self.obtain_formatted_dataframe(clinical_conf, clustering_method)
        
        updrs_conf = {
            'updrs': [
                {'off': 'LEDD', 'on': 'LEDD'},
                {'off': 'ChangeTotalU3', 'on': 'ChangeTotalU3'},
                {'off': 'ChangeBrady14Items', 'on': 'ChangeBrady14Items'},
                {'off': 'ChangeLimbsRigidity5Items', 'on': 'ChangeLimbsRigidity5Items'},
                {'off': 'ChangeLimbsRestTrem', 'on': 'ChangeLimbsRestTrem'},
                {'off': 'LogPower', 'on': 'LogPower'}
            ],
            'subplots': [
                {'groups': ['Resistant', 'Intermediate', 'Responsive'], 'visits': ['Baseline', 'Year 1', 'Year 2'], 'sessions': ['on']},
            ]}
        dataframes_updrs_change = self.obtain_formatted_dataframe(updrs_conf, clustering_method)

        dataframes_clinical.append(dataframes_updrs_change.pop(0))

        updrs_conf = {
            'updrs': [
                {'off': 'TotalU3', 'on': 'TotalU3'},
                {'off': 'Brady14Items', 'on': 'Brady14Items'},
                {'off': 'LimbsRigidity5Items', 'on': 'LimbsRigidity5Items'},
                {'off': 'LimbsRestTrem', 'on': 'LimbsRestTrem'},
                {'off': 'LogPower', 'on': 'LogPower'}
            ],
            'subplots': [
                {'groups': ['Resistant', 'Intermediate', 'Responsive'], 'visits': ['Baseline', 'Year 1', 'Year 2'],
                 'sessions': ['off', 'on']},
            ]}
        dataframes_updrs = self.obtain_formatted_dataframe(updrs_conf, clustering_method)

        # Initialize STATS table
        stats_table = dict()
        
        groups_labels = ['Resistant', 'Intermediate', 'Responsive']
        for group_label in groups_labels:
            stats_table[f"{group_label}-UPDRSMetrics"] = pd.DataFrame()
            stats_table[f"{group_label}-ClinicalMetrics"] = pd.DataFrame()
            stats_table[f"{group_label}-DopamineResponsiveness"] = pd.DataFrame()

            # Set names for measurements
            stats_table[f"{group_label}-UPDRSMetrics"]["Clinical Measurement"] = [
                f"MDS-UPDRS part III, mean \u00B1 std",
                f"MDS-UPDRS-Bradykinesia, mean \u00B1 std",
                f"MDS-UPDRS-Rigidity, mean \u00B1 std",
                f"MDS-UPDRS-LimbsRestingTremor, mean \u00B1 std",
                f"LogPower, mean \u00B1 std",
            ]

            stats_table[f"{group_label}-DopamineResponsiveness"]["Clinical ratings"] = [
                "MDS-UPDRS part III - %C, mean (range)",
                "MDS-UPDRS-%CBradykinesia, mean (range)",
                "MDS-UPDRS-%CRigidity, mean (range)",
                "MDS-UPDRS-%CLimbsRestTrem, mean (range)"
            ]

            stats_table[f"{group_label}-ClinicalMetrics"]["Metric Name"] = [
                "Age, y",
                "Sex, M/F",
                "H&Y stage, median",
                "Disease duration, y",
                "LED",
            ]

            updrs_metrics_change = ["ChangeTotalU3", "ChangeBrady14Items", "ChangeLimbsRigidity5Items", "ChangeLimbsRestTrem"]
            clinical_metrics = ["Age", "Gender", "HoeYah", "MonthSinceDiag", "LEDD"]
            updrs_metrics = ["TotalU3", "Brady14Items", "LimbsRigidity5Items", "LimbsRestTrem", "LogPower"]

            visit_names = ["Baseline", "Year 1", "Year 2"]
            sessions = ["OFF", "ON"]

            for i, visit_name in enumerate(visit_names):
                # Dopamine responsiveness per group epr visit
                appended_data = []
                for j, metric in enumerate(updrs_metrics_change):
                    filtUPDRS_df = dataframes_updrs_change[j].groupby(['Visit', 'Group']).get_group((visit_name, group_label))
                    appended_data.append(f"{filtUPDRS_df[metric].mean():.3f} ({np.min(filtUPDRS_df[metric]):.3f}-{np.max(filtUPDRS_df[metric]):.3f})")
                stats_table[f"{group_label}-DopamineResponsiveness"][visit_name] = appended_data

                # Clinical metrics per group per visit
                appended_data = []
                for j, metric in enumerate(clinical_metrics):
                    filtUPDRS_df = dataframes_clinical[j].groupby(['Visit', 'Group']).get_group((visit_name, group_label))
                    string_append = {
                        'Age': f"{filtUPDRS_df[metric].mean():.3f} ({np.min(filtUPDRS_df[metric])}-{np.max(filtUPDRS_df[metric])})",
                        'Gender': f"{(filtUPDRS_df[metric] == 1).sum()}/{(filtUPDRS_df[metric] == 2).sum()}",
                        'HoeYah': f"{np.median(filtUPDRS_df[metric])} ({np.min(filtUPDRS_df[metric])}-{np.max(filtUPDRS_df[metric])})",
                        'MonthSinceDiag': f"{filtUPDRS_df[metric].mean() / 12:.3f} ({np.min(filtUPDRS_df[metric])/12:.3f} - {np.max(filtUPDRS_df[metric])/12:.3f})",
                        'LEDD': f"{filtUPDRS_df[metric].mean():.3f} ({np.min(filtUPDRS_df[metric]):.3f}-{np.max(filtUPDRS_df[metric]):.3f})"
                    }.get(metric)
                    appended_data.append(string_append)
                stats_table[f"{group_label}-ClinicalMetrics"][visit_name] = appended_data

                # UPDRS metrics per group, per pisit, and per session
                for ses_name in sessions:
                    appended_data = []
                    for j, metric in enumerate(updrs_metrics):
                        filtUPDRS_df = dataframes_updrs[j].groupby(['Visit', 'Group', 'Session']).get_group((visit_name, group_label, ses_name))
                        appended_data.append(f"{filtUPDRS_df[metric].mean():.3f}\u00B1{filtUPDRS_df[metric].std():.3f}")
                    stats_table[f"{group_label}-UPDRSMetrics"][f"{visit_name}-{ses_name}"] = appended_data
            
            if True: #####################################################################################

                pcU3_C = self.__create_string_mean_std(percentageChangeFunction(database[self.sheet_names[i]]["TotalU3"],
                            database[self.sheet_names[i - 1]]["TotalU3"], alpha=0.5))
                pcBrady_C = self.__create_string_mean_std(percentageChangeFunction(database[self.sheet_names[i]]["Brady14Items"],
                            database[self.sheet_names[i - 1]]["Brady14Items"], alpha=0.5))
                pcRigi_C = self.__create_string_mean_std(percentageChangeFunction(database[self.sheet_names[i]]["LimbsRigidity5Items"],
                            database[self.sheet_names[i - 1]]["LimbsRigidity5Items"], alpha=0.5))
                pcTremor_C = self.__create_string_mean_std(percentageChangeFunction(database[self.sheet_names[i]]["LimbsRestTrem"],
                            database[self.sheet_names[i - 1]]["LimbsRestTrem"], alpha=0.5))
                stats_table[f"{group_label}-UPDRSMetrics"][f"% Change {i}"] = [pcU3_C, pcBrady_C, pcRigi_C, pcTremor_C, "NA"]

                # Mean diff and CI
                u3C = self.__create_string_meandiff_ci(database[self.sheet_names[i]]["TotalU3"], database[self.sheet_names[i - 1]]["TotalU3"])
                bradyC = self.__create_string_meandiff_ci(database[self.sheet_names[i]]["Brady14Items"], database[self.sheet_names[i - 1]]["Brady14Items"])
                rigiC = self.__create_string_meandiff_ci(database[self.sheet_names[i]]["LimbsRigidity5Items"], database[self.sheet_names[i - 1]]["LimbsRigidity5Items"])
                tremorC = self.__create_string_meandiff_ci(database[self.sheet_names[i]]["LimbsRestTrem"], database[self.sheet_names[i - 1]]["LimbsRestTrem"])
                stats_table[f"{group_label}-UPDRSMetrics"][f"MeansDiff (CI) {i}"] = [u3C, bradyC, rigiC, tremorC, "NA"]

                # Cohen's d and CI
                u3C = self.__create_string_cohen_d_ci(database[self.sheet_names[i]]["TotalU3"], database[self.sheet_names[i - 1]]["TotalU3"])
                bradyC = self.__create_string_cohen_d_ci(database[self.sheet_names[i]]["Brady14Items"], database[self.sheet_names[i - 1]]["Brady14Items"])
                rigiC = self.__create_string_cohen_d_ci(database[self.sheet_names[i]]["LimbsRigidity5Items"], database[self.sheet_names[i - 1]]["LimbsRigidity5Items"])
                tremorC = self.__create_string_cohen_d_ci(database[self.sheet_names[i]]["LimbsRestTrem"], database[self.sheet_names[i - 1]]["LimbsRestTrem"])
                stats_table[f"{group_label}-UPDRSMetrics"][f"Cohen's d (CI) {i}"] = [u3C, bradyC, rigiC, tremorC, "NA"]

                # p-value
                u3C = self.__paired_ttest(database[self.sheet_names[i]]["TotalU3"], database[self.sheet_names[i - 1]]["TotalU3"])
                bradyC = self.__paired_ttest(database[self.sheet_names[i]]["Brady14Items"], database[self.sheet_names[i - 1]]["Brady14Items"])
                rigiC = self.__paired_ttest(database[self.sheet_names[i]]["LimbsRigidity5Items"], database[self.sheet_names[i - 1]]["LimbsRigidity5Items"])
                tremorC = self.__paired_ttest(database[self.sheet_names[i]]["LimbsRestTrem"], database[self.sheet_names[i - 1]]["LimbsRestTrem"])
                stats_table[f"{group_label}-UPDRSMetrics"][f"p-value {i}"] = [u3C, bradyC, rigiC, tremorC, "NA"]

        p_value_names = ["p-value Baseline-Year1 OFF", "p-value Baseline-Year2 OFF", "p-value Year1-Year2 OFF",
                         "p-value Baseline-Year1 ON", "p-value Baseline-Year2 ON", "p-value Year1-Year2 ON"]
        sheet_pairs = [[0, 2], [0, 4], [2, 4], [1, 3], [1, 5], [3, 5]]
        for i, name in enumerate(p_value_names):
            stats_table[f"{group_label}-UPDRSMetrics"][name] = [
                self.__paired_ttest(database[self.sheet_names[sheet_pairs[i][0]]]["TotalU3"], database[self.sheet_names[sheet_pairs[i][1]]]["TotalU3"]),
                self.__paired_ttest(database[self.sheet_names[sheet_pairs[i][0]]]["Brady14Items"], database[self.sheet_names[sheet_pairs[i][1]]]["Brady14Items"]),
                self.__paired_ttest(database[self.sheet_names[sheet_pairs[i][0]]]["LimbsRigidity5Items"], database[self.sheet_names[sheet_pairs[i][1]]]["LimbsRigidity5Items"]),
                self.__paired_ttest(database[self.sheet_names[sheet_pairs[i][0]]]["LimbsRestTrem"], database[self.sheet_names[sheet_pairs[i][1]]]["LimbsRestTrem"]),
                "NA"
            ]

        p_value_names = ["cohen's d Baseline-Year1 OFF", "cohen's d Baseline-Year2 OFF", "cohen's d Year1-Year2 OFF",
                         "cohen's d Baseline-Year1 ON", "cohen's d Baseline-Year2 ON", "cohen's d Year1-Year2 ON"]
        sheet_pairs = [[0, 2], [0, 4], [2, 4], [1, 3], [1, 5], [3, 5]]
        for i, name in enumerate(p_value_names):
            stats_table[f"{group_label}-UPDRSMetrics"][name] = [
                self.__create_string_cohen_d_ci(database[self.sheet_names[sheet_pairs[i][0]]]["TotalU3"],
                                         database[self.sheet_names[sheet_pairs[i][1]]]["TotalU3"]),
                self.__create_string_cohen_d_ci(database[self.sheet_names[sheet_pairs[i][0]]]["Brady14Items"],
                                         database[self.sheet_names[sheet_pairs[i][1]]]["Brady14Items"]),
                self.__create_string_cohen_d_ci(database[self.sheet_names[sheet_pairs[i][0]]]["LimbsRigidity5Items"],
                                         database[self.sheet_names[sheet_pairs[i][1]]]["LimbsRigidity5Items"]),
                self.__create_string_cohen_d_ci(database[self.sheet_names[sheet_pairs[i][0]]]["LimbsRestTrem"],
                                         database[self.sheet_names[sheet_pairs[i][1]]]["LimbsRestTrem"]),
                "NA"
            ]

        with pd.ExcelWriter(os.path.join(results_folder, "table_summary_stats.xlsx"), engine='openpyxl') as writer:
            for sheet_name, data in stats_table.items():
                data.to_excel(writer, sheet_name=sheet_name, index=False)

        return 0

    def create_stats_table(self):
        percentageChangeFunction = self.percentage_change_basic
        if self.trem_filtered_flag:
            analysis_name = f"Stats_{self.filtered_string}"
        else:
            analysis_name = f"Stats"
        results_folder = self.results_stats
        Path(os.path.join(results_folder)).mkdir(parents=True, exist_ok=True)

        database = copy.deepcopy(self.data)

        stats_table = dict()
        stats_table["HandsTremorFiltered"] = pd.DataFrame()
        stats_table["Complete(No-NaNs)"] = pd.DataFrame()
        stats_table["SampleMetrics"] = pd.DataFrame()
        stats_table["DopamineResponsiveness"] = pd.DataFrame()

        # Get Idx for HandsTremorFiltered
        idxCommon = self.get_subjects_above_threshold(["U17ResTremRUE", "U17ResTremLUE"], self.thresh_trem_filter,
                        database[self.sheet_names[0]], database[self.sheet_names[2]], database[self.sheet_names[4]])
        handsTremorFiltered = dict()

        # Set names for measurements
        stats_table["Complete(No-NaNs)"]["Clinical ratings and quantified measurements"] = [
            "MDS-UPDRS part III",
            "MDS-UPDRS-Brad",
            "MDS-UPDRS-Rig",
            "MDS-UPDRS-RestTrem",
            "Tremor power, rest, uV2"
        ]
        stats_table["HandsTremorFiltered"]["Clinical ratings and quantified measurements"] = [
            "MDS-UPDRS part III",
            "MDS-UPDRS-Brad",
            "MDS-UPDRS-Rig",
            "MDS-UPDRS-RestTrem",
            "Tremor power, rest, uV2"
        ]

        stats_table[f"DopamineResponsiveness"]["Clinical ratings"] = [
            "MDS-UPDRS part III, mean (range)",
            "MDS-UPDRS-ChangeBrad, mean (range)",
            "MDS-UPDRS-ChangeRig, mean (range)",
            "MDS-UPDRS-RespRestTrem, mean (range)"
        ]
        names_columns = ["Baseline", "Year 1", "Year 2"]
        for i, name in enumerate(names_columns):
            stats_table["DopamineResponsiveness"][name] = [
                f"{np.mean(database[self.sheet_names[2 * i]]['ChangeTotalU3']):.3f} ({np.min(database[self.sheet_names[2 * i]]['ChangeTotalU3']):.3f}-{np.max(database[self.sheet_names[2 * i]]['ChangeTotalU3']):.3f})",
                f"{np.mean(database[self.sheet_names[2 * i]]['ChangeBrady14Items']):.3f} ({np.min(database[self.sheet_names[2 * i]]['ChangeBrady14Items']):.3f}-{np.max(database[self.sheet_names[2 * i]]['ChangeBrady14Items']):.3f})",
                f"{np.mean(database[self.sheet_names[2 * i]]['ChangeLimbsRigidity5Items']):.3f} ({np.min(database[self.sheet_names[2 * i]]['ChangeLimbsRigidity5Items']):.3f}-{np.max(database[self.sheet_names[2 * i]]['ChangeLimbsRigidity5Items']):.3f})",
                f"{np.mean(database[self.sheet_names[2 * i]]['ResponseRestTrem']):.3f} ({np.min(database[self.sheet_names[2 * i]]['ResponseRestTrem']):.3f}-{np.max(database[self.sheet_names[2 * i]]['ResponseRestTrem']):.3f})",
            ]
        p_value_names = ["p-value Baseline-Year1", "p-value Baseline-Year2", "p-value Year1-Year2"]
        sheet_pairs = [[0, 2], [0, 4], [2, 4]]
        for i, name in enumerate(p_value_names):
            stats_table["DopamineResponsiveness"][name] = [
                self.__paired_ttest(database[self.sheet_names[sheet_pairs[i][0]]]["ChangeTotalU3"],
                                  database[self.sheet_names[sheet_pairs[i][1]]]["ChangeTotalU3"]),
                self.__paired_ttest(database[self.sheet_names[sheet_pairs[i][0]]]["ChangeBrady14Items"],
                                  database[self.sheet_names[sheet_pairs[i][1]]]["ChangeBrady14Items"]),
                self.__paired_ttest(database[self.sheet_names[sheet_pairs[i][0]]]["ChangeLimbsRigidity5Items"],
                                  database[self.sheet_names[sheet_pairs[i][1]]]["ChangeLimbsRigidity5Items"]),
                self.__paired_ttest(database[self.sheet_names[sheet_pairs[i][0]]]["ResponseRestTrem"],
                                  database[self.sheet_names[sheet_pairs[i][1]]]["ResponseRestTrem"])
            ]

        # For the population metrics table
        visit_names = ["Baseline", "Year 1", "Year 2"]
        for i, visit_name in enumerate(visit_names):
            stats_table["SampleMetrics"][f"Metric for {visit_name}"] = [
                "Age, y",
                "Sex, M/F",
                "H&Y stage, median",
                "Disease duration, y",
                "LED",
            ]
            # This scores are for participants that have data, no filtering yet
            stats_table["SampleMetrics"][f"Values for {visit_name}"] = [
                f"{np.mean(database[self.sheet_names[2 * i]]['Age']):.3f} ({np.min(database[self.sheet_names[2 * i]]['Age'])}-{np.max(database[self.sheet_names[2 * i]]['Age'])})",
                f"{(database[self.sheet_names[2 * i]]['Gender'] == 1).sum()}/{(database[self.sheet_names[2 * i]]['Gender'] == 2).sum()}",
                f"{np.median(database[self.sheet_names[2 * i]]['HoeYah'])} ({np.min(database[self.sheet_names[2 * i]]['HoeYah'])}-{np.max(database[self.sheet_names[2 * i]]['HoeYah'])})",
                f"{np.mean(database[self.sheet_names[2 * i]]['MonthSinceDiag']) / 12:.3f} ({np.min(database[self.sheet_names[2 * i]]['MonthSinceDiag']) / 12:.3f} - {np.max(database[self.sheet_names[2 * i]]['MonthSinceDiag']) / 12:.3f})",
                f"{np.mean(database[self.sheet_names[2 * i +1]]['LEDD']):.3f} ({np.min(database[self.sheet_names[2 * i +1]]['LEDD']):.3f}-{np.max(database[self.sheet_names[2 * i +1]]['LEDD']):.3f})"
            ]

        # For all the other metrics table
        for sheet, data in database.items():
            handsTremorFiltered[sheet] = data.iloc[list(idxCommon)].reset_index(drop=True)

        for i, sheet in enumerate(database.keys()):
            totalsU3_C = self.__create_string_mean_std(database[sheet]["TotalU3"])
            totalsU3_F = self.__create_string_mean_std(handsTremorFiltered[sheet]["TotalU3"])
            brady_C = self.__create_string_mean_std(database[sheet]["Brady14Items"])
            brady_F = self.__create_string_mean_std(handsTremorFiltered[sheet]["Brady14Items"])
            rigi_C = self.__create_string_mean_std(database[sheet]["LimbsRigidity5Items"])
            rigi_F = self.__create_string_mean_std(handsTremorFiltered[sheet]["LimbsRigidity5Items"])
            tremor_C = self.__create_string_mean_std(database[sheet]["LimbsRestTrem"])
            tremor_F = self.__create_string_mean_std(handsTremorFiltered[sheet]["LimbsRestTrem"])
            power_C = self.__create_string_mean_std(database[sheet]["LogPower"])
            power_F = self.__create_string_mean_std(handsTremorFiltered[sheet]["LogPower"])
            stats_table["Complete(No-NaNs)"][sheet] = [totalsU3_C, brady_C, rigi_C, tremor_C, power_C]
            stats_table["HandsTremorFiltered"][sheet] = [totalsU3_F, brady_F, rigi_F, tremor_F, power_F]

            if i in [1, 3, 5]:
                pcU3_C = self.__create_string_mean_std(percentageChangeFunction(database[self.sheet_names[i]]["TotalU3"], 
                            database[self.sheet_names[i - 1]]["TotalU3"], alpha=0.5))
                pcU3_F = self.__create_string_mean_std(percentageChangeFunction(handsTremorFiltered[self.sheet_names[i]]["TotalU3"],
                            handsTremorFiltered[self.sheet_names[i - 1]]["TotalU3"], alpha=0.5))
                pcBrady_C = self.__create_string_mean_std(percentageChangeFunction(database[self.sheet_names[i]]["Brady14Items"],
                            database[self.sheet_names[i - 1]]["Brady14Items"], alpha=0.5))
                pcBrady_F = self.__create_string_mean_std(percentageChangeFunction(handsTremorFiltered[self.sheet_names[i]]["Brady14Items"],
                            handsTremorFiltered[self.sheet_names[i - 1]]["Brady14Items"], alpha=0.5))
                pcRigi_C = self.__create_string_mean_std(percentageChangeFunction(database[self.sheet_names[i]]["LimbsRigidity5Items"],
                            database[self.sheet_names[i - 1]]["LimbsRigidity5Items"], alpha=0.5))
                pcRigi_F = self.__create_string_mean_std(percentageChangeFunction(handsTremorFiltered[self.sheet_names[i]]["LimbsRigidity5Items"],
                            handsTremorFiltered[self.sheet_names[i - 1]]["LimbsRigidity5Items"], alpha=0.5))
                pcTremor_C = self.__create_string_mean_std(percentageChangeFunction(database[self.sheet_names[i]]["LimbsRestTrem"],
                            database[self.sheet_names[i - 1]]["LimbsRestTrem"], alpha=0.5))
                pcTremor_F = self.__create_string_mean_std(percentageChangeFunction(handsTremorFiltered[self.sheet_names[i]]["LimbsRestTrem"],
                            handsTremorFiltered[self.sheet_names[i - 1]]["LimbsRestTrem"], alpha=0.5))
                stats_table["Complete(No-NaNs)"][f"% Change {i}"] = [pcU3_C, pcBrady_C, pcRigi_C, pcTremor_C, "NA"]
                stats_table["HandsTremorFiltered"][f"% Change {i}"] = [pcU3_F, pcBrady_F, pcRigi_F, pcTremor_F, "NA"]

                # Mean diff and CI
                u3C = self.__create_string_meandiff_ci(database[self.sheet_names[i]]["TotalU3"], database[self.sheet_names[i - 1]]["TotalU3"])
                u3F = self.__create_string_meandiff_ci(handsTremorFiltered[self.sheet_names[i]]["TotalU3"], handsTremorFiltered[self.sheet_names[i - 1]]["TotalU3"])
                bradyC = self.__create_string_meandiff_ci(database[self.sheet_names[i]]["Brady14Items"], database[self.sheet_names[i - 1]]["Brady14Items"])
                bradyF = self.__create_string_meandiff_ci(handsTremorFiltered[self.sheet_names[i]]["Brady14Items"], handsTremorFiltered[self.sheet_names[i - 1]]["Brady14Items"])
                rigiC = self.__create_string_meandiff_ci(database[self.sheet_names[i]]["LimbsRigidity5Items"], database[self.sheet_names[i - 1]]["LimbsRigidity5Items"])
                rigiF = self.__create_string_meandiff_ci(handsTremorFiltered[self.sheet_names[i]]["LimbsRigidity5Items"], handsTremorFiltered[self.sheet_names[i - 1]]["LimbsRigidity5Items"])
                tremorC = self.__create_string_meandiff_ci(database[self.sheet_names[i]]["LimbsRestTrem"], database[self.sheet_names[i - 1]]["LimbsRestTrem"])
                tremorF = self.__create_string_meandiff_ci(handsTremorFiltered[self.sheet_names[i]]["LimbsRestTrem"], handsTremorFiltered[self.sheet_names[i - 1]]["LimbsRestTrem"])
                stats_table["Complete(No-NaNs)"][f"MeansDiff (CI) {i}"] = [u3C, bradyC, rigiC, tremorC, "NA"]
                stats_table["HandsTremorFiltered"][f"MeansDiff (CI) {i}"] = [u3F, bradyF, rigiF, tremorF, "NA"]

                # Cohen's d and CI
                u3C = self.__create_string_cohen_d_ci(database[self.sheet_names[i]]["TotalU3"], database[self.sheet_names[i - 1]]["TotalU3"])
                u3F = self.__create_string_cohen_d_ci(handsTremorFiltered[self.sheet_names[i]]["TotalU3"], handsTremorFiltered[self.sheet_names[i - 1]]["TotalU3"])
                bradyC = self.__create_string_cohen_d_ci(database[self.sheet_names[i]]["Brady14Items"], database[self.sheet_names[i - 1]]["Brady14Items"])
                bradyF = self.__create_string_cohen_d_ci(handsTremorFiltered[self.sheet_names[i]]["Brady14Items"], handsTremorFiltered[self.sheet_names[i - 1]]["Brady14Items"])
                rigiC = self.__create_string_cohen_d_ci(database[self.sheet_names[i]]["LimbsRigidity5Items"], database[self.sheet_names[i - 1]]["LimbsRigidity5Items"])
                rigiF = self.__create_string_cohen_d_ci(handsTremorFiltered[self.sheet_names[i]]["LimbsRigidity5Items"], handsTremorFiltered[self.sheet_names[i - 1]]["LimbsRigidity5Items"])
                tremorC = self.__create_string_cohen_d_ci(database[self.sheet_names[i]]["LimbsRestTrem"], database[self.sheet_names[i - 1]]["LimbsRestTrem"])
                tremorF = self.__create_string_cohen_d_ci(handsTremorFiltered[self.sheet_names[i]]["LimbsRestTrem"], handsTremorFiltered[self.sheet_names[i - 1]]["LimbsRestTrem"])
                stats_table["Complete(No-NaNs)"][f"Cohen's d (CI) {i}"] = [u3C, bradyC, rigiC, tremorC, "NA"]
                stats_table["HandsTremorFiltered"][f"Cohen's d (CI) {i}"] = [u3F, bradyF, rigiF, tremorF, "NA"]

                # p-value
                u3C = self.__paired_ttest(database[self.sheet_names[i]]["TotalU3"], database[self.sheet_names[i - 1]]["TotalU3"])
                u3F = self.__paired_ttest(handsTremorFiltered[self.sheet_names[i]]["TotalU3"], handsTremorFiltered[self.sheet_names[i - 1]]["TotalU3"])
                bradyC = self.__paired_ttest(database[self.sheet_names[i]]["Brady14Items"], database[self.sheet_names[i - 1]]["Brady14Items"])
                bradyF = self.__paired_ttest(handsTremorFiltered[self.sheet_names[i]]["Brady14Items"], handsTremorFiltered[self.sheet_names[i - 1]]["Brady14Items"])
                rigiC = self.__paired_ttest(database[self.sheet_names[i]]["LimbsRigidity5Items"], database[self.sheet_names[i - 1]]["LimbsRigidity5Items"])
                rigiF = self.__paired_ttest(handsTremorFiltered[self.sheet_names[i]]["LimbsRigidity5Items"], handsTremorFiltered[self.sheet_names[i - 1]]["LimbsRigidity5Items"])
                tremorC = self.__paired_ttest(database[self.sheet_names[i]]["LimbsRestTrem"], database[self.sheet_names[i - 1]]["LimbsRestTrem"])
                tremorF = self.__paired_ttest(handsTremorFiltered[self.sheet_names[i]]["LimbsRestTrem"], handsTremorFiltered[self.sheet_names[i - 1]]["LimbsRestTrem"])
                stats_table["Complete(No-NaNs)"][f"p-value {i}"] = [u3C, bradyC, rigiC, tremorC, "NA"]
                stats_table["HandsTremorFiltered"][f"p-value {i}"] = [u3F, bradyF, rigiF, tremorF, "NA"]

        p_value_names = ["p-value Baseline-Year1 OFF", "p-value Baseline-Year2 OFF", "p-value Year1-Year2 OFF",
                         "p-value Baseline-Year1 ON", "p-value Baseline-Year2 ON", "p-value Year1-Year2 ON"]
        sheet_pairs = [[0, 2], [0, 4], [2, 4], [1, 3], [1, 5], [3, 5]]
        for i, name in enumerate(p_value_names):
            stats_table["Complete(No-NaNs)"][name] = [
                self.__paired_ttest(database[self.sheet_names[sheet_pairs[i][0]]]["TotalU3"], database[self.sheet_names[sheet_pairs[i][1]]]["TotalU3"]),
                self.__paired_ttest(database[self.sheet_names[sheet_pairs[i][0]]]["Brady14Items"], database[self.sheet_names[sheet_pairs[i][1]]]["Brady14Items"]),
                self.__paired_ttest(database[self.sheet_names[sheet_pairs[i][0]]]["LimbsRigidity5Items"], database[self.sheet_names[sheet_pairs[i][1]]]["LimbsRigidity5Items"]),
                self.__paired_ttest(database[self.sheet_names[sheet_pairs[i][0]]]["LimbsRestTrem"], database[self.sheet_names[sheet_pairs[i][1]]]["LimbsRestTrem"]),
                "NA"
            ]
            stats_table["HandsTremorFiltered"][name] = [
                self.__paired_ttest(handsTremorFiltered[self.sheet_names[sheet_pairs[i][0]]]["TotalU3"], handsTremorFiltered[self.sheet_names[sheet_pairs[i][1]]]["TotalU3"]),
                self.__paired_ttest(handsTremorFiltered[self.sheet_names[sheet_pairs[i][0]]]["Brady14Items"], handsTremorFiltered[self.sheet_names[sheet_pairs[i][1]]]["Brady14Items"]),
                self.__paired_ttest(handsTremorFiltered[self.sheet_names[sheet_pairs[i][0]]]["LimbsRigidity5Items"], handsTremorFiltered[self.sheet_names[sheet_pairs[i][1]]]["LimbsRigidity5Items"]),
                self.__paired_ttest(handsTremorFiltered[self.sheet_names[sheet_pairs[i][0]]]["LimbsRestTrem"], handsTremorFiltered[self.sheet_names[sheet_pairs[i][1]]]["LimbsRestTrem"]),
                "NA"
            ]

        p_value_names = ["cohen's d Baseline-Year1 OFF", "cohen's d Baseline-Year2 OFF", "cohen's d Year1-Year2 OFF",
                         "cohen's d Baseline-Year1 ON", "cohen's d Baseline-Year2 ON", "cohen's d Year1-Year2 ON"]
        sheet_pairs = [[0, 2], [0, 4], [2, 4], [1, 3], [1, 5], [3, 5]]
        for i, name in enumerate(p_value_names):
            stats_table["Complete(No-NaNs)"][name] = [
                self.__create_string_cohen_d_ci(database[self.sheet_names[sheet_pairs[i][0]]]["TotalU3"],
                                         database[self.sheet_names[sheet_pairs[i][1]]]["TotalU3"]),
                self.__create_string_cohen_d_ci(database[self.sheet_names[sheet_pairs[i][0]]]["Brady14Items"],
                                         database[self.sheet_names[sheet_pairs[i][1]]]["Brady14Items"]),
                self.__create_string_cohen_d_ci(database[self.sheet_names[sheet_pairs[i][0]]]["LimbsRigidity5Items"],
                                         database[self.sheet_names[sheet_pairs[i][1]]]["LimbsRigidity5Items"]),
                self.__create_string_cohen_d_ci(database[self.sheet_names[sheet_pairs[i][0]]]["LimbsRestTrem"],
                                         database[self.sheet_names[sheet_pairs[i][1]]]["LimbsRestTrem"]),
                "NA"
            ]
            stats_table["HandsTremorFiltered"][name] = [
                self.__create_string_cohen_d_ci(handsTremorFiltered[self.sheet_names[sheet_pairs[i][0]]]["TotalU3"],
                                         handsTremorFiltered[self.sheet_names[sheet_pairs[i][1]]]["TotalU3"]),
                self.__create_string_cohen_d_ci(handsTremorFiltered[self.sheet_names[sheet_pairs[i][0]]]["Brady14Items"],
                                         handsTremorFiltered[self.sheet_names[sheet_pairs[i][1]]]["Brady14Items"]),
                self.__create_string_cohen_d_ci(handsTremorFiltered[self.sheet_names[sheet_pairs[i][0]]]["LimbsRigidity5Items"],
                                         handsTremorFiltered[self.sheet_names[sheet_pairs[i][1]]]["LimbsRigidity5Items"]),
                self.__create_string_cohen_d_ci(handsTremorFiltered[self.sheet_names[sheet_pairs[i][0]]]["LimbsRestTrem"],
                                         handsTremorFiltered[self.sheet_names[sheet_pairs[i][1]]]["LimbsRestTrem"]),
                "NA"
            ]

        with pd.ExcelWriter(os.path.join(results_folder, "table_summary_stats.xlsx"), engine='openpyxl') as writer:
            for sheet_name, data in stats_table.items():
                data.to_excel(writer, sheet_name=sheet_name, index=False)

        return 0
    
    def __create_string_cohen_d_ci(self, data1, data2):
        z = 1.96  # Critical value z for a 95% confidence
        confidence_level = 0.95
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        n1 = len(data1)
        n2 = len(data2)
        d = self.__cohens_d(data1, data2)
        std_error_d = np.sqrt(((n1 + n2) / (n1 * n2)) + ((d ** 2) / (2 * (n1 + n2))))
        margin_error = z * std_error_d
        lower_bound = d - margin_error
        upper_bound = d + margin_error
        sumStr = f"{d:.3f} ({lower_bound:.3f}-{upper_bound:.3f})"
        return sumStr
    
    def __cohens_d(self, data1, data2):
        n1 = len(data1)
        n2 = len(data2)
        dof = n1 + n2 - 2
        std_pooled = np.sqrt((((n1 - 1) * (np.std(data1) ** 2)) + ((n2 - 1) * (np.std(data2) ** 2))) / dof)
        d = abs(np.mean(data1) - np.mean(data2)) / std_pooled
        J = self.__hedges_correction(dof)
        return J * d
    
    @staticmethod
    def __create_string_mean_std(data):
        meanKey = np.mean(data)
        stdKey = np.std(data)
        sumStr = f"{meanKey:.3f}\u00B1{stdKey:.3f}"
        return sumStr
    
    @staticmethod
    def __create_string_meandiff_ci(data1, data2):
        mean1, mean2 = np.mean(data1), np.mean(data2)
        mean_diff = mean1 - mean2
        mean_diff = abs(mean_diff)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        se_diff = np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))
        confidence_level = 0.95
        alpha = 1 - confidence_level
        z = stats.norm.ppf(1 - alpha / 2)
        margin_error = z * se_diff
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error

        sumStr = f"{mean_diff:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
        return sumStr
    
    @staticmethod
    def __paired_ttest(data1, data2):
        t_stat, p_value = stats.ttest_rel(data2, data1)
        return f"{p_value:.5f}"
    
    @staticmethod
    def __independent_ttest(data1, data2):
        # Perform the independent t-test
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)  # Use Welch's t-test by default
        return f"{p_value:.5f}"
    
    @staticmethod
    def __hedges_correction(df):
        if df == 0:
            return np.nan
        else:
            J = np.exp(gammaln(df / 2) - np.log(np.sqrt(df / 2)) - gammaln((df - 1) / 2))
            return J
    
    @staticmethod
    def __pairwise_ttests_posthoc(dataFrame, scores_col, visit_col, condition_col, subject_col, p_adjust_method='bonferroni'):
        all_pairwise_results = []

        # Part 1: Comparisons between clusters per visit
        for visit in dataFrame[visit_col].unique():
            visit_df = dataFrame[dataFrame[visit_col] == visit]
            pairwise_results_clusters = pg.pairwise_tests(dv=scores_col, between=condition_col, subject=subject_col,
                                                          data=visit_df, padjust=p_adjust_method)
            pairwise_results_clusters[visit_col] = visit
            all_pairwise_results.append(pairwise_results_clusters)

        # Part 2: Comparisons between visits per cluster
        for cluster in dataFrame[condition_col].unique():
            cluster_df = dataFrame[dataFrame[condition_col] == cluster]
            pairwise_results_visits = pg.pairwise_tests(dv=scores_col, within=visit_col, subject=subject_col,
                                                        data=cluster_df, padjust=p_adjust_method)
            pairwise_results_visits[condition_col] = cluster
            all_pairwise_results.append(pairwise_results_visits)

        # Combine all results into a single DataFrame
        final_pairwise_results_df = pd.concat(all_pairwise_results, ignore_index=True)

        return final_pairwise_results_df

    @staticmethod
    def __pairwise_ttests_posthoc_3factors(dataFrame, scores_col, visit_col, condition_col, subject_col, session_col,
                                  p_adjust_method='bonferroni'):
        all_pairwise_results = []

        # Part 1: Comparisons between clusters per visit and session
        for visit in dataFrame[visit_col].unique():
            for session in dataFrame[session_col].unique():
                visit_session_df = dataFrame[(dataFrame[visit_col] == visit) & (dataFrame[session_col] == session)]
                if len(visit_session_df[condition_col].unique()) > 1:  # Ensure there are multiple groups to compare
                    pairwise_results_clusters = pg.pairwise_tests(dv=scores_col, between=condition_col, subject=subject_col,
                                                                  data=visit_session_df, padjust=p_adjust_method)
                    pairwise_results_clusters[visit_col] = visit
                    pairwise_results_clusters[session_col] = session
                    all_pairwise_results.append(pairwise_results_clusters)

        # Part 2: Comparisons between visits per cluster and session
        for cluster in dataFrame[condition_col].unique():
            for session in dataFrame[session_col].unique():
                cluster_session_df = dataFrame[(dataFrame[condition_col] == cluster) & (dataFrame[session_col] == session)]
                if len(cluster_session_df[visit_col].unique()) > 1:  # Ensure there are multiple visits to compare
                    pairwise_results_visits = pg.pairwise_tests(dv=scores_col, within=visit_col, subject=subject_col,
                                                                data=cluster_session_df, padjust=p_adjust_method)
                    pairwise_results_visits[condition_col] = cluster
                    pairwise_results_visits[session_col] = session
                    all_pairwise_results.append(pairwise_results_visits)

        # Part 3: Comparisons between sessions within each visit and cluster
        for cluster in dataFrame[condition_col].unique():
            for visit in dataFrame[visit_col].unique():
                visit_cluster_df = dataFrame[(dataFrame[condition_col] == cluster) & (dataFrame[visit_col] == visit)]
                if len(visit_cluster_df[session_col].unique()) > 1:  # Ensure there are multiple sessions to compare
                    pairwise_results_sessions = pg.pairwise_tests(dv=scores_col, within=session_col, subject=subject_col,
                                                                  data=visit_cluster_df, padjust=p_adjust_method)
                    pairwise_results_sessions[condition_col] = cluster
                    pairwise_results_sessions[visit_col] = visit
                    all_pairwise_results.append(pairwise_results_sessions)

        # Combine all results into a single DataFrame
        final_pairwise_results_df = pd.concat(all_pairwise_results, ignore_index=True)

        return final_pairwise_results_df