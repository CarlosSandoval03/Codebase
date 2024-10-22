import os
import sys
import copy
from pathlib import Path
import numpy as np
from numpy import unique
from numpy import where
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from scipy import stats
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import matplotlib
# matplotlib.use('TkAgg')  # or 'Agg'
import matplotlib.pyplot as plt

from utilitiespkg import DataHandling
from clusteringpkg import Clustering

class UPDRSPlotting(Clustering, DataHandling):
    def __init__(self, data, database_v, trem_filtered=True):
        super().__init__(data, database_v, trem_filtered=trem_filtered)
        print(" --- UPDRS PLOTTING CLASS INITIALIZATION --- ")

        self.percentage_change_function = self.percentage_change_basic
        self.percentage_change_function_name = "Basic"

    def create_histograms(self, updrs_keys, visits, sessions, FilteredByHandsTremorFlag=False):
        """
        CREATES HISTOGRAMS FOR THE UPDRS SCORES SPECIFIED IN updrs_keys.
        Args:
            updrs_keys: Column names that will be plotted.
            visits: Visits that will be plotted.
            sessions: Session that will be plotted.
            FilteredByHandsTremorFlag: Indicate whether the subjects without tremor will be filtered out. If the database was already filtered it has no effect.

        Returns: 0. It creates the figures and saves them to the results folder.

        """
        includeGaussianCurve = True
        style = "plotly"
        typeC = "singleVisit"

        FilteredByHandsTremorFlag_noaction = False
        if ~self.trem_filtered_flag and FilteredByHandsTremorFlag:
            FilteredByHandsTremorFlag = FilteredByHandsTremorFlag
            FilteredByHandsTremorFlag_noaction = FilteredByHandsTremorFlag
        else:
            FilteredByHandsTremorFlag = False

        if FilteredByHandsTremorFlag_noaction:
            analysis_name = f"Histograms_{self.filtered_string}"
        else:
            analysis_name = f"Histograms"

        results_folder = os.path.join(self.results_folder, analysis_name)

        nbins = 15
        visits = [v - 1 for v in visits]
        database_c = copy.deepcopy(self.data)
        Path(os.path.join(results_folder)).mkdir(parents=True, exist_ok=True)
        for updrs_key in updrs_keys:
            for ises, ses in enumerate(sessions):
                for visit in visits:
                    if self.get_database_version() == "ppp":
                        if FilteredByHandsTremorFlag:
                            idx = self.get_subjects_above_threshold(["U17ResTremRUE", "U17ResTremLUE"], self.thresh_trem_filter, database_c[self.sheet_names[2 * visit]])
                            data_serie = pd.Series([database_c[self.sheet_names[2 * visit + ises]][updrs_key][i] for i in idx])
                        else:
                            data_serie = database_c[self.sheet_names[2 * visit + ises]][updrs_key]
                    elif self.get_database_version() == "drdr":
                        if FilteredByHandsTremorFlag:
                            idx = self.get_subjects_above_threshold(["OFFU17RUE", "OFFU17LUE"], self.thresh_trem_filter, database_c[self.sheet_names[0]])
                            if isinstance(updrs_key, list):
                                data_serie = pd.Series([database_c[self.sheet_names[ises]][updrs_key[ises]][i] for i in idx])
                            else:
                                data_serie = pd.Series([database_c[self.sheet_names[ises]][updrs_key][i] for i in idx])
                        else:
                            if isinstance(updrs_key, list) == True:
                                data_serie = database_c[self.sheet_names[ises]][updrs_key[ises]]
                            else:
                                data_serie = database_c[self.sheet_names[ises]][updrs_key]
                    else:
                        raise ValueError("Select a correct dataset. Either ppp or drdr.")

                    cleaned_Args = self.remove_rows_with_nans(data_serie)
                    data_serie = cleaned_Args[0]

                    changes_data = {"temp": data_serie}

                    self.__histogram_backend(changes_data, updrs_key, visit, nbins, results_folder, style=style,
                                             typeC=typeC, ses=ses, includeGaussianCurve=includeGaussianCurve)
        return 0

    def plot_percentage_change(self, updrs_keys, visits, type_comparison="offon", FilteredByHandsTremorFlag=False):
        """
        CREATES PERCENTAGE-CHANGE HISTOGRAMS.
        Args:
            updrs_keys: List of column names that will be analyzed.
            visits: The visits that whose data will be plotted.
            type_comparison: Selection of comparison, Either "offon" or "longitudinal".
            FilteredByHandsTremorFlag: Indicate whether the subjects without tremor will be filtered out.
            If the database was already filtered it has no effect.

        Returns: 0. Creates the figures in the results folder.

        """
        includeGaussianCurve = True
        style = "plotly"

        FilteredByHandsTremorFlag_noaction = False
        if ~self.trem_filtered_flag and FilteredByHandsTremorFlag:
            FilteredByHandsTremorFlag = FilteredByHandsTremorFlag
            FilteredByHandsTremorFlag_noaction = FilteredByHandsTremorFlag
        else:
            FilteredByHandsTremorFlag = False

        if FilteredByHandsTremorFlag_noaction:
            analysis_name = f"Percentage-Change_{self.filtered_string}"
        else:
            analysis_name = f"Percentage-Change"

        results_folder = os.path.join(self.results_folder, analysis_name)

        nbins = 10
        visits = [v - 1 for v in visits]
        database_c = copy.deepcopy(self.data)
        Path(os.path.join(results_folder)).mkdir(parents=True, exist_ok=True)
        for updrs_key in updrs_keys:
            if type_comparison == "offon":
                for visit in visits:
                    if self.get_database_version() == "ppp":
                        if FilteredByHandsTremorFlag:
                            idx = self.get_subjects_above_threshold(["U17ResTremRUE", "U17ResTremLUE"], self.thresh_trem_filter,
                                                            database_c[self.sheet_names[2 * visit]])
                            offD = pd.Series([database_c[self.sheet_names[2 * visit]][updrs_key][i] for i in idx])
                            onD = pd.Series([database_c[self.sheet_names[2 * visit + 1]][updrs_key][i] for i in idx])
                        else:
                            offD = database_c[self.sheet_names[2 * visit]][updrs_key]
                            onD = database_c[self.sheet_names[2 * visit + 1]][updrs_key]
                    elif self.get_database_version() == "drdr":
                        if FilteredByHandsTremorFlag:
                            idx = self.get_subjects_above_threshold(["OFFU17RUE", "OFFU17LUE"], self.thresh_trem_filter,
                                                                    database_c[self.sheet_names[0]])
                            if isinstance(updrs_key, list):
                                offD = pd.Series([database_c[self.sheet_names[0]][updrs_key[0]][i] for i in idx])
                                onD = pd.Series([database_c[self.sheet_names[1]][updrs_key[1]][i] for i in idx])
                            else:
                                offD = pd.Series([database_c[self.sheet_names[0]][updrs_key][i] for i in idx])
                                onD = pd.Series([database_c[self.sheet_names[1]][updrs_key][i] for i in idx])
                        else:
                            if isinstance(updrs_key, list):
                                offD = database_c[self.sheet_names[0]][updrs_key[0]]
                                onD = database_c[self.sheet_names[1]][updrs_key[1]]
                            else:
                                offD = database_c[self.sheet_names[0]][updrs_key]
                                onD = database_c[self.sheet_names[1]][updrs_key]
                    else:
                        raise ValueError("Select a correct dataset. Either ppp or drdr.")

                    cleaned_Args = self.remove_rows_with_nans(offD, onD)
                    offD, onD = cleaned_Args[0], cleaned_Args[1]

                    changes_data = {"temp": self.percentage_change_function(copy.deepcopy(onD), copy.deepcopy(offD))}

                    self.__histogram_backend(changes_data, updrs_key, visit, nbins, results_folder, style=style,
                                             typeC=type_comparison, includeGaussianCurve=includeGaussianCurve)

            elif type_comparison == "longitudinal":
                if len(visits) <= 1:
                    raise ValueError("In a longitudinal study you need to specify more than 1 session.")
                for ses in ["off", "on"]:
                    if self.get_database_version() == "ppp":
                        if len(visits) > 3:
                            raise ValueError("This database only have 3 visits: 1-Baseline, 2-Year1, and 3-Year 2.")
                        if FilteredByHandsTremorFlag:
                            idx = self.get_subjects_above_threshold(["U17ResTremRUE", "U17ResTremLUE"], self.thresh_trem_filter,
                                                            database_c[self.sheet_names[0]], database_c[self.sheet_names[2]],
                                                            database_c[self.sheet_names[4]])
                            d1 = pd.Series([database_c[self.sheet_names[0]][updrs_key][i] for i in idx])
                            d2 = pd.Series([database_c[self.sheet_names[2]][updrs_key][i] for i in idx])
                            d3 = pd.Series([database_c[self.sheet_names[4]][updrs_key][i] for i in idx])
                        else:
                            d1 = database_c[self.sheet_names[0]][updrs_key]
                            d2 = database_c[self.sheet_names[2]][updrs_key]
                            d3 = database_c[self.sheet_names[4]][updrs_key]
                    elif self.get_database_version() == "drdr":
                        if len(visits) > 1:
                            raise ValueError(
                                "This database can not have longitudinal studies since there is only 1 visit.")
                    else:
                        raise ValueError("Select a correct dataset. Either ppp or drdr.")

                    cleaned_Args = self.remove_rows_with_nans(d1, d2, d3)

                    if len(visits) == 2:
                        changes_data = {"temp": self.percentage_change_function(copy.deepcopy(cleaned_Args[visits[0]]),
                                                                                copy.deepcopy(cleaned_Args[visits[1]]))}
                        self.__histogram_backend(changes_data, updrs_key, visits, nbins, results_folder,
                                                 style, type_comparison, ses, includeGaussianCurve)
                    elif len(visits) == 3:
                        changes_data = {
                            "temp": self.percentage_change_function(cleaned_Args[visits[0]], cleaned_Args[visits[1]])}
                        self.__histogram_backend(changes_data, updrs_key, [visits[0], visits[1]], nbins, results_folder,
                                                 style, type_comparison, ses, includeGaussianCurve)

                        changes_data = {
                            "temp": self.percentage_change_function(cleaned_Args[visits[0]], cleaned_Args[visits[2]])}
                        self.__histogram_backend(changes_data, updrs_key, [visits[0], visits[2]], nbins, results_folder,
                                                 style, type_comparison, ses, includeGaussianCurve)

                        changes_data = {
                            "temp": self.percentage_change_function(cleaned_Args[visits[1]], cleaned_Args[visits[2]])}
                        self.__histogram_backend(changes_data, updrs_key, [visits[1], visits[2]], nbins, results_folder,
                                                 style, type_comparison, ses, includeGaussianCurve)
        return 0

    def plot_scatter(self, conf_dicts, FilteredByHandsTremorFlag=False):
        """
        CREATES SCATTER PLOTS OF THE DATA, INCLUDING A LINEAR REGRESSION.
        Args:
            conf_dicts: List of dictionaries. Each dictionary's settings will generate a different figure. Settings
            include name of the variables for X and Y axis, visits, and sessions to be plotted.
            E.g.: conf_dicts=[{'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [{'off': 'AvgRestTrem', 'on': 'AvgRestTrem'},
            {'off': 'AvgLimbsBradyRig', 'on': 'AvgLimbsBradyRig'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']}]
            FilteredByHandsTremorFlag: Indicate whether the subjects without tremor will be filtered out.
            If the database was already filtered it has no effect.

        Returns: 0. It saves the figures to the results folder.

        """
        style = "plotly"

        FilteredByHandsTremorFlag_noaction = False
        if ~self.trem_filtered_flag and FilteredByHandsTremorFlag:
            FilteredByHandsTremorFlag = FilteredByHandsTremorFlag
            FilteredByHandsTremorFlag_noaction = FilteredByHandsTremorFlag
        else:
            FilteredByHandsTremorFlag = False

        if FilteredByHandsTremorFlag_noaction:
            analysis_name = f"ScatterPlot_{self.filtered_string}"
        else:
            analysis_name = f"ScatterPlot"

        results_folder = os.path.join(self.results_folder, analysis_name)
        Path(os.path.join(results_folder)).mkdir(parents=True, exist_ok=True)

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'black', 'cyan', 'magenta', 'olive', 'pink']

        df = copy.deepcopy(self.data)

        for confs in conf_dicts:
            if len(confs['x']) > 1 and len(confs['x']) != len(confs['y']):
                raise ValueError("Length of X axis element should be 1 or equal to length of Y axis elements.")
            timeName, sesName = self.__get_visit_session_naming(confs['visit'], confs['ses'])

            fig = go.Figure()
            plt.figure(figsize=(10, 6))
            colIdx = 0

            for i_data, dataY in enumerate(confs['y']):
                if len(confs['x']) == len(confs['y']):
                    dataX = confs['x'][i_data]
                else:
                    dataX = confs['x'][0]
                if len(confs['y']) > 1:
                    nameImageExtra = "MultipleMetrics_"
                else:
                    nameImageExtra = ""
                    # fig = go.Figure()
                    # plt.figure(figsize=(10, 6))
                    colIdx = 0

                for visit in confs['visit']:
                    for ses in confs['ses']:
                        if ses == "off":
                            frame = 2 * (visit - 1)
                        elif ses == "on":
                            frame = 2 * (visit - 1) + 1
                        else:
                            raise ValueError(
                                "Session has to be either \"off\" or \"on\". Please check the \"ses\" field in your conf difctionary.")
                        if FilteredByHandsTremorFlag:
                            if self.get_database_version() == "ppp":
                                idx = self.get_subjects_above_threshold(["U17ResTremRUE", "U17ResTremLUE"], 1,
                                                                df[self.sheet_names[2 * (visit - 1)]])
                            elif self.get_database_version() == "drdr":
                                idx = self.get_subjects_above_threshold(["OFFU17RUE", "OFFU17LUE"], 1,
                                                                df[self.sheet_names[2 * (visit - 1)]])
                            xD = pd.Series([df[self.sheet_names[frame]][dataX[ses]][i] for i in idx])
                            if dataX[ses] == 'LEDD':
                                xD = pd.Series([df[self.sheet_names[2 * (visit - 1) + 1]][dataX[ses]][i] for i in idx])
                            yD = pd.Series([df[self.sheet_names[frame]][dataY[ses]][i] for i in idx])
                            if dataY[ses] == 'LEDD':
                                yD = pd.Series([df[self.sheet_names[2 * (visit - 1) + 1]][dataY[ses]][i] for i in idx])
                        else:
                            xD = df[self.sheet_names[frame]][dataX[ses]]
                            if dataX[ses] == 'LEDD':
                                xD = df[self.sheet_names[2 * (visit - 1) + 1]][dataX[ses]]
                            yD = df[self.sheet_names[frame]][dataY[ses]]
                            if dataY[ses] == 'LEDD':
                                yD = df[self.sheet_names[2 * (visit - 1) + 1]][dataY[ses]]

                        xD, yD = self.remove_rows_with_nans(xD, yD)

                        df_p = {}
                        df_p["X"] = xD
                        df_p["Y"] = yD

                        # Calculate and plot regression line
                        slope, intercept, r_value, p_value, std_err = stats.linregress(xD, yD)
                        x_range = np.linspace(min(xD), max(xD), 100)
                        y_range = slope * x_range + intercept
                        if self.get_database_version() == "ppp":
                            if len(confs['y']) > 1:
                                label_name = f'visit{visit}-{ses}_{dataY["off"]}'
                            else:
                                label_name = f'visit{visit}-{ses}'
                        else:
                            if len(confs['y']) > 1:
                                label_name = f'{ses}_{dataY["off"]}'
                            else:
                                label_name = f'{ses}'

                        if style == "all" or style == "plotly":
                            fig.add_trace(go.Scatter(x=xD, y=yD, mode='markers', name=label_name, marker=dict(color=colors[colIdx])))
                            fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', line=dict(color=colors[colIdx], dash='dash'), showlegend=False))
                        if style == "all" or style == "sns":
                            sns.scatterplot(data=df_p, x='X', y='Y', color=colors[colIdx], label=label_name)
                            sns.regplot(data=df_p, x='X', y='Y', scatter=False, color=colors[colIdx],
                                        line_kws={'linestyle': '-', 'linewidth': 2}, ci=None)  # Add ci=None for deleting shaded area around regression lines
                            plt.title('Data Distribution')
                            plt.xlabel(dataX['off'])
                            plt.ylabel(dataY['off'])
                            plt.legend(title='Visits and Sessions')
                        colIdx = colIdx + 1
                if len(confs['y']) == 1:
                    if style == "all" or style == "plotly":
                        fig.update_layout(title='Data Distribution', xaxis_title=dataX['off'], yaxis_title=dataY['off'])
                        if self.get_database_version() == "ppp":
                            figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{timeName}_{sesName}_plotly.png")
                        else:
                            figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{sesName}_plotly.png")
                        fig.write_image(figure_name)
                        del fig
                    if style == "all" or style == "sns":
                        if self.get_database_version() == "ppp":
                            figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{timeName}_{sesName}_sns.png")
                        else:
                            figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{sesName}_sns.png")
                        plt.savefig(figure_name, bbox_inches='tight', dpi=600)
                        plt.clf()
                        plt.close()

            if len(confs['y']) > 1:
                nameImageExtra = "MultipleMetrics_"
                if style == "all" or style == "plotly":
                    fig.update_layout(title='Data Distribution', xaxis_title=dataX['off'], yaxis_title=dataY['off'])
                    if self.get_database_version() == "ppp":
                        figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{timeName}_{sesName}_plotly.png")
                    else:
                        figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{sesName}_plotly.png")
                    fig.write_image(figure_name)
                    del fig
                if style == "all" or style == "sns":
                    if self.get_database_version() == "ppp":
                        figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{timeName}_{sesName}_sns.png")
                    else:
                        figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{sesName}_sns.png")
                    plt.savefig(figure_name, bbox_inches='tight', dpi=600)
                    plt.clf()
                    plt.close()

        return 0

    def plot_scatter_of_percentage_change(self, conf_dicts, FilteredByHandsTremorFlag=False):
        """
        CREATES SCATTER PLOTS OF THE DATA, INCLUDING A LINEAR REGRESSION.
        Args:
            conf_dicts: List of dictionaries. Each dictionary's settings will generate a different figure. Settings
            include name of the variables for X and Y axis, visits, and sessions to be plotted.
            E.g.: {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [{'off': 'ChangeLimbsBradyRig', 'on': 'ChangeLimbsBradyRig'},
             {'off': 'ResponseRestTrem', 'on': 'ResponseRestTrem'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']}
            FilteredByHandsTremorFlag: Indicate whether the subjects without tremor will be filtered out.
            If the database was already filtered it has no effect.

        Returns: 0. It saves the figures to the results folder.

        """
        style = "plotly"

        FilteredByHandsTremorFlag_noaction = False
        if ~self.trem_filtered_flag and FilteredByHandsTremorFlag:
            FilteredByHandsTremorFlag = FilteredByHandsTremorFlag
            FilteredByHandsTremorFlag_noaction = FilteredByHandsTremorFlag
        else:
            FilteredByHandsTremorFlag = False

        if FilteredByHandsTremorFlag_noaction:
            analysis_name = f"ScatterPlot_PC-{self.percentage_change_function_name}_{self.filtered_string}"
        else:
            analysis_name = f"ScatterPlot_PC-{self.percentage_change_function_name}"

        results_folder = os.path.join(self.results_folder, analysis_name)
        Path(os.path.join(results_folder)).mkdir(parents=True, exist_ok=True)

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'black']
        df = copy.deepcopy(self.data)

        for confs in conf_dicts:
            if len(confs['x']) > 1 and len(confs['x']) != len(confs['y']):
                raise ValueError("Length of X axis element should be 1 or equal to length of Y axis elements.")
            timeName, sesName = self.__get_visit_session_naming(confs['visit'], confs['ses'])
            fig = go.Figure()
            plt.figure(figsize=(10, 6))
            colIdx = 0

            for i_data, dataY in enumerate(confs['y']):
                if len(confs['x']) == len(confs['y']):
                    dataX = confs['x'][i_data]
                else:
                    dataX = confs['x'][0]
                if len(confs['y']) > 1:
                    nameImageExtra = "MultipleMetrics_"
                else:
                    nameImageExtra = ""
                    # fig = go.Figure()
                    # plt.figure(figsize=(10, 6))
                    colIdx = 0

                for visit in confs['visit']:
                    sess = confs['ses']
                    frameOff = 2 * (visit - 1)
                    frameOn = 2 * (visit - 1) + 1
                    if FilteredByHandsTremorFlag:
                        if self.get_database_version() == "ppp":
                            idx = self.get_subjects_above_threshold(["U17ResTremRUE", "U17ResTremLUE"], self.thresh_trem_filter, df[self.sheet_names[frameOff]])
                        elif self.get_database_version() == "drdr":
                            idx = self.get_subjects_above_threshold(["OFFU17RUE", "OFFU17LUE"], self.thresh_trem_filter, df[self.sheet_names[frameOff]])

                        xD = pd.Series([df[self.sheet_names[frameOff]][dataX["off"]][i] for i in idx])
                        y2 = pd.Series([df[self.sheet_names[frameOff]][dataY["off"]][i] for i in idx])
                        y1 = pd.Series([df[self.sheet_names[frameOn]][dataY["off"]][i] for i in idx])

                        if dataX["off"] == "LEDD":
                            xD = pd.Series([df[self.sheet_names[frameOn]][dataX["off"]][i] for i in idx])
                        if dataY["off"] == "LEDD":
                            y2 = pd.Series([df[self.sheet_names[frameOn]][dataY["off"]][i] for i in idx])
                    else:
                        xD = df[self.sheet_names[frameOff]][dataX["off"]]
                        y2 = df[self.sheet_names[frameOff]][dataY["off"]]
                        y1 = df[self.sheet_names[frameOn]][dataY["off"]]

                        if dataX["off"] == "LEDD":
                            xD = df[self.sheet_names[frameOn]][dataX["off"]]
                        if dataX["off"] == "LEDD":
                            y2 = df[self.sheet_names[frameOn]][dataY["off"]]

                    yD = pd.Series(self.percentage_change_function(y1, y2))
                    xD, yD = self.remove_rows_with_nans(xD, yD)

                    df_p = {"X": xD, "Y": yD}

                    # Calculate and plot regression line
                    slope, intercept, r_value, p_value, std_err = stats.linregress(xD, yD)
                    x_range = np.linspace(min(xD), max(xD), 100)
                    y_range = slope * x_range + intercept
                    if self.get_database_version() == "ppp":
                        if len(confs['y']) > 1:
                            label_name = f'visit{visit}_{dataY["off"]}'
                        else:
                            label_name = f'visit{visit}'
                    else:
                        if len(confs['y']) > 1:
                            label_name = f'{dataY["off"]}'
                        else:
                            label_name = f'Not Implemented'

                    if style == "all" or style == "plotly":
                        fig.add_trace(
                            go.Scatter(x=xD, y=yD, mode='markers', name=label_name, marker=dict(color=colors[colIdx])))
                        fig.add_trace(
                            go.Scatter(x=x_range, y=y_range, mode='lines', line=dict(color=colors[colIdx], dash='dash'),
                                       showlegend=False))
                    if style == "all" or style == "sns":
                        sns.scatterplot(data=df_p, x='X', y='Y', color=colors[colIdx], label=label_name)
                        sns.regplot(data=df_p, x='X', y='Y', scatter=False, color=colors[colIdx],
                                    line_kws={'linestyle': '-', 'linewidth': 2},
                                    ci=None)  # Add ci=None for deleting shaded area around regression lines
                        plt.title('Data Distribution')
                        plt.xlabel(dataX['off'])
                        plt.ylabel("% Change")
                        plt.legend(title='Visits and Sessions')
                    colIdx = colIdx + 1

                    if len(confs['y']) == 1:
                        if style == "all" or style == "plotly":
                            fig.update_layout(title='Data Distribution', xaxis_title=dataX['off'],
                                              yaxis_title="% Change")
                            if self.get_database_version() == "ppp":
                                figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{timeName}_{sesName}_plotly.png")
                            else:
                                figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{sesName}_plotly.png")
                            fig.write_image(figure_name)
                            del fig
                        if style == "all" or style == "sns":
                            if self.get_database_version() == "ppp":
                                figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{timeName}_{sesName}_sns.png")
                            else:
                                figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{sesName}_sns.png")
                            plt.savefig(figure_name, bbox_inches='tight', dpi=600)
                            plt.clf()
                            plt.close()

            if len(confs['y']) > 1:
                if style == "all" or style == "plotly":
                    fig.update_layout(title='Data Distribution', xaxis_title=dataX['off'], yaxis_title="% Change")
                    if self.get_database_version() == "ppp":
                        figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}_{timeName}_{sesName}_plotly.png")
                    else:
                        figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}_{sesName}_plotly.png")
                    fig.write_image(figure_name)
                    del fig
                if style == "all" or style == "sns":
                    if self.get_database_version() == "ppp":
                        figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}_{timeName}_{sesName}_sns.png")
                    else:
                        figure_name = os.path.join(results_folder, f"sp_{nameImageExtra}{dataX['off']}_{sesName}_sns.png")
                    plt.savefig(figure_name, bbox_inches='tight', dpi=600)
                    plt.clf()
                    plt.close()
        return 0

    def plot_rainclouds_by_session(self, updrs_conf, FilteredByHandsTremorFlag=False):
        """
        CREATES RAINCLOUD PLOTS FOR THE UPDRS SCORES, VISITS, AND SESSIONS, SPECIFIED IN THE CONFIG DICTIONARY.
        Args:
            updrs_conf: List of dictionaries with the settings for the plots. E.g.: {'updrs':[{'off':'AvgKineticTremor',
            'on': 'AvgKineticTremor'},{'off': 'AvgRestTrem', 'on': 'AvgRestTrem'}],'visit':[1,2,3],'ses':['off','on']}
            FilteredByHandsTremorFlag: Indicate whether the subjects without tremor will be filtered out.
            If the database was already filtered it has no effect.

        Returns: 0. It saves the figures to the results folder.

        """
        FilteredByHandsTremorFlag_noaction = False
        if ~self.trem_filtered_flag and FilteredByHandsTremorFlag:
            FilteredByHandsTremorFlag = FilteredByHandsTremorFlag
            FilteredByHandsTremorFlag_noaction = FilteredByHandsTremorFlag
        else:
            FilteredByHandsTremorFlag = False

        if FilteredByHandsTremorFlag_noaction:
            analysis_name = f"Rainclouds_{self.filtered_string}"
        else:
            analysis_name = f"Rainclouds"

        results_folder = os.path.join(self.results_folder, analysis_name)
        Path(os.path.join(results_folder)).mkdir(parents=True, exist_ok=True)

        database = copy.deepcopy(self.data)

        for confs in updrs_conf:
            for updrs_key in confs["updrs"]:
                df = {}
                key = updrs_key["off"]
                dx = "Visit"
                dhue = "Group"

                df[key] = pd.Series(dtype='float')
                df[dx] = pd.Series(dtype='str')
                df[dhue] = pd.Series(dtype='str')

                for visit in confs["visit"]:
                    for ses in confs["ses"]:
                        sheet = 2 * (visit - 1) if ses == "off" else 2 * (visit - 1) + 1
                        data_series = database[self.sheet_names[sheet]][updrs_key[ses]]

                        if FilteredByHandsTremorFlag:
                            if self.get_database_version() == "ppp":
                                idx = self.get_subjects_above_threshold(["U17ResTremRUE", "U17ResTremLUE"], 1,
                                                                database[self.sheet_names[2 * (visit - 1)]])
                            elif self.get_database_version() == "drdr":
                                idx = self.get_subjects_above_threshold(["OFFU17RUE", "OFFU17LUE"], 1,
                                                                database[self.sheet_names[2 * (visit - 1)]])
                            data_series = pd.Series([data_series[i] for i in idx])

                        if not df[key].empty:
                            df[key] = pd.concat([df[key], data_series], ignore_index=True)
                        else:
                            df[key] = data_series

                        lengthData = len(data_series)

                        visit_label = {
                            1: "Baseline" if self.get_database_version() == "ppp" else "DRDR",
                            2: "Year 1",
                            3: "Year 2"
                        }.get(visit, "Unknown")
                        df[dx] = pd.concat([df[dx], pd.Series([visit_label] * lengthData)], ignore_index=True)

                        ses_label = "OFF" if ses == "off" else "ON"
                        df[dhue] = pd.concat([df[dhue], pd.Series([ses_label] * lengthData)], ignore_index=True)

                cleaned = self.remove_rows_with_nans(df[key], df[dx], df[dhue])
                df[key], df[dx], df[dhue] = cleaned[0], cleaned[1], cleaned[2]

                timeName, sesName = self.__get_visit_session_naming(confs['visit'], confs['ses'])

                f, ax = plt.subplots(figsize=(24, 10))
                ax = self.__RainCloudSNS(x=dx, y=key, hue=dhue, data=df, palette="Set2", bw_method=0.2, linewidth=2,
                                  jitter=0.25, move=0.8, width_viol=.8,
                                  ax=ax, orient="h", alpha=.7, dodge=True, pointplot=True)
                plt.title("MDS - UPDRS: Repeated Measures")
                # if style == "all" or style == "sns":
                if self.get_database_version() == "ppp":
                    figure_name = os.path.join(results_folder, f"rc_{key}_{timeName}_{sesName}_sns.png")
                else:
                    figure_name = os.path.join(results_folder, f"rc_{updrs_key['off']}-{updrs_key['on']}_{timeName}_{sesName}_sns.png")
                plt.savefig(figure_name, bbox_inches='tight', dpi=600)
                plt.clf()
                plt.close()

        return 0

    def create_updrs_arbitrary_clusters(self, responsiveness_key="AvgLimbsRestTrem"):
        """
        CREATES EXCEL FILE WITH THE RESPONSIVENESS PROFILE PER SUBJECT, BASED ON THE IMPROVEMENT BETWEEN OFF-ON MEDICATION CONDITION.
        Args:
            responsiveness_key: Column that will be used as indicator of Tremor. Default to Sum of Tremor by U17a-d and U18 (AvgLimbsRestTrem).
        Returns: 0. Creates the Excel file.

        """
        # Profile: Resistant = 1 and Responsive = 2, No meet rigidity and brady criteria = 3 (excluded), neither responsive nor resistant = 0
        results_folder = self.results_folder
        stats_table = dict()
        visitname = ["Baseline", "Year 1", "Year 2"]
        stats_table[f"LongitudinalSubjectProfile"] = pd.DataFrame()
        final_table = pd.DataFrame()

        if self.trem_filtered_flag:
            analysis_name = self.filtered_string
        else:
            analysis_name = f""

        for visit in [0, 1, 2]:
            # responsivenessRestTremor = pd.Series(database[sheets[2*visit+1]]["ResponseRestTrem"])
            responsivenessRestTremor = pd.Series(self.percentage_change_function(self.data[self.sheet_names[2 * visit + 1]][responsiveness_key], self.data[self.sheet_names[2 * visit]][responsiveness_key]))
            # responsivenessRestTremor = pd.Series(self.percentage_change_function(self.data[self.sheet_names[2 * visit + 1]]["WRestTrem"], self.data[self.sheet_names[2 * visit]]["WRestTrem"]))
            # responsivenessRestTremor = pd.Series(self.percentage_change_function(self.data[self.sheet_names[2 * visit + 1]]["AvgLimbsRestTrem"], self.data[self.sheet_names[2 * visit]]["AvgLimbsRestTrem"]))

            responsivenessRigidity = pd.Series(
                self.percentage_change_function(self.data[self.sheet_names[2 * visit + 1]]["AvgLimbsRigidity4Items"],
                                        self.data[self.sheet_names[2 * visit]]["AvgLimbsRigidity4Items"]))
            responsivenessBrady = pd.Series(self.percentage_change_function(self.data[self.sheet_names[2 * visit + 1]]["AvgBrady5Items"],
                                                                    self.data[self.sheet_names[2 * visit]]["AvgBrady5Items"]))
            responsivenessRigBrady = pd.Series(
                [(responsivenessRigidity[i] + responsivenessBrady[i]) / 2 for i, _ in enumerate(responsivenessBrady)])

            idxRigidity = responsivenessRigidity < 20
            idxBrady = responsivenessBrady < 20
            idxRigBrady = responsivenessRigBrady <= 20
            idxRigBrady = pd.Series(
                [idxRigidity[i] | idxBrady[i] for i in range(len(idxRigidity))])  # Where this is True, Profile = 3

            idxResistant = pd.Series(responsivenessRestTremor <= 20)
            idxResponsive = pd.Series(responsivenessRestTremor >= 50)
            idxNeither = pd.Series((responsivenessRestTremor > 20) & (responsivenessRestTremor < 50))

            stats_table[f"SubjectProfile_{visitname[visit]}"] = pd.DataFrame()
            stats_table[f"Summary_SubjectProfile_{visitname[visit]}"] = pd.DataFrame()

            stats_table[f"SubjectProfile_{visitname[visit]}"]["Subject"] = self.data[self.sheet_names[2 * visit]]["Subject"]
            stats_table[f"SubjectProfile_{visitname[visit]}"]["Responsiveness"] = self.__create_responsiveness_array(
                idxRigBrady, idxResistant, idxResponsive, idxNeither, [0, 1, 2, 3], responsivenessRestTremor,
                rigbrady_filtering=False, display_improvement_trem_percent=False)
            stats_table[f"SubjectProfile_{visitname[visit]}"][
                "ResponsivenessDescription"] = self.__create_responsiveness_array(idxRigBrady, idxResistant,
                idxResponsive, idxNeither, ["Neither", "Resistant", "Responsive", "RigBrady_excluded"],
                responsivenessRestTremor, rigbrady_filtering=False, display_improvement_trem_percent=False)
            stats_table[f"SubjectProfile_{visitname[visit]}"]["ResponsivenessBradyRigExclusion"] = (
                self.__create_responsiveness_array(idxRigBrady, idxResistant, idxResponsive, idxNeither,
                [0, 1, 2, 3], responsivenessRestTremor, rigbrady_filtering=True, display_improvement_trem_percent=False))
            stats_table[f"SubjectProfile_{visitname[visit]}"]["ResponsivenessDescriptionBradyRigExclusion"] = (
                self.__create_responsiveness_array(idxRigBrady, idxResistant, idxResponsive, idxNeither,
                ["Neither", "Resistant", "Responsive", "RigBrady_excluded"], responsivenessRestTremor,
                rigbrady_filtering=True, display_improvement_trem_percent=False))

            stats_table[f"Summary_SubjectProfile_{visitname[visit]}"]["Profile"] = ["Resistant", "Responsive",
                                                                                    "Neither", "RigBrady_Excluded"]
            stats_table[f"Summary_SubjectProfile_{visitname[visit]}"]["# Participants"] = [
                (stats_table[f"SubjectProfile_{visitname[visit]}"]["Responsiveness"] == 1).sum(),
                (stats_table[f"SubjectProfile_{visitname[visit]}"]["Responsiveness"] == 2).sum(),
                (stats_table[f"SubjectProfile_{visitname[visit]}"]["Responsiveness"] == 0).sum(),
                (stats_table[f"SubjectProfile_{visitname[visit]}"]["Responsiveness"] == 3).sum()
            ]

            profile_df = pd.DataFrame()
            profile_df["Subject"] = self.data[self.sheet_names[2 * visit]]["Subject"]
            profile_df[f"Responsiveness_{visitname[visit]}"] = self.__create_responsiveness_array(
                idxRigBrady, idxResistant, idxResponsive, idxNeither, [0, 1, 2, 3], responsivenessRestTremor,
                rigbrady_filtering=False, display_improvement_trem_percent=False)

            profile_df[f"Responsiveness_%_{visitname[visit]}"] = responsivenessRestTremor
            if final_table.empty:
                final_table = profile_df
            else:
                final_table = pd.merge(final_table, profile_df, on="Subject", how="outer")

        final_table, summary_table = self.__include_average_responsiveness_columns(final_table)
        stats_table[f"LongitudinalSubjectProfile"] = final_table
        stats_table["Summary_Subjects_Profile"] = summary_table

        with pd.ExcelWriter(os.path.join(results_folder, f"{analysis_name}Dopamine_responsiveness_profile.xlsx"), engine='openpyxl') as writer:
            for sheet_name, data in stats_table.items():
                data.to_excel(writer, sheet_name=sheet_name, index=False)
        return 0

    def plot_rainclouds_by_group(self, updrs_conf, clustering_method, FilteredByHandsTremorFlag=False, path_to_labels="", model_labels="", model_name=""):
        """
        CREATES RAINCLOUD PLOTS DIVIDED BY TREMOR RESPONSIVENESS BY VISIT, TO EVALUATE PROGRESSION OF THE IMPROVEMENT RATES.
        Args:
            updrs_conf: List of dictionaries with the Settings for the figures. It specifies which UPDRS scores will be
            plotted, the visits that will be included, and the session (OFF or ON, for responsiveness, it does not matter)
            clustering_method: Specifies the method that will be used for dividing the dataset
            FilteredByHandsTremorFlag: Indicate whether the subjects without tremor will be filtered out.
            If the database was already filtered it has no effect.
            path_to_labels: If 'clustering_method' is 'other' then it needs to be provided
            model_labels: String with the name of the column that contains the labels. Default to best models for each clustering method.
            model_name: To be included in the name of the figure, for differentiate between runs.
        Returns: 0. It saves the figures to the results folder.

        """
        (cluster_labels, condition_labels_map) = self.return_labels(method=clustering_method, model=model_labels, path_to_labels=path_to_labels)

        FilteredByHandsTremorFlag_noaction = False
        if ~self.trem_filtered_flag and FilteredByHandsTremorFlag:
            FilteredByHandsTremorFlag = FilteredByHandsTremorFlag
            FilteredByHandsTremorFlag_noaction = FilteredByHandsTremorFlag
        else:
            FilteredByHandsTremorFlag = False

        if FilteredByHandsTremorFlag_noaction:
            analysis_name = f"Rainclouds_Responsiveness_{self.filtered_string}"
        else:
            analysis_name = f"Rainclouds_Responsiveness"

        results_folder = os.path.join(self.results_folder, analysis_name)
        Path(os.path.join(results_folder)).mkdir(parents=True, exist_ok=True)
        
        database = copy.deepcopy(self.data)

        visits = [0, 1, 2]
        visitTime = "Visit"
        condition = "Group"
        subjects = "Subject"
        scores = "UPDRS_scores"
        visit_labels_map = {0: "Baseline", 1: "Year 1", 2: "Year 2"}
        for confs in updrs_conf:
            for updrs_key in confs["updrs"]:
                df = pd.DataFrame()
                scores = updrs_key["off"]
                visit_labels = []
                condition_labels = []
                subject_labels = []
                score_values = []
                unique_labels = cluster_labels.unique()  # [cluster_labels.unique() != 0]

                if condition_labels_map == {}:
                    for clust in unique_labels:
                        condition_labels_map[clust] = f"Cluster{clust}"

                for visit in visits:
                    for ses in unique_labels:
                        if FilteredByHandsTremorFlag:
                            idx = self.get_subjects_above_threshold(["U17ResTremRUE", "U17ResTremLUE"], self.thresh_trem_filter,
                                                            database[self.sheet_names[2 * visit]])
                            databaseFiltered = database[self.sheet_names[2 * visit]].loc[idx].reset_index(drop=True)
                            cluster_labels_array = cluster_labels.loc[idx].reset_index(drop=True)
                        else:
                            cluster_labels_array = cluster_labels
                            databaseFiltered = database[self.sheet_names[2 * visit]]

                        updrs_scores = databaseFiltered[updrs_key["off"]][cluster_labels_array == ses]
                        lengthData = len(updrs_scores)

                        visit_label = visit_labels_map.get(visit, "Unknown")
                        ses_label = condition_labels_map.get(ses, "Unknown")

                        visit_labels.extend([visit_label] * lengthData)
                        condition_labels.extend([ses_label] * lengthData)
                        subject_labels.extend(databaseFiltered["Subject"][cluster_labels_array == ses].tolist())
                        score_values.extend(updrs_scores.tolist())

                df = pd.DataFrame({
                    visitTime: visit_labels,
                    condition: condition_labels,
                    subjects: subject_labels,
                    scores: score_values
                })

                f, ax = plt.subplots(figsize=(24, 10))
                if len(unique_labels) > 1:
                    ax = self.__RainCloudSNS(x=visitTime, y=scores, hue=condition, data=df, palette="Set2", bw_method=0.2,
                                      linewidth=2, jitter=0.25, move=0.8, width_viol=.8,
                                      ax=ax, orient="h", alpha=.7, dodge=True, pointplot=True)
                else:
                    ax = self.__RainCloudSNS(x=visitTime, y=scores, data=df, palette="Set2", bw_method=0.2, linewidth=2,
                                      jitter=0.25, move=0.8, width_viol=.8,
                                      ax=ax, orient="h", alpha=.7, dodge=True, pointplot=True)

                plt.title(f"MDS - UPDRS: Clustering by {clustering_method}", fontsize=24)
                figure_name = os.path.join(results_folder, f"progression_{updrs_key['off']}_{clustering_method}{model_name}.png")
                plt.savefig(figure_name, bbox_inches='tight', dpi=900)
                plt.clf()
                plt.close()

        return 0

    def plot_rainclouds_by_session_and_group(self, updrs_conf, clustering_method, FilteredByHandsTremorFlag=False, path_to_labels="", model_labels="", model_name=""):
        """
        CREATES RAINCLOUD PLOTS DIVIDED BY TREMOR RESPONSIVENESS BY VISIT, TO EVALUATE PROGRESSION OF THE IMPROVEMENT RATES.
        Args:
            updrs_conf: List of dictionaries with the Settings for the figures. It specifies which UPDRS scores will be
            plotted, the visits that will be included, and the session (OFF or ON, for responsiveness, it does not matter)
            clustering_method: Specifies the method that will be used for dividing the dataset
            FilteredByHandsTremorFlag: Indicate whether the subjects without tremor will be filtered out.
            If the database was already filtered it has no effect.
            path_to_labels: If 'clustering_method' is 'other' then it needs to be provided
            model_labels: String with the name of the column that contains the labels. Default to best models for each clustering method.
            model_name: To be included in the name of the figure, for differentiate between runs.
        Returns: 0. It saves the figures to the results folder.

        """
        (cluster_labels, condition_labels_map) = self.return_labels(method=clustering_method, model=model_labels, path_to_labels=path_to_labels)

        FilteredByHandsTremorFlag_noaction = False
        if ~self.trem_filtered_flag and FilteredByHandsTremorFlag:
            FilteredByHandsTremorFlag = FilteredByHandsTremorFlag
            FilteredByHandsTremorFlag_noaction = FilteredByHandsTremorFlag
        else:
            FilteredByHandsTremorFlag = False

        if FilteredByHandsTremorFlag_noaction:
            analysis_name = f"Rainclouds_GroupsSessions_{self.filtered_string}"
        else:
            analysis_name = f"Rainclouds_GroupsSessions"

        results_folder = os.path.join(self.results_folder, analysis_name)
        Path(os.path.join(results_folder)).mkdir(parents=True, exist_ok=True)

        database = copy.deepcopy(self.data)

        visits = [0, 1, 2]
        visitTime = "Visit"
        condition = "Group"
        subjects = "Subject"
        scores = "UPDRS_scores"
        visit_labels_map = {0: "Baseline", 1: "Year 1", 2: "Year 2"}
        for confs in updrs_conf:
            num_subplots = len(confs['subplots'])
            for updrs_key in confs["updrs"]:
                df_final = pd.DataFrame()
                scores = updrs_key["off"]
                visit_labels_final = []
                condition_labels_final = []
                subject_labels_final = []
                score_values_final = []
                # unique_labels = cluster_labels.unique()  # [cluster_labels.unique() != 0]

                if condition_labels_map == {}:
                    for clust in unique_labels:
                        condition_labels_map[clust] = f"Cluster{clust}"

                # Create plot
                fig, axs = plt.subplots(nrows=1, ncols=num_subplots, figsize=(num_subplots*12, num_subplots*5))
                if num_subplots == 1:
                    axs = [axs]

                for idx_subplot, subplot_conf in enumerate(confs['subplots']):
                    visit_labels = []
                    condition_labels = []
                    subject_labels = []
                    score_values = []
                    unique_labels = self.__get_unique_labels(condition_labels_map, subplot_conf['groups'])

                    visits_per_conf = self._get_sheet_names_per_visit(subplot_conf['visits'], subplot_conf['sessions'])
                    for visit in visits_per_conf:
                        for label in unique_labels:
                            if FilteredByHandsTremorFlag:
                                idx = self.get_subjects_above_threshold(["U17ResTremRUE", "U17ResTremLUE"],
                                                                        self.thresh_trem_filter,
                                                                        database[visit])
                                databaseFiltered = database[visit].loc[idx].reset_index(drop=True)
                                cluster_labels_array = cluster_labels.loc[idx].reset_index(drop=True)
                            else:
                                cluster_labels_array = cluster_labels
                                databaseFiltered = database[visit]

                            visit_idx = 0 if "1" in visit else 1 if "2" in visit else 2
                            visit_label = visit_labels_map.get(visit_idx, "Unknown")
                            ses_label = condition_labels_map.get(label, "Unknown")

                            if "OFF" in visit:
                                ses_label = f"{ses_label}OFF"
                                key_ses_id = updrs_key["off"]
                            else:
                                ses_label = f"{ses_label}ON"
                                key_ses_id = updrs_key["on"]

                            updrs_scores = databaseFiltered[key_ses_id][cluster_labels_array == label]
                            lengthData = len(updrs_scores)

                            visit_labels.extend([visit_label] * lengthData)
                            condition_labels.extend([ses_label] * lengthData)
                            subject_labels.extend(databaseFiltered["Subject"][cluster_labels_array == label].tolist())
                            score_values.extend(updrs_scores.tolist())

                    df = pd.DataFrame({
                        visitTime: visit_labels,
                        condition: condition_labels,
                        subjects: subject_labels,
                        scores: score_values
                    })

                    if (len(unique_labels) > 1) or (len(df[condition].unique()) > 1):
                        self.__RainCloudSNS(x=visitTime, y=scores, hue=condition, data=df, palette="Set2",
                                            bw_method=0.2, linewidth=2, jitter=0.25, move=0.8, width_viol=.8,
                                            ax=axs[idx_subplot], orient="h", alpha=.7, dodge=True, pointplot=True)
                    else:
                        self.__RainCloudSNS(x=visitTime, y=scores, data=df, palette="Set2",
                                            bw_method=0.2, linewidth=2, jitter=0.25, move=0.8, width_viol=.8,
                                            ax=axs[idx_subplot], orient="h", alpha=.7, dodge=True, pointplot=True)

                    # Optionally, set the title for each subplot
                    axs[idx_subplot].set_title(f"{''.join(subplot_conf['groups'])}", fontsize=20)

                    visit_labels_final.extend(visit_labels)
                    condition_labels_final.extend(condition_labels)
                    subject_labels_final.extend(subject_labels)
                    score_values_final.extend(score_values)

                df_final = pd.DataFrame({
                    visitTime: visit_labels_final,
                    condition: condition_labels_final,
                    subjects: subject_labels_final,
                    scores: score_values_final
                })

                plt.suptitle(f"MDS - UPDRS: Clustering by {clustering_method}", fontsize=24)
                plt.tight_layout()
                figure_name = os.path.join(results_folder, f"progression_{updrs_key['off']}_{clustering_method}{model_name}.png")
                plt.savefig(figure_name, bbox_inches='tight', dpi=900)
                plt.clf()
                plt.close()

        return 0

    def plot_responsiveness_bars(self, updrs_conf, clustering_method, FilteredByHandsTremorFlag=False, path_to_labels="", model_labels="", model_name=""):
        (cluster_labels, condition_labels_map) = self.return_labels(method=clustering_method, model=model_labels, path_to_labels=path_to_labels)

        FilteredByHandsTremorFlag_noaction = False
        if ~self.trem_filtered_flag and FilteredByHandsTremorFlag:
            FilteredByHandsTremorFlag = FilteredByHandsTremorFlag
            FilteredByHandsTremorFlag_noaction = FilteredByHandsTremorFlag
        else:
            FilteredByHandsTremorFlag = False

        if FilteredByHandsTremorFlag_noaction:
            analysis_name = f"Bars_Responsiveness_{self.filtered_string}"
        else:
            analysis_name = f"Bars_Responsiveness"

        results_folder = os.path.join(self.results_folder, analysis_name)
        Path(os.path.join(results_folder)).mkdir(parents=True, exist_ok=True)

        database = copy.deepcopy(self.data)

        visitTime = "Visit"
        condition = "Group"
        subjects = "Subject"
        sessions = "Session"
        visit_labels_map = {0: "Baseline", 1: "Year 1", 2: "Year 2"}
        for confs in updrs_conf:
            num_subplots = len(confs['subplots'])
            for updrs_key in confs['updrs']:
                scores = updrs_key["off"]
                calculate_ylimits = True

                # Create plot
                fig, axs = plt.subplots(nrows=1, ncols=num_subplots, figsize=(num_subplots * 12, num_subplots*5))
                if num_subplots == 1:
                    axs = [axs]

                for idx_subplot, subplot_conf in enumerate(confs['subplots']):
                    visit_labels = []
                    condition_labels = []
                    subject_labels = []
                    score_values = []
                    session_labels = []
                    unique_labels = self.__get_unique_labels(condition_labels_map, subplot_conf['groups'])

                    visits_per_conf = self._get_sheet_names_per_visit(subplot_conf['visits'], subplot_conf['sessions'])
                    for visit in visits_per_conf:
                        for label in unique_labels:
                            if FilteredByHandsTremorFlag:
                                idx = self.get_subjects_above_threshold(["U17ResTremRUE", "U17ResTremLUE"], self.thresh_trem_filter,database[visit])
                                databaseFiltered = database[visit].loc[idx].reset_index(drop=True)
                                cluster_labels_array = cluster_labels.loc[idx].reset_index(drop=True)
                            else:
                                cluster_labels_array = cluster_labels
                                databaseFiltered = database[visit]

                            visit_idx = 0 if "1" in visit else 1 if "2" in visit else 2
                            visit_label = visit_labels_map.get(visit_idx, "Unknown")
                            cond_label = condition_labels_map.get(label, "Unknown")

                            if "OFF" in visit:
                                ses_label = f"OFF"
                                key_ses_id = updrs_key["off"]
                            else:
                                ses_label = f"ON"
                                key_ses_id = updrs_key["on"]

                            updrs_scores = databaseFiltered[key_ses_id][cluster_labels_array == label]
                            lengthData = len(updrs_scores)

                            visit_labels.extend([visit_label] * lengthData)
                            session_labels.extend([ses_label] * lengthData)
                            condition_labels.extend([cond_label] * lengthData)
                            subject_labels.extend(databaseFiltered["Subject"][cluster_labels_array == label].tolist())
                            score_values.extend(updrs_scores.tolist())

                    df = pd.DataFrame({
                        visitTime: visit_labels,
                        condition: condition_labels,
                        sessions: session_labels,
                        subjects: subject_labels,
                        scores: score_values
                    })

                    if calculate_ylimits:
                        calculate_ylimits = False
                        y_limits = [np.min(score_values) - 5, np.max(score_values) + 5]

                    if confs['subplot_divider'] == 'groups':
                        hue_val = sessions
                        bars_divider = visitTime
                    else:
                        hue_val = visitTime
                        bars_divider = sessions

                    # if (len(unique_labels) > 1) or (len(df[condition].unique()) > 1):
                    if subplot_conf['merge_sessions']:
                        df = df.groupby([subjects, visitTime], as_index=False)[scores].mean()
                        self._BarsPlotSNS(x=bars_divider, y=scores, data=df, palette="Set2",
                                            bw_method=0.2, linewidth=2, jitter=0.25, move=0.5, width_viol=.6,
                                            ax=axs[idx_subplot], orient="v", alpha=.7, dodge=True, pointplot=True, y_limits=y_limits)
                    else:
                        self._BarsPlotSNS(x=bars_divider, y=scores, hue=hue_val, data=df, palette="Set2",
                                            bw_method=0.2, linewidth=2, jitter=0.25, move=0.6, width_viol=.6,
                                            ax=axs[idx_subplot], orient="v", alpha=.7, dodge=True, pointplot=True, y_limits=y_limits)

                    # Optionally, set the title for each subplot
                    axs[idx_subplot].set_title(f"{''.join(subplot_conf['groups'])}", fontsize=24)

                plt.suptitle(f"MDS - UPDRS Responsiveness per Group: Clustering by {clustering_method}", fontsize=28)
                plt.tight_layout()
                figure_name = os.path.join(results_folder, f"barsplot_{updrs_key['off']}_{clustering_method}{model_name}.png")
                plt.savefig(figure_name, bbox_inches='tight', dpi=900)
                plt.clf()
                plt.close()
                
        return 0

    def averaged_histograms(self, updrs_conf, clustering_method, FilteredByHandsTremorFlag=False, path_to_labels="", model_labels="", model_name=""):
        FilteredByHandsTremorFlag_noaction = False
        if ~self.trem_filtered_flag and FilteredByHandsTremorFlag:
            FilteredByHandsTremorFlag = FilteredByHandsTremorFlag
            FilteredByHandsTremorFlag_noaction = FilteredByHandsTremorFlag
        else:
            FilteredByHandsTremorFlag = False

        if FilteredByHandsTremorFlag_noaction:
            analysis_name = f"Averaged_Histograms_{self.filtered_string}"
        else:
            analysis_name = f"Averaged_Histograms"

        results_folder = os.path.join(self.results_folder, analysis_name)
        Path(os.path.join(results_folder)).mkdir(parents=True, exist_ok=True)

        if "subplots" in updrs_conf.keys():
            num_subplots = len(updrs_conf['subplots'])
        else:
            num_subplots = 1

        # Create plot
        # fig, axs = plt.subplots(nrows=1, ncols=num_subplots, figsize=(num_subplots * 12, num_subplots * 5))
        # if num_subplots == 1:
        #     axs = [axs]

        # Retrieve formatted data frames
        dataframes = self.obtain_formatted_dataframe(updrs_conf, clustering_method, FilteredByHandsTremorFlag=FilteredByHandsTremorFlag, path_to_labels="", model_labels="")

        visitTime = "Visit"
        condition = "Group"
        subjects = "Subject"
        sessions = "Session"
        subplots_confs = updrs_conf['subplots']
        num_keys = len(updrs_conf['updrs'])
        for idx_subplot, df in enumerate(dataframes):
            scores = updrs_conf['updrs'][int(np.floor(idx_subplot/num_subplots))]['off']
            # df_change = df.groupby([subjects, condition, visitTime]).apply(self._apply_percentage_change, score_column=scores)
            df_change = df.groupby([subjects, condition, sessions], as_index=False)[scores].mean()

            changes_data = {"temp": df_change[scores]}

            self.__histogram_backend(changes_data, scores, nbins=15, results_folder=results_folder, style="plotly",
                                     typeC="averaged", includeGaussianCurve=True)

        return 0

    def obtain_formatted_dataframe(self, confs, clustering_method, FilteredByHandsTremorFlag=False, path_to_labels="", model_labels=""):
        """
        RETURN A FORMATTED DATAFRAME INCLUDING.
        Args:
            confs:
            clustering_method:
            FilteredByHandsTremorFlag:
            path_to_labels:
            model_labels:

        Returns:

        """
        (cluster_labels, condition_labels_map) = self.return_labels(method=clustering_method, model=model_labels,
                                                                    path_to_labels=path_to_labels)

        database = copy.deepcopy(self.data)
        dataframes_final = list()

        visitTime = "Visit"
        condition = "Group"
        subjects = "Subject"
        sessions = "Session"
        visit_labels_map = {0: "Baseline", 1: "Year 1", 2: "Year 2"}

        for updrs_key in confs['updrs']:
            scores = updrs_key["off"]

            for idx_subplot, subplot_conf in enumerate(confs['subplots']):
                visit_labels = []
                condition_labels = []
                subject_labels = []
                score_values = []
                session_labels = []
                unique_labels = self.__get_unique_labels(condition_labels_map, subplot_conf['groups'])

                visits_per_conf = self._get_sheet_names_per_visit(subplot_conf['visits'], subplot_conf['sessions'])
                for visit in visits_per_conf:
                    for label in unique_labels:
                        if FilteredByHandsTremorFlag:
                            idx = self.get_subjects_above_threshold(["U17ResTremRUE", "U17ResTremLUE"],
                                                                    self.thresh_trem_filter, database[visit])
                            databaseFiltered = database[visit].loc[idx].reset_index(drop=True)
                            cluster_labels_array = cluster_labels.loc[idx].reset_index(drop=True)
                        else:
                            cluster_labels_array = cluster_labels
                            databaseFiltered = database[visit]

                        visit_idx = 0 if "1" in visit else 1 if "2" in visit else 2
                        visit_label = visit_labels_map.get(visit_idx, "Unknown")
                        cond_label = condition_labels_map.get(label, "Unknown")

                        if "OFF" in visit:
                            ses_label = f"OFF"
                            key_ses_id = updrs_key["off"]
                        else:
                            ses_label = f"ON"
                            key_ses_id = updrs_key["on"]

                        updrs_scores = databaseFiltered[key_ses_id][cluster_labels_array == label]
                        lengthData = len(updrs_scores)

                        visit_labels.extend([visit_label] * lengthData)
                        condition_labels.extend([cond_label] * lengthData)
                        session_labels.extend([ses_label] * lengthData)
                        subject_labels.extend(databaseFiltered["Subject"][cluster_labels_array == label].tolist())
                        score_values.extend(updrs_scores.tolist())

                df = pd.DataFrame({
                    visitTime: visit_labels,
                    condition: condition_labels,
                    sessions: session_labels,
                    subjects: subject_labels,
                    scores: score_values
                })

                dataframes_final.append(df)

        return dataframes_final

    def _get_sheet_names_per_visit(self, visits, sessions):
        list_sheets = set()
        for visit in visits:
            if visit == "Baseline":
                visit = 'Visit 1'
                if self.get_database_version() == 'drdr':
                    visit = 'UPDRS'
            elif visit == 'Year 1':
                visit = 'Visit 2'
            else:
                visit = 'Visit 3'
            for sheet_name in self.sheet_names:
                words_in_sheet_name = sheet_name.upper().split()
                for session in sessions:
                    if (visit.upper() in sheet_name.upper()) and (session.upper() in words_in_sheet_name):
                        list_sheets.add(sheet_name)

        return list(list_sheets)

    def _apply_percentage_change(self, group, score_column):
        try:
            # Separate ON and OFF scores based on the provided score column
            ratingOn = group.loc[group['Session'] == 'ON', score_column].values
            ratingOff = group.loc[group['Session'] == 'OFF', score_column].values

            # Apply the percentage change function
            change = self.percentage_change_basic(ratingOn, ratingOff)

            # Add the change as a new column
            group['PercentChange'] = change[0] if len(change) == 1 else change
        except Exception as e:
            group['PercentChange'] = None  # In case of an error (e.g., missing ON/OFF session)

        return group

    @staticmethod
    def __get_unique_labels(condition_labels_map, groups):
        unique_labels = set()

        for group in groups:
            for key, value in condition_labels_map.items():
                if (group == value) or (group in value):
                    unique_labels.add(key)

        return list(unique_labels)

    @staticmethod
    def __include_average_responsiveness_columns(data):
        final_table = copy.deepcopy(data)
        final_table["Average_Responsiveness"] = final_table[
            ['Responsiveness_%_Baseline', 'Responsiveness_%_Year 1', 'Responsiveness_%_Year 2']].mean(axis=1)

        conditions = [
            (final_table['Average_Responsiveness'] >= 50),
            (final_table['Average_Responsiveness'] <= 20)
        ]
        choices = [2, 1]
        final_table['Final_Responsiveness'] = np.select(conditions, choices, default=0)

        summary_table = pd.DataFrame()
        summary_table["Profile"] = [
            "Resistant",
            "Responsive",
            "Neither",
            "RigBrady_Excluded"
        ]
        summary_table["Participants"] = [
            sum(final_table['Final_Responsiveness'] == 1),
            sum(final_table['Final_Responsiveness'] == 2),
            sum(final_table['Final_Responsiveness'] == 0),
            sum(final_table['Final_Responsiveness'] == 3)
        ]
        return final_table, summary_table

    @staticmethod
    def __create_responsiveness_array(idxRigBrady, idxResistant, idxResponsive, idxNeither, values, restTremResp,
                                            rigbrady_filtering=True, display_improvement_trem_percent=True):
        array = []
        if display_improvement_trem_percent:
            for i, resi in enumerate(idxResistant):
                if resi:
                    array.append(f"{values[1]} : {restTremResp[i]}")
                elif idxResponsive[i]:
                    array.append(f"{values[2]} : {restTremResp[i]}")
                elif idxNeither[i]:
                    array.append(f"{values[0]} : {restTremResp[i]}")
        else:
            for i, resi in enumerate(idxResistant):
                if resi:
                    array.append(int(values[1]) if values[1] in [0, 1, 2, 3] else values[1])
                elif idxResponsive[i]:
                    array.append(int(values[2]) if values[2] in [0, 1, 2, 3] else values[2])
                elif idxNeither[i]:
                    array.append(int(values[0]) if values[0] in [0, 1, 2, 3] else values[0])

        array = pd.Series(array)
        if rigbrady_filtering:
            array[idxRigBrady] = int(values[3]) if values[3] in [0, 1, 2, 3] else values[3]

        return array

    @staticmethod
    def __RainCloudSNS(x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient="v", width_viol=.7,
                     width_box=.15, palette="Set2", bw=.2, linewidth=1, cut=0., scale="area", jitter=0.5, move=0.,
                     offset=None, point_size=3, ax=None, pointplot=False, alpha=None, dodge=False, linecolor='red',
                     **kwargs):

        if orient == 'h':  # swap x and y
            x, y = y, x

        if ax is None:
            ax = plt.gca()

        if offset is None:
            offset = max(width_box / 1.8, .15) + .05

        # Define the properties for different plot elements
        kwcloud = {k.replace("cloud_", ""): v for k, v in kwargs.items() if k.startswith("cloud_")}
        kwbox = {k.replace("box_", ""): v for k, v in kwargs.items() if k.startswith("box_")}
        kwrain = {k.replace("rain_", ""): v for k, v in kwargs.items() if k.startswith("rain_")}
        kwpoint = {k.replace("point_", ""): v for k, v in kwargs.items() if k.startswith("point_")}

        # Draw the half-violin (cloud) plot
        sns.violinplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order,
                       orient=orient if orient == 'v' else 'h', width=width_viol, inner=None,
                       palette=palette, bw_method=bw, linewidth=linewidth, cut=cut, density_norm=scale,
                       split=(hue is not None), ax=ax, **kwcloud)

        # Draw the boxplot (umbrella)
        sns.boxplot(x=x, y=y, hue=hue, data=data, orient=orient, width=width_box,
                    order=order, hue_order=hue_order, palette=palette, dodge=dodge,
                    ax=ax, **kwbox)

        # Draw the stripplot (rain)
        sns.stripplot(x=x, y=y, hue=hue, data=data, orient=orient,
                      order=order, hue_order=hue_order, palette=palette, jitter=jitter,
                      dodge=dodge, size=point_size, ax=ax, **kwrain)

        # Add pointplot (if needed)
        if pointplot:
            if hue is not None:
                sns.pointplot(x=x, y=y, hue=hue, data=data, orient=orient,
                              order=order, hue_order=hue_order, dodge=width_box / 2.,
                              palette=palette if hue is not None else linecolor, ax=ax, **kwpoint)
            else:
                sns.pointplot(x=x, y=y, data=data, orient=orient,
                              order=order, dodge=False,
                              color='red', ax=ax, **kwpoint)

        # Adjust alpha transparency
        if alpha is not None:
            for collection in ax.collections + ax.artists:
                collection.set_alpha(alpha)

        # Prune legend and adjust plot limits
        if hue is not None:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:len(labels) // (4 if pointplot else 3)], labels[:len(labels) // (4 if pointplot else 3)],
                      bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=str(hue))
        else:
            ax.legend().remove()

        if orient == "h":
            ylim = list(ax.get_ylim())
            ylim[-1] -= (width_box + width_viol) / 4.
            ax.set_ylim(ylim)
        elif orient == "v":
            xlim = list(ax.get_xlim())
            xlim[-1] -= (width_box + width_viol) / 4.
            ax.set_xlim(xlim)

        return ax

    @staticmethod
    def _BarsPlotSNS(x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient="v", width_viol=.7,
                     width_box=.11, palette="Set2", bw=.2, linewidth=1, cut=0., scale="area", jitter=0.5, move=0.,
                     offset=None, point_size=3, ax=None, pointplot=False, alpha=None, dodge=False, linecolor='red',
                     y_limits=[-25, 100], **kwargs):

        if orient == 'h':  # swap x and y
            x, y = y, x

        if ax is None:
            ax = plt.gca()

        if offset is None:
            offset = max(width_box / 1.2, .10) + .05

        # Define the properties for different plot elements
        kwcloud = {k.replace("cloud_", ""): v for k, v in kwargs.items() if k.startswith("cloud_")}
        kwrain = {k.replace("rain_", ""): v for k, v in kwargs.items() if k.startswith("rain_")}

        # Plot the barplot
        sns.barplot(x=x, y=y, hue=hue, data=data, errorbar="sd", capsize=.2, palette=palette,
                    edgecolor="black", err_kws={'linewidth': 1.5}, dodge=True if hue is not None else False, ax=ax)

        # Move the x/y based on orientation
        new_x = move + offset if orient == "v" else 0
        new_y = move + offset if orient == "h" else 0

        # Plot the violin plot
        sns.violinplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order, orient=orient, width=width_viol,
                       inner=None, palette=palette, bw_method=bw, linewidth=linewidth,
                       dodge=True, cut=cut, density_norm='area', split=(hue is not None), alpha=0.4, ax=ax, **kwcloud)

        # Draw the stripplot (points, slightly offset)
        sns.stripplot(x=x, y=y, hue=hue, data=data, orient=orient, order=order, hue_order=hue_order,
                      palette=palette, edgecolor="black", linewidth=1.2, jitter=jitter,
                      dodge=True if hue is not None else False, size=point_size, ax=ax, **kwrain)

        # Combine handles from all plots for the legend
        if hue is not None:
            handles_bar, labels_bar = ax.get_legend_handles_labels()  # Get handles from the barplot
            handles_violin, labels_violin = ax.get_legend_handles_labels()  # Get handles from the violin plot

            # Use the bar handles and the violin labels for the combined legend
            handles = handles_bar[:2]  # Use only the first set of handles for bars
            labels = labels_bar[:2]

            ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=str(hue))
        else:
            ax.legend().remove()

        # Adjust the plot limits to ensure visibility of the elements
        if orient == "h":
            ylim = list(ax.get_ylim())
            ylim[-1] -= (width_box + width_viol) / 9
            ax.set_ylim(ylim)
        elif orient == "v":
            xlim = list(ax.get_xlim())
            xlim[-1] -= (width_box + width_viol) / 9
            ax.set_xlim(xlim)
            ax.set_ylim(y_limits)

        return ax

    @staticmethod
    def __get_visit_session_naming(visits, sessions):
        timeNamesPos = ["basel", "year1", "year2"]
        if len(visits) == 1:
            timeName = timeNamesPos[visits[0] - 1]
        elif len(visits) == 3:
            timeName = "long"
        else:
            timeName = f"{timeNamesPos[visits[0] - 1]}-{timeNamesPos[visits[1] - 1]}"
        if len(sessions) == 1:
            sesName = sessions[0]
        else:
            sesName = "OffOn"
        return timeName, sesName

    @staticmethod
    def __histogram_backend(changes_data, updrs_key, nbins, results_folder, visit=0, style="all",
                                 typeC="offon", ses="off", includeGaussianCurve=True):
        if includeGaussianCurve:
            stat, p_value = stats.shapiro(changes_data['temp'])
            # Fit Gaussian distribution
            mu, std = np.mean(changes_data['temp']), np.std(changes_data['temp'])
            x = np.linspace(min(changes_data['temp']), max(changes_data['temp']), 100)
            p = stats.norm.pdf(x, mu, std)
            hist_values, bin_edges = np.histogram(changes_data['temp'], bins=nbins, density=True)
            p = p / max(p) * max(
                [(bin_edges[i + 1] - bin_edges[i]) * hist_values[i] for i, _ in enumerate(hist_values)]
            ) * 100

        if style == "all" or style == "plotly":
            # fig = px.histogram(changes_data, x='temp', histnorm='percent', title='Clinical Tremor Severity', nbins=10)
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=changes_data["temp"],
                histnorm='percent',
                name='Histogram',
                xbins=dict(size=bin_edges[1] - bin_edges[0]),
                showlegend=False
            ))
            if includeGaussianCurve:
                fig.add_trace(go.Scatter(x=x, y=p, mode='lines', name='Gaussian Fit', showlegend=False))
                # Add Shapiro-Wilk p-value text
                fig.add_annotation(
                    x=0.95, y=0.95,
                    text=f"Shapiro p- {p_value:.4f}",
                    showarrow=False,
                    xref='paper', yref='paper',
                    xanchor='right', yanchor='top',
                    font=dict(size=14, color="black")
                )
            x_label = f'{updrs_key} (MDS-UPDRS) score' if typeC == "singleVisit" else f'{updrs_key} (MDS-UPDRS) (% Change)'
            fig.update_layout(title='Clinical Tremor Severity', xaxis_title=x_label, yaxis_title='Density (%)')
            if typeC == "offon":
                figure_name = os.path.join(results_folder, f"pc_{updrs_key}_visit{visit + 1}_plotly.png")
            elif typeC == "singleVisit":
                figure_name = os.path.join(results_folder, f"hist_{updrs_key}_visit{visit + 1}_{ses}_plotly.png")
            elif typeC == "averaged":
                figure_name = os.path.join(results_folder, f"pc_{updrs_key}_averaged.png")
            else:
                figure_name = os.path.join(results_folder, f"pc_{updrs_key}_{ses}_visits{visit[0] + 1}-{visit[1] + 1}_plotly.png")
            fig.write_image(figure_name)
            del fig

        if style == "all" or style == "sns":
            sns.histplot(changes_data['temp'], kde=False, stat='percent', bins=nbins)

            if includeGaussianCurve:
                plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
                plt.text(0.95, 0.95, f"Shapiro p- {p_value:.4f}",
                         horizontalalignment='right', verticalalignment='top',
                         transform=plt.gca().transAxes,
                         fontsize=14, color='black')
            plt.title('Clinical Tremor Severity')
            plt.xlabel(x_label)
            plt.ylabel('Density (%)')
            # plt.legend()
            if typeC == "offon":
                figure_name = os.path.join(results_folder, f"pc_{updrs_key}_visit{visit + 1}_sns.png")
            elif typeC == "singleVisit":
                figure_name = os.path.join(results_folder, f"hist_{updrs_key}_visit{visit + 1}_{ses}_sns.png")
            elif typeC == "averaged":
                figure_name = os.path.join(results_folder, f"pc_{updrs_key}_averaged.png")
            else:
                figure_name = os.path.join(results_folder, f"pc_{updrs_key}_{ses}_visits{visit[0] + 1}-{visit[1] + 1}_sns.png")
            plt.savefig(figure_name, bbox_inches='tight', dpi=900)
            plt.clf()
            plt.close()

            return 0