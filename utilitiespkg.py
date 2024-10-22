import os
import sys
import copy
from frozendict import frozendict
import pandas as pd

class SearchDirectories:
    def __init__(self):
        print(" --- SEARCH DIRECTORIES CLASS OBJECT --- ")

    @staticmethod
    def search_name_pattern(directory, pattern):
        matching_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if pattern in file:
                    matching_files.append(os.path.join(root, file))
        return matching_files

    @staticmethod
    def search_name_patterns(directory, patterns):
        matching_files = []

        if isinstance(patterns, str):
            patterns = [patterns]

        for file in os.listdir(directory):
            full_path = os.path.join(directory, file)

            if os.path.isfile(full_path):
                if all(pattern in file for pattern in patterns):
                    matching_files.append(full_path)

        return matching_files

    @staticmethod
    def extract_participants_list(file_path):
        df = pd.read_excel(file_path)
        participants = df["Subject"]

        return participants

class DatabaseStructure:
    def __init__(self, data, database_v, trem_filtered = True):
        # print(" --- INITIALIZING ROOT CLASS OBJECT --- ")
        self.data = data
        self.database = database_v
        self.sheet_names = list(self.data.keys())


        copy_data = copy.deepcopy(self.data)
        self._raw_data = self.__freeze(copy_data)

        if self.database == "ppp":
            self._folder_root = "PPP-POM_cohort"
        elif self.database == "drdr":
            self._folder_root = "fMRI_DRDR"
        else:
            raise ValueError("Database version is not valid. Select:\n"
                             "ppp - Personalized Parkinson Project\n"
                             "drdr - Dopamine Resistant vs Dopamine Responsive\n")

        self.results_folder = f"/project/3024023.01/{self._folder_root}/updrs_analysis/"

        if trem_filtered:
            self.filtered_string = "TremFiltered"
            self.trem_filtered_flag = True
        else:
            self.filtered_string = ""
            self.trem_filtered_flag = False

        self.thresh_trem_filter = 1
        self.name_pattern_responsiveness_file = "Dopamine_responsiveness_profile";
        self.path_to_arbitrary_responsiveness_profile = os.path.join(self.results_folder, f"{self.filtered_string}Dopamine_responsiveness_profile.xlsx")
        
    def get_database_version(self):
        return self.database

    def __freeze(self, data):
        """Recursively converts mutable types to their immutable counterparts."""
        if isinstance(data, dict):
            return frozendict({key: self.__freeze(value) for key, value in data.items()})
        elif isinstance(data, list):
            return tuple(self.__freeze(item) for item in data)
        elif isinstance(data, set):
            return frozenset(self.__freeze(item) for item in data)
        else:
            return data

    def reset_data_to_orig(self):
        """
        Resets the working data to the version of the dataset used when creating the class object.
        Returns: 0. It updates the working data on the backend.

        """

        self.data = copy.deepcopy(self._raw_data)
        return 0

class DataHandling(DatabaseStructure):
    """
    CLASS THAT PROVIDES TOOLS FOR DATA MANAGEMENT (CLEANING, FILTERING, AND SIMPLE COMPUTATIONS).
    """
    def __init__(self, data, database_v, trem_filtered=True):
        # print(" --- INITIALIZING DATA HANDLING CLASS OBJECT --- ")
        super().__init__(data, database_v, trem_filtered=trem_filtered)

    def remove_unmedicated_subjects(self):
        """
        LOOKS INTO THE DATABASE AND REMOVES THE ROWS WHERE THE MEDICATION INFORMATION WAS MISSING IN AT LEAST 1 VISIT.
        IT UPDATES THE DATABASE IN THE STRUCTURE.
        Returns: The cleaned database.

        """

        sheets = list(self.data.keys())
        if self.get_database_version() == "ppp":
            sheets = [sheets[1], sheets[3], sheets[5]]

        valid_subjects = set(self.data[sheets[0]].loc[self.data[sheets[0]]["LEDD"] > 0, "Subject"])
        for sheet in sheets[1:]:
            sheet_subjects = set(self.data[sheet].loc[self.data[sheet]["LEDD"] > 0, "Subject"])
            valid_subjects = valid_subjects.intersection(sheet_subjects)

        for sheet in sheets:
            self.data[sheet] = self.data[sheet][self.data[sheet]["Subject"].isin(valid_subjects)].reset_index(
                drop=True)

        return copy.deepcopy(self.data)

    @staticmethod
    def remove_rows_with_nans(*args):
        data_series = [copy.deepcopy(series) for series in args]
        # Create a boolean mask for non-NaN values
        mask = pd.Series([True] * len(data_series[0]))
        for series in data_series:
            mask &= ~series.isnull()
        # Filter each series based on the mask
        cleaned_series = [series[mask] for series in data_series]
        return cleaned_series

    def get_subjects_above_threshold(self, eval_keys=["U17ResTremRUE", "U17ResTremLUE"], threshold=1, *dfs):
        """
        GET SUBJECTS THAT SURPASS THE THRESHOLD IN THE eval_keys LIST.
        Args:
            eval_keys: List of 2 UPDRS keys used to evaluate the threshold. Typically, the Arms tremor scores.
            threshold: Threshold that will be evaluated. Default to 1.
            *dfs: Dataframes that will be checked.

        Returns: A tuple with the index of the subjects above threshold, and a Serie with the Subject IDs.

        """
        dataframes = [copy.deepcopy(df) for df in dfs]
        indices = []
        for df in dataframes:
            idxRUE = df[eval_keys[0]] >= threshold
            idxLUE = df[eval_keys[1]] >= threshold
            indices.append((idxRUE | idxLUE))

        combined_indices = indices[0]
        for idx in indices[1:]:
            combined_indices = combined_indices & idx

        result_indexes = combined_indices[combined_indices].index.tolist()
        
        tremor_subjects = self.data[self.sheet_names[0]]['Subject'].loc[result_indexes].reset_index(drop=True)
        # return (result_indexes, tremor_subjects)
        return result_indexes

    @staticmethod
    def percentage_change_elble(ratingOn, ratingOff, alpha=0.5):
        if len(ratingOn) != len(ratingOff):
            raise ValueError("Rating ON and Rating OFF have different number of elements.")
        if len(ratingOn) > 1:
            change = [100 * (pow(10, alpha * (ratingOn[i] - ratingOff[i])) - 1) for i, _ in enumerate(ratingOn)]
        else:
            change = 100 * (pow(10, alpha * (ratingOn - ratingOff)) - 1)
        return [-1 * x for x in change]  # [-1 * x for x in change] # change

    @staticmethod
    def percentage_change_basic(ratingOn, ratingOff, alpha=0.5):
        if len(ratingOn) != len(ratingOff):
            raise ValueError("Rating ON and Rating OFF have different number of elements.")
        if len(ratingOn) > 1:
            change = [
                100 * ((ratingOff[i] - ratingOn[i]) / ratingOff[i]) if ratingOff[i] != 0 else 0
                for i in range(len(ratingOn))
            ]
        else:
            change = 100 * ((ratingOff - ratingOn) / ratingOff)
        return change  # [-1 * x for x in change]
