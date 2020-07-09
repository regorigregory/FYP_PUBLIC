import pandas as pd

class Wrapper():

    def __init__(self, SELECTED_CSV_FILE_PATH):
        self.set_selected_file(SELECTED_CSV_FILE_PATH)

    def set_selected_file(self, SELECTED_CSV_FILE_PATH):
        self._SELECTED_CSV_FILE_PATH = SELECTED_CSV_FILE_PATH
        self._DF = pd.read_csv(self._SELECTED_CSV_FILE_PATH)
        self._DF.columns = [str.strip(col) for col in self._DF.columns]

    def get_description(self):
        results_data_frame = pd.read_csv(self._SELECTED_CSV_FILE_PATH )
        results_data_frame.columns = [str.strip(col) for col in results_data_frame.columns]
        return results_data_frame.describe()

    def get_best_n_params(self, n, selected_metric):
        res = self._DF.sort_values(by=selected_metric).head(n)
        res = res[["match", "gap", "egap", selected_metric]].values
        return res