import logging
import os
import sys
class Wrapper():
    def __init__(self, file_path, default_header = True):
        self.logging = logging
        self.header = False
        self._file_path = file_path
        if(default_header): self.write_csv_header()
        self._header_func = self._get_header_legacy
        self._csv_line_func = self.format_stereo_matching_results

    def set_header_function(self, func):
        self._header_func = func
    def set_line_function(self, func):
        self._csv_line_func = func

    def write_csv_header(self):
       if(not os.path.exists(self._file_path)):
          path_components = os.path.split(self._file_path)
          try:
            os.makedirs(path_components[0])
          except FileExistsError:
              sys.stdout.write("Directory '{0}' already exists.".format(path_components[0]))
       if(not os.path.isfile(self._file_path)):
            with open(self._file_path, "a+") as f:
                message_header = self._header_func()
                message_header = ",".join(message_header)+"\n"
                f.write(message_header)
                f.close()
       else:
           sys.stdout.write("File {0} already exists, header has not been written.\n".format(self._file_path))



    def append_new_sm_results(self, spec_dict, selected_keys=None, separator = ","):
        if(selected_keys is None):
            raise Exception("Selected keys to be logged have not been specified. Please use the \"selected_keys\" keyword argument.")
        formatted_message = self._csv_line_func(spec_dict, selected_keys, separator =separator)

        with open(self._file_path, 'a+') as f:
            f.write(formatted_message+"\n")

    def print_sm_results(self, spec_dict, selected_keys=None, separator = ","):
        formatted_message = self._csv_line_func(spec_dict, selected_keys, separator =separator)
        sys.stdout(formatted_message+"\n")

    ###################################################################
    # Static methods ##################################################
    ###################################################################

    @staticmethod
    def _get_header_legacy():
        return "img_name,is_img_preprocessed,alg_type,is_parallel,match, gap,egap,matrix_init_mode,convolution_filters," \
               "filter_strategy,matching_mode,runtime,euclid_distance,mse,avg_err".split(",")
    @staticmethod
    def get_header_v2():
        return "experiment_id,match,gap,egap,algo,init_method,dataset,scene,image_filename,img_res,preprocessing_method,kernel_size,kernel_spec,are_occlusions_errors,mse,avg,eucledian,bad1,bad2,bad10,runtime"\
            .split(",")

    @staticmethod
    def get_header_v3():
        return "experiment_id,match,gap,egap,algo,init_method,dataset,scene,image_filename,img_res,preprocessing_method,kernel_size,kernel_spec,are_occlusions_errors,abs_error,mse,avg,eucledian,bad1,bad2,bad4,bad8,runtime"\
            .split(",")

    @staticmethod
    def format_stereo_matching_results(spec_dict, selected_keys, separator =","):
        csv_line_template = "{img_name}" + separator + \
                  "{is_img_preprocessed}"+ separator + \
                  "{alg_type}"+ separator + \
                  "{is_parallel}"+ separator + \
                  "{match}"+ separator + \
                  "{gap}"+ separator + \
                  "{egap}"+ separator + \
                  "{matrix_init_mode}"+ separator + \
                  "{convolution_filters}" + separator + \
                  "{filter_strategy}" + separator + \
                  "{matching_mode}" + separator + \
                  "{runtime}" + separator + \
                  "{euclid_distance}" + separator + \
                  "{mse}" + separator + \
                  "{avg_err}"
        formatted_csv_line = csv_line_template.format(
            is_parallel=spec_dict["is_parallel"],
            img_name=spec_dict["img_name"],
            alg_type=spec_dict["alg_type"],
            is_img_preprocessed=spec_dict["is_img_preprocessed"],
            convolution_filters=spec_dict["convolution_filters"],
            filter_strategy=spec_dict["filter_strategy"],
            matching_mode=spec_dict["matching_mode"],
            match=spec_dict["match"],
            gap=spec_dict["gap"],
            egap=spec_dict["egap"],
            matrix_init_mode=spec_dict["matrix_init_mode"],
            runtime=spec_dict["runtime"],

            euclid_distance=spec_dict["euclid_distance"],
            mse=spec_dict["mse"],
            avg_err=spec_dict["avg_err"]
        )
        return formatted_csv_line

    @staticmethod
    def format_stereo_matching_results_v2(spec_dict, selected_keys, separator =","):
        new_line = list()
        spec_dict = {k.upper():v for k,v in spec_dict.items()}
        selected_keys= [key.upper() for key in selected_keys]
        for key in selected_keys:
            new_line.append(str(spec_dict[key.upper()]))
        return separator.join(new_line)

if __name__ == "__main__":
    import numpy as np
    from components.matchers.OriginalMatcher import Wrapper as m
    from components.utils.Metrix import Wrapper as me
    from components.utils.CSVWriter2 import Wrapper as csv

    ROOT_PATH = os.path.join("..", "..")
    EXPERIMENT_TITLE = "EXP_000-LOG_TEST"

    INIT_METHOD = "original"
    DATASET = "middlebury"

    DATASET_FOLDER = os.path.join(ROOT_PATH, "datasets", DATASET)
    LOG_FOLDER = os.path.join(ROOT_PATH, "experiments", "logs")
    CSV_FILEPATH = os.path.join(LOG_FOLDER, EXPERIMENT_TITLE + ".csv")
    IMG_RES = "400X375"
    PREPROCESSING_METHOD = "None"
    KERNEL_SIZE = 1
    KERNEL_SPEC = "None"
    MATCH = 60
    GAP = -20
    EGAP = -1
    EXP_PARAMS = {"experiment_id": EXPERIMENT_TITLE, "match": MATCH, "gap": GAP, "egap": EGAP, \
                  "algo": str(m.__module__), "init_method": "default", "dataset": DATASET, \
                  "preprocessing_method": "None", "kernel_size": 1, "kernel_spec": "None"}
    CSV_FILEPATH = os.path.join(LOG_FOLDER, EXPERIMENT_TITLE + ".csv")
    EXP_PARAMS["are_occlusions_errors"] = ARE_OCCLUSIONS_ERRORS = True
    disp = np.zeros([10,10])
    gt = np.zeros([10, 10])
    occ = np.zeros([10,10])

    disp.fill(10)
    gt.fill(10)
    occ.fill(10)
    occ[0]=0
    EXP_PARAMS["bad1"], EXP_PARAMS["bad2"], EXP_PARAMS["bad4"], EXP_PARAMS["bad8"], EXP_PARAMS["abs_error"], EXP_PARAMS["mse"], EXP_PARAMS["avg"], EXP_PARAMS["eucledian"] =\
    BAD1, BAD2, BAD4, BAD8, ABS_ERROR, MSE, AVG, EUCLEDIAN = me.evaluate_over_all(disp, gt, occ, occlusions_counted_in_errors = ARE_OCCLUSIONS_ERRORS)

    EXP_PARAMS["scene"] = "TEST"
    EXP_PARAMS["runtime"] = 10

    EXP_PARAMS["image_filename"] = "test.png"
    EXP_PARAMS["img_res"] = "{0}x{1}".format(10,10)



    csv_logger = csv(CSV_FILEPATH, default_header=False)
    csv_logger.set_header_function(csv_logger.get_header_v2)
    csv_logger.write_csv_header()
    csv_logger.set_line_function(csv.format_stereo_matching_results_v2)
    csv_logger.append_new_sm_results(EXP_PARAMS, csv.get_header_v3())
    csv_logger.append_new_sm_results(EXP_PARAMS, csv.get_header_v3())

