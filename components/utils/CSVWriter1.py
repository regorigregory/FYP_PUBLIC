import logging
import os
import sys
class Wrapper():

    def __init__(self, file_path = "./logs/numba_simple_scores.csv"):
        self.logging = logging
        self.header = False
        self._file_path = file_path
        self.write_csv_header()
        self.logging.basicConfig(filename=file_path, filemode = "a", level = logging.INFO, format="%(message)s")

    def log_matching_started(self, spec_dict):
        pass

    def write_csv_header(self):

       if(not os.path.exists(self._file_path)):
          path_components = os.path.split(self._file_path)
          try:
            os.makedirs(path_components[0])
          except FileExistsError:
              sys.stdout.write("Directory '{0}' already exists.".format(path_components[0]))
       if(not os.path.isfile(self._file_path)):
            with open(self._file_path, "a+") as f:
                message_header = "img_name,is_img_preprocessed,alg_type,is_parallel,match, gap,egap,matrix_init_mode,convolution_filters," \
                                  "filter_strategy,matching_mode,runtime,output_file_path,bad1,bad15,bad2,bad10,avg_err\n"
                f.write(message_header)
                f.close()
       else:
           sys.stdout.write("File {0} already exists, header has not been written.\n".format(self._file_path))


    def format_stereo_matching_results(self, spec_dict, separator =","):
        message = "{img_name}" + separator + \
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
                  "{output_file_path}" + separator + \
                  "{bad1}" + separator + \
                  "{bad15}" + separator + \
                  "{bad2}" + separator + \
                  "{bad10}" + separator + \
                  "{avg_err}" + "\n"
        formatted_message = message.format(
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
            output_file_path=spec_dict["output_file_path"],
            bad1=spec_dict["bad1"],
            bad15=spec_dict["bad15"],
            bad2=spec_dict["bad2"],
            bad10=spec_dict["bad10"],
            avg_err=spec_dict["avg_err"]
        )
        return formatted_message

    def append_new_sm_results(self, spec_dict, separator = ","):
        formatted_message = self.format_stereo_matching_results(spec_dict, separator =",")
        with open(self._file_path, 'a+') as f:
            f.write(formatted_message+"\n")

    def print_sm_results(self, spec_dict, separator = ","):
        formatted_message = self.format_stereo_matching_results(spec_dict, separator=",")
        sys.stdout(formatted_message+"\n")

    def log_sm_results(self, spec_dict, separator = ","):
        self.log_matching_ended(spec_dict, separator)

    def log_matching_ended(self, spec_dict, separator = ","):
        formatted_message = self.format_stereo_matching_results(spec_dict, separator =",")
        self.logging.info(formatted_message)