import logging


class ExperimentLogger():

    def __init__(self):
        self.logging = logging
        self.logging.basicConfig(filename="matching.log", filemode = "a", level = logging.INFO, format="%(asctime)s: %(message)s")
        """is_parallel: whether the algorithm is run parallelly
         img_name: img name from the dataset
         alg_type: SimpleParallel/patch_matcher
         is_img_preprocessed: yes or no,
         convolution_filters: list of convolutional filters used,
         filter_strategy: if multiple filters used, how are their results merged
         matching_mode: occ, non occ
         match: match
         gap: gap
         egap:egap
         matrix_init_mode: matrix_init_mode 0, 1 or 2 atm"""

    def log_matching_started(self, spec_dict):

        message = "MATCH_START: " \
                  "img_name: {img_name};" \
                  " is_img_preprocessed: {is_img_preprocessed};" \
                  "alg_type: {alg_type};" \
                  " is_parallel: {is_parallel}; " \
                  " match: {match};" \
                  " gap: {gap};" \
                  " egap: {egap};" \
                  " matrix_init_mode: {matrix_init_mode}" \
                  "convolution_filters: {convolution_filters};" \
                  " filter_strategy: {filter_strategy}," \
                  " matching_mode: {matching_mode}"\
            .format(
            is_parallel=spec_dict["is_parallel"],
            img_name=spec_dict["img_name"],
            alg_type=spec_dict["alg_type"],
            is_img_preprocessed=spec_dict["is_img_preprocessed"],
            convolution_filters = spec_dict["convolution_filters"],
            filter_strategy=spec_dict["filter_strategy"],
            matching_mode=spec_dict["matching_mode"],
            match=spec_dict["match"],
            gap=spec_dict["gap"],
            egap= spec_dict["egap"],
            matrix_init_mode=spec_dict["matrix_init_mode"]
               )
        self.logging.info(message)

    def log_matching_ended(self, spec_dict):
        test = "test"
        message = "MATCH_END: " \
                  "img_name: {img_name};" \
                  " is_img_preprocessed: {is_img_preprocessed};" \
                  " alg_type: {alg_type};" \
                  " is_parallel: {is_parallel}; " \
                  " match: {match};" \
                  " gap: {gap};" \
                  " egap: {egap};" \
                  " matrix_init_mode: {matrix_init_mode}" \
                  " convolution_filters: {convolution_filters};" \
                  " filter_strategy: {filter_strategy}," \
                  " matching_mode: {matching_mode}; " \
                  " runtime: {runtime}; " \
                  " output_file_path: {output_file_path}" \
                  " Metrix@ 1: {bad1}," \
                  " b1.5: {bad15}," \
                  " b2:{bad2}," \
                  " b10:{bad10}" \
                  " avg_err: {avg_err}"\
            .format(
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
            output_file_path = spec_dict["output_file_path"],
            bad1 = spec_dict["bad1"],
            bad15= spec_dict["bad15"],
            bad2=spec_dict["bad2"],
            bad10=spec_dict["bad10"],
            avg_err=spec_dict["avg_err"]
        )
        self.logging.info(message)
        print(message)
