import glob
import subprocess
#Code was built based on: https://github.com/ozendelait/rvc_devkit/tree/release

import sys
sys.path.append("../../")

from  components.utils.SimpleProgressBar import SimpleProgressBar
def print_options(opts):
    for i, opt in enumerate(opts):
        print("[{0}] {1}".format(i+1, opt))
    return input()

def process_integer_input(allowed_range, question):
    print(question)
    selected_value = input()
    try:
        selected_value = int(selected_value)
        if (selected_value not in allowed_range):
            print("The provided value is not in the allowed range: {0}\n \
                  Please select a value accordingly!".format(str(allowed_range)))
            selected_value = process_integer_input(allowed_range, question)
        else:
            return selected_value
    except Exception:
        return process_integer_input(allowed_range, question)

def process_string_input(question):
    print(question)
    selected_value = input()
    try:
        selected_value = str(selected_value)
        return selected_value
    except Exception:
        return process_string_input(question)


def process_options(opts, selected_id):
    try:
        return opts[int(int(selected_id)-1)]
    except ValueError:
        print("The entered value is not an integer!\nPlease try again!")
        selected_id= print_options(opts)
        return process_options(opts, selected_id)
    except IndexError:
        print("The entered value is not a valid index!\nPlease try again!")
        selected_id = print_options(opts)
        return process_options(opts, selected_id)

def run_eval(path_to_alg_runfile, method_name, left, right, gt,
                         nonocc, outpath, kernel_width, kernel_height, match, gap, egap, test_set=False, preprocessing = "none"):
    args_to_pass=[]
    if(path_to_alg_runfile.endswith(".py")):
        args_to_pass.append(sys.executable)
    else:
        raise NotImplementedError("Non-python algorithms are not supported yet.")
    args_to_pass.extend([path_to_alg_runfile, method_name, left, right,
                         outpath, str(kernel_width), str(kernel_height), str(match), str(gap), str(egap), str(test_set)])


    if (not test_set):
        args_to_pass.extend([gt, nonocc])
    # update!!! Has to be passed as the last parameter
    args_to_pass.append(preprocessing)


    proc = subprocess.Popen(args_to_pass)

    return_code = proc.wait()
    del proc
    if return_code != 0:
        print('Algorithm {0} call failed with code {1}.'.format(path_to_alg_runfile, return_code))
        return False
    return True


if __name__ == "__main__":
    import os
    import time
    imgs = dict(left="im0.png", right="im1.png", gt = "disp0GT.pfm", nonocc="mask0nocc.png")
    working=True
    progress_bar = SimpleProgressBar.get_instance()
    ui = dict(selected_resolution=None, selected_subset=None, method_name=None, kernel_height=None,
              kernel_width=None, match=None, gap=None, egap=None
              )

    #########################################################################################################
    ########### Don't forget to change it back to run_temp after the other benchmarking in progress has ended
    #########################################################################################################

    ui["path_to_alg_runfile"] = os.path.join(".", "alg-universal", "run_v2.py")

    while(working):

        training_data_available = glob.glob("./training*")
        training_data_available_print = [dir_name[-1] for dir_name in training_data_available]
        test_data_available = glob.glob("./test*")

        print("Which resolution would you like to evaluate?")
        ui["selected_resolution"] = process_options(training_data_available_print,
                                              print_options(training_data_available_print))

        print("Which dataset would you like to benchmark?")
        selectable_datasets = ["training", "test", "*"]
        ui["selected_subset"] = process_options(selectable_datasets, print_options(selectable_datasets))

        question = "Please type in the method's name:"
        ui["method_name"] = process_string_input(question)

        preprocessing_options = ["none", "naive_median","naive_horizontal", "naive_vertical", "naive_typo", "naive_all"]
        ui["preprocessing"] = process_options(preprocessing_options, print_options(preprocessing_options))


        allowed_range = [i for i in range(1,30, 2)]
        question = "Please select kernel height (int):"

        ui["kernel_height"] = process_integer_input(allowed_range, question)

        question = "Please select kernel width (int):"
        ui["kernel_width"] = process_integer_input(allowed_range,question)

        allowed_range = [i for i in range(-1000, 1000)]

        question = "Please select match value (int):"
        ui["match"] = process_integer_input(allowed_range, question)

        question = "Please gap value(int):"
        ui["gap"] = process_integer_input(allowed_range, question)

        question = "Please egap value (int):"
        ui["egap"] = process_integer_input(allowed_range, question)

        print("The given parameters:")
        for k, v in ui.items():
            print("{0}: {1}".format(k, v))

        counter = 1

        p = os.path.join(".", ui["selected_subset"]+ui["selected_resolution"],"*", imgs["left"])
        lefties = sorted(glob.glob(p), reverse=True)

        p = os.path.join(".",ui["selected_subset"]+ui["selected_resolution"],"*", imgs["right"])
        righties = sorted(glob.glob(p), reverse=True)

        p = os.path.join(".",ui["selected_subset"]+ui["selected_resolution"],"*", imgs["gt"])
        gts = sorted(glob.glob(p), reverse=True)

        p = os.path.join(".",ui["selected_subset"]+ui["selected_resolution"],"*", imgs["nonocc"])
        nonoccs = sorted(glob.glob(p), reverse=True)

        #print("The found files: ")
        #for f in lefties:
        #   print(f)

        for i,l in enumerate(lefties):
            r = righties[i]
            is_test_set = True if "test" in r else False
            if(not is_test_set):
                gt = gts[i]
                nonocc = nonoccs[i]
            else:
                gt=None
                nonocc=None

            ui["outpath"] = os.path.dirname(l)
            progress_bar.progress_bar(counter-1, len(lefties), header="Generating disparities in progress.")
            left = l
            right = r

            tic = time.time()
            run_eval(ui["path_to_alg_runfile"], ui["method_name"], left, right, gt,
                         nonocc, ui["outpath"], ui["kernel_width"],
                     ui["kernel_height"], ui["match"], ui["gap"], ui["egap"],
                     test_set=is_test_set, preprocessing = ui["preprocessing"] )

            toc=time.time()
            runtime = toc-tic
            print("{0}th runtime: {1}".format(counter, runtime))
            progress_bar.progress_bar(counter, len(lefties), header="Generating disparities in progress.")
            counter+=1


