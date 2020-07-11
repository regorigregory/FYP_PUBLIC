import glob
import subprocess
import sys
from  components.utils.SimpleProgressBar import SimpleProgressBar
def print_options(opts):
    for i, opt in enumerate(opts):
        print("[{0}] {1}".format(i+1, opt))
    return input()
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

def run_eval(method_name, left, right, path_to_alg_runfile, outpath):
    if(path_to_alg_runfile.endswith(".py")):
        args_to_pass = [sys.executable]
    else:
        raise NotImplementedError("Non-python algorithms are not supported yet.")
    args_to_pass.extend([path_to_alg_runfile, method_name, left, right, outpath])

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
    imgs = dict(left="im0.png", right="im1.png")
    working=True
    progress_bar = SimpleProgressBar.get_instance()
    while(working):

        algs_available = glob.glob("alg-*")

        print("Which algorithm would you like to run?")

        selected_alg = process_options(algs_available, print_options(algs_available))

        training_data_available = glob.glob("training*")
        training_data_available_print = [dir_name[-1] for dir_name in training_data_available]
        test_data_available = glob.glob("test*")

        print("Which resolution would you like to evaluate?")
        selected_resolution= process_options(training_data_available_print, print_options(training_data_available_print))


        lefties = sorted(glob.glob("./*"+selected_resolution+"/*/"+imgs["left"]))
        righties = sorted(glob.glob("./*"+selected_resolution+"/*/"+imgs["right"]))

        alg_runfile = glob.glob(selected_alg+"/run*")[0]
        counter =1
        for l,r in zip(lefties, righties):
            outpath = os.path.abspath(os.path.dirname(l))
            tic = time.time()
            progress_bar.progress_bar(counter-1, len(lefties), header="Generating disparities in progress.")
            run_eval(selected_alg, os.path.abspath(l), os.path.abspath(r), alg_runfile, outpath)
            toc=time.time()
            runtime = toc-tic
            print("{0}th runtime: {1}".format(counter, runtime))
            progress_bar.progress_bar(counter, len(lefties), header="Generating disparities in progress.")
            counter+=1


