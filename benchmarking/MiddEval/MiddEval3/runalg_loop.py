import glob
import subprocess
import sys
from  components.utils.SimpleProgressBar import SimpleProgressBar
import numpy as np

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

    ui["path_to_alg_runfile"] = os.path.join(".", "alg-universal", "run_v2.py")
## to be continued from here!!!
    patches = [np.ones((1, 1)),
               np.ones((1, 3)),
               np.ones((1, 5)),
               np.ones((1, 7)),
               np.ones((3, 1)),
               np.ones((3, 3)),
               np.ones((3, 5)),
               np.ones((3, 7)),
               np.ones((5, 1)),
               np.ones((5, 3)),
               np.ones((7, 1)),
               np.ones((7, 3)),
               np.ones((7, 5)),
               np.ones((9, 3)),
               np.ones((9, 5)),
               np.ones((9, 7)),
               np.ones((11, 3)),
               np.ones((11, 5)),
               np.ones((13, 3)),
               np.ones((13, 5)),
               np.ones((13, 7)),
               np.ones((15, 3)),
               np.ones((15, 5)),
               np.ones((15, 7)),
               np.ones((15, 9))
               ]

    training_data_available = glob.glob("training*")
    ui["selected_resolution"] = "Q"
    ui["selected_subset"] = "training"

    p = os.path.join(".", ui["selected_subset"] + ui["selected_resolution"], "*", imgs["left"])
    lefties = sorted(glob.glob(p))

    p = os.path.join(".", ui["selected_subset"] + ui["selected_resolution"], "*", imgs["right"])
    righties = sorted(glob.glob(p))

    p = os.path.join(".", ui["selected_subset"] + ui["selected_resolution"], "*", imgs["gt"])
    gts = sorted(glob.glob(p))

    p = os.path.join(".", ui["selected_subset"] + ui["selected_resolution"], "*", imgs["nonocc"])
    nonoccs = sorted(glob.glob(p))
    counter = 1
    is_test_set = False

    options = dict(none="none")#
    #, naive_median="naive_median", naive_vertical="naive_vertical",
    #               naive_horizontal="naive_horizontal", naive_typo="naive_typo",
    #               naive_all="naive_all")


    for pv in options.values():
        for m in range (10,120, 20):
            for p in patches:
                ui["method_name"] = "bm_"+str(m)+"_"+str(p.shape[0])+"x"+str(p.shape[1])
                ui["kernel_height"] = p.shape[0]
                ui["kernel_width"] = p.shape[1]

                ui["match"] = str(m)
                ui["gap"] = str(-20)
                ui["egap"] = str(-1)

                for i,l in enumerate(lefties):
                    r = righties[i]
                    gt = os.path.abspath(gts[i])
                    nonocc = os.path.abspath(nonoccs[i])

                    ui["outpath"] = os.path.abspath(os.path.dirname(l))
                    progress_bar.progress_bar(counter-1, len(lefties)*len(patches)*len(options.values())*6, header="Generating disparities in progress.")
                    left = os.path.abspath(l)
                    right = os.path.abspath(r)

                    tic = time.time()
                    run_eval(ui["path_to_alg_runfile"], ui["method_name"], left, right, gt,
                                 nonocc, ui["outpath"], ui["kernel_width"],
                             ui["kernel_height"], ui["match"], ui["gap"], ui["egap"],
                             test_set=is_test_set,
                             preprocessing = pv)

                    toc=time.time()
                    runtime = toc-tic
                    print("{0}th runtime: {1}".format(counter, runtime))
                    progress_bar.progress_bar(counter, len(lefties)*len(patches)*len(options.values())*6, header="Generating disparities in progress.")
                    counter+=1


