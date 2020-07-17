import os
import numpy as np
import cv2
import glob
import project_helpers

def add_masks_to_raw_disparity(disp, occl):
    mask = (occl==0)
    mod_disp = disp.copy()
    mod_disp[mask] = 0
    return mod_disp
def get_image_paths(rootpath=os.path.join(project_helpers.get_project_dir(), "datasets", "middlebury"), year=2003, scene="teddy", size=""):
    size = size if size != "Q" else ""
    directory = os.path.join(rootpath, "middlebury_"+str(year), scene+size)
    imgs = list(["im2", "im6", "disp2", "nonocc"])
    extensions = ["png", "pgm", "ppm"]
    paths = list()
    print(directory)
    for img in imgs:
        for ext in extensions:
            found = False
            temp_path = os.path.join(directory, img+"."+ext)
            if os.path.isfile(temp_path):
                paths.append(temp_path)
                found=True
                break
        if not found:
            raise Exception("The dataset folder is corrupt. The file '{0}' is missing".format(temp_path))

        """
            temp_path = os.path.join(directory, img+"*")
            scan = glob.glob(temp_path)
            paths.append(temp_path[0])
        return paths"""
    return paths

def get_images(rootpath, year, scene, size="", grayscale=True):
    img_paths = get_image_paths(rootpath, year, str(scene), size)
    loaded_imgs = list()
    for path in img_paths:
        read_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        temp_img = cv2.imread(path, read_mode)
        print(path)
        loaded_imgs.append(temp_img)

    return np.array(loaded_imgs), np.array(img_paths)

def get_images_v2(rootpath, dataset, year, size="Q", grayscale=True):
    img_directory = os.path.join(rootpath,dataset, dataset+"_"+str(year), size, "*.png")
    img_paths = glob.glob(img_directory)
    if(len(img_paths)!=24):
        raise Exception("The given directory does not contain every image for all the four datasets (4*6).")
    SCENES = ["teddy", "cones", "tsukaba", "venus"]
    scene_dict = dict(im2=None, im6=None, nonocc=None, disc=None, groundtruth=None, unknown=None)
    IMAGE_PATH_DICT = dict((scene, scene_dict.copy()) for scene in SCENES)
    LOADED_IMG_DICT = dict((scene, scene_dict.copy()) for scene in SCENES)

    for path in img_paths:
        filename = os.path.split(path)[-1][0:-4]
        scene, img_type = filename.split("_")[0], filename.split("_")[1]
        read_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        temp_img = cv2.imread(path, read_mode)
        print(path)
        IMAGE_PATH_DICT[scene][img_type] =path
        LOADED_IMG_DICT[scene][img_type] =temp_img
    return LOADED_IMG_DICT,IMAGE_PATH_DICT

def read_image_binary(path):
    with open(path, "rb") as f:
        b = f.read()
        return b

def get_gt_paths_2003(size="Q", mask="groundtruth"):
    ds_root = project_helpers.get_datasets_dir()
    ds_subdir = os.path.join("middlebury", "middlebury_2003")
    if size != "Q" and size != "H":
        raise Exception("Unsupported size '{0}' has been passed as a parameter.".format(size))
    ext = ".pgm" if size == "H" and mask != "nonocc" else ".png"
    found_files = glob.glob(os.path.join(ds_root, ds_subdir, size, "*"+mask + ext))
    return found_files



def get_groundtruths_files_2003(size="Q", mask="groundtruth", binary_mode=False):
    found_files = get_gt_paths_2003(size=size, mask=mask)
    read_function = (lambda path: read_image_binary(path)) if binary_mode else (lambda path: cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    loaded_images = {os.path.split(path)[1].split("_")[0]: read_function(path) for path in found_files}
    return loaded_images

def convert_pfm_to_png_for_viz(path_to_pfm):

    temp = cv2.imread(path_to_pfm, cv2.IMREAD_UNCHANGED)
    temp = np.where(temp == np.inf, 0, temp)
    temp = temp / temp.max() * 256
    return temp

if __name__ == "__main__":
    import os
    import cv2
    import matplotlib.pyplot as plt
    ROOT = project_helpers.get_project_dir()

    SELECTED_DATASET = "middlebury_2003"
    SELECTED_SCENE = "teddy"
    SELECTED_IMAGE = "2-6"
    DATASET_ROOT = os.path.join(ROOT, "datasets", "middlebury")
    IMG_LOAD_PATH = os.path.join(ROOT, "datasets", "middlebury", SELECTED_DATASET, SELECTED_SCENE+"H" )
    test_paths = get_image_paths(os.path.join(ROOT, "datasets", "middlebury"), 2003, "cones", size="H")
    read_images = []
    """for p in test_paths:
        read_images.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    loaded_imgs, loaded_paths = get_images(DATASET_ROOT, 2003, "cones", size="H")"""