from enum import Enum
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Difficulity(Enum):
    MODERATE = "clean"
    NIGHTMARE = "final"

class Wrapper:
    def __init__(self, rootPath = os.path.join("datasets", "sintel", "training"), difficulity = Difficulity.MODERATE,\
                 occ_path = "occlusions" , disp_path = "disparities", outtta_path = "outofframe"):
        self.rootPath = rootPath
        self.DIRECTION  = ["_left", "_right"]
        self.right_path = os.path.join(rootPath, difficulity.value+"_right")
        self.left_path = os.path.join(rootPath, difficulity.value+"_left")
        self.difficulity = difficulity
        self.OCCLUSION = os.path.join(rootPath, occ_path)
        self.DISPARITY = os.path.join(rootPath, disp_path)
        self.OUTTAFRAME = os.path.join(rootPath, outtta_path)
        self.selected_scene_file_number = 0
        self.pointer_of_selected_scene=-1
        self.__get_all_scene_file_paths()
        self.set_selected_scene('cave_4')

    def __get_available_scenes(self):
        scene_dir_names = {i.name: 0 for i in os.scandir(self.right_path) if os.path.isdir(i.path)}
        self.__scene_dir_names = scene_dir_names

    def __get_all_scene_file_paths(self):
        self.__get_available_scenes()
        self.selected_scene_paths = {}
        for k in self.__scene_dir_names.keys():
            all_right_img = sorted([i.path for i in os.scandir(os.path.join(self.right_path, k))])
            all_left_img = sorted([i.path for i in os.scandir(os.path.join(self.left_path, k))])
            all_disparity = sorted([i.path for i in os.scandir(os.path.join(self.DISPARITY, k))])
            all_occlusion = sorted([i.path for i in os.scandir(os.path.join(self.OCCLUSION, k))])
            all_outtaframe = sorted([i.path for i in os.scandir(os.path.join(self.OUTTAFRAME, k))])
            self.selected_scene_file_number = len(all_outtaframe)
            self.pointer_of_selected_scene = -1
            self.selected_scene_paths[k] = {"right": all_right_img, "left": all_left_img,
                                            "disparity":all_disparity,
                                   "occlusions":all_occlusion, "outofframe":all_outtaframe}

    def __get_selected_scene_next_file_paths(self):
        self.pointer_of_selected_scene +=1
        max = self.selected_scene_file_number
        if (not self.pointer_of_selected_scene < max):
            self.pointer_of_selected_scene = 0
        return [
        self.selected_scene_paths[self.selected_scene_name]["left"][self.pointer_of_selected_scene],
        self.selected_scene_paths[self.selected_scene_name]["right"][self.pointer_of_selected_scene],
        self.selected_scene_paths[self.selected_scene_name]["disparity"][self.pointer_of_selected_scene],
        self.selected_scene_paths[self.selected_scene_name]["occlusions"][self.pointer_of_selected_scene],
        self.selected_scene_paths[self.selected_scene_name]["outofframe"][self.pointer_of_selected_scene]]

    def convert_disp_raw_to_real_disp(self, disp_raw):
        #channel order: blue, green, red
        R = np.floor(disp_raw[:,:, 2] / 4)
        G = np.floor(disp_raw[:,:,1] * (2 ** 6) % 256)

        DISP = R * 4 + G / (2 ** 6)
        return DISP

    def get_selected_scene_next_files(self, grayscale=True):
        paths = self.__get_selected_scene_next_file_paths()
        colormode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

        left, right, occ, outofframe = [cv2.imread(path, colormode) for i, path in enumerate(paths) if i !=2]
        occ = 255-occ
        outofframe = 255 - outofframe
        disp = cv2.imread(paths[2], cv2.IMREAD_UNCHANGED)
        disp = self.convert_disp_raw_to_real_disp(disp)

        return left.astype(np.float64), right.astype(np.float64), disp.astype(np.float64), occ.astype(
            np.float64), outofframe.astype(np.float64)

    def get_available_scenes(self):
        self.__get_available_scenes()
        return [str(i) for i in self.__scene_dir_names.keys()]

    def get_selected_scene_next_sliced(self, dim, starting_row=0, grayscale=True):
        l, r, d, o, oof = self.get_selected_scene_next_files(grayscale)
        end_row = starting_row+dim[0]
        if grayscale:
            return l[starting_row:end_row, 0:dim[1]], r[starting_row:end_row, 0:dim[1]], \
                   d[starting_row:end_row, 0:dim[1]], o[starting_row:end_row, 0:dim[1]],\
                   oof[starting_row:end_row, 0:dim[1]]
        return l[starting_row:end_row, 0:dim[1], :], r[starting_row:end_row, 0:dim[1], :], \
                   d[starting_row:end_row, 0:dim[1], :], o[starting_row:end_row, 0:dim[1], :],\
                   oof[starting_row:end_row, 0:dim[1], :]

    def get_images_by_scene_and_id(self, scene, id, grayscale = True):
        self.pointer_of_selected_scene = id-1
        self.selected_scene_name = scene
        return self.get_selected_scene_next_files(grayscale)

    def get_selected_scene_random_sclice(self):
        l, r, d, o, oof = self.get_selected_scene_next_files()

        rows = l.shape[0]
        random_scanline_index = np.random.randint(0, rows)
        scanline =  np.array([l[random_scanline_index], r[random_scanline_index], d[random_scanline_index],  o[random_scanline_index], oof[random_scanline_index]])
        scanline = scanline[:,:,np.newaxis]
        return scanline

    def concat_random_scanlines(self, i, j):
        return np.hstack([i,j])

    def get_random_selected_scene_image(self, rows):
        init = [self.get_selected_scene_random_sclice() for i in range(rows)]
        return np.stack(init, axis=1)

    def reset_iterator(self):
        self.pointer_of_selected_scene=-1

    def set_selected_scene(self, scene_selected):
        self.pointer_of_selected_scene = -1
        self.selected_scene_name = scene_selected
        self.__selected_scene_files = self.__get_all_scene_file_paths()



    def print_available_scenes(self):
        self.__get_available_scenes()
        output = ""
        for k in self.__scene_dir_names.keys():
            output+= k + ", "
        print(output)

    def get_selected_scene_file_count(self):
        return len(self.selected_scene_paths[self.selected_scene_name]["left"])

    @staticmethod
    def plot_disp_and_gt(disp, gt, disp_params, gt_params, dpi=150):
        columns = 2

        disp_title_builder = ""
        for k,v in disp_params.items():
            disp_title_builder+="{0}: {1}; ".format(k, v)

        gt_title_builder = ""
        for k,v in gt_params.items():
            gt_title_builder+="{0}: {1}; ".format(k, v)

        plt.figure(figsize=(8, 6), dpi=dpi)
        plt.subplots_adjust(wspace=0.8)
        ax = plt.subplot(1,columns, 1)
        ax.set_title(disp_title_builder)
        plt.imshow(disp, cmap = cm.viridis)

        ax = plt.subplot(1, columns, 2)
        ax.set_title(gt_title_builder)
        plt.imshow(gt, cmap=cm.viridis)
        plt.show()

    @staticmethod
    def plot_images_to_compare(imgs):
        columns = len(imgs)
        baseWidth = 100
        baseHeight = 30

        plt.figure(figsize=(8, 6), dpi=100)

        for i, img in enumerate(imgs):
            plt.subplot(1,columns, i+1)
            plt.imshow(img, cmap = "gray")
        plt.show()

    def convolve_slices_and_save(self, model, slice_height = 120, slice_width=1024, starting_row = 0, output_directory="../../optimization/pickled_slice_predictions", grayscale = False):
        import pickle

        for scene in self.get_available_scenes():
            self.set_selected_scene(scene)
            temp = self.get_selected_scene_next_sliced([slice_height, slice_width], starting_row, grayscale)
            leftee = temp[0][np.newaxis, :]
            rightee = temp[1][np.newaxis, :]

            left_prediction = model.predict(leftee)
            right_prediction = model.predict(rightee)

            left_and_right = np.array([left_prediction, right_prediction])

            # lf : left and right
            # sr : starting row
            # w  : width
            # h  : height

            filename = scene+"_im-{3}-lr_w-{0}_h-{1}_sr-{2}".format(slice_width, slice_height, starting_row, self.pointer_of_selected_scene)
            filepath = os.path.join(output_directory, filename)

            if(not os.path.isdir(output_directory)):
                os.makedirs(output_directory)

            with open(filepath, "wb") as slice_convolved:
                print("Writing file {0}".format(filepath))
                pickle.dump(left_and_right, slice_convolved)
                slice_convolved.close()

    def get_all_sliced_disparities(self, ROW_HEIGHT_PER_SLICE, NUMBER_OF_COLUMNS, STARTING_ROW_FOR_SLICES=0):

        SLICE_DIM = [ROW_HEIGHT_PER_SLICE, NUMBER_OF_COLUMNS]
        SCENES = self.get_available_scenes()
        SCENES = sorted(SCENES)

        main_temp = []

        for i in range(len(SCENES)):
            self.set_selected_scene(SCENES[i])

            # very unhealthy way of abusing the computer's resources

            temp = self.get_selected_scene_next_sliced(SLICE_DIM, starting_row=STARTING_ROW_FOR_SLICES)
            temp = np.array(temp)
            main_temp.append(temp)

        main_temp = np.array(main_temp)
        stacked_imgs = main_temp[:, 2:]
        return stacked_imgs

    @staticmethod
    def add_masks_to_raw_disparity(disp, mask1, outtaframe):
        occlusion_added = np.where(mask1 == 0, 0, disp)
        outofframe_added = np.where(outtaframe == 0, 0, occlusion_added)
        return outofframe_added

    @staticmethod
    def list_pickled_convolved(pickled_path, width, height):
        import glob
        found = glob.glob(os.path.join(pickled_path, "*w-"+str(width)+"*h-"+str(height)+"*"))
        return found
    @staticmethod
    def load_pickled_convolved(path):
        import pickle
        with open(path, "rb") as pickled_convolved:
            return np.squeeze(pickle.load(pickled_convolved))



if __name__ == "__main__":
    path = os.path.join("..", "..", "datasets", "sintel", "training")
    reader = Wrapper(rootPath=path)
    reader.print_available_scenes()
    width = 512
    height = 436
    img_dim = [height, width, 3]
    imgs =reader.get_selected_scene_next_files()
    reader.plot_images_to_compare(imgs)
    #Testing loading convolved and pickled image
    #pickleees_paths = Wrapper.list_pickled_convolved("../../optimization/pickled_slice_predictions", width, height)
    #one_pickle = Wrapper.load_pickled_convolved(pickleees_paths[0])
    """ 
    #Testing convolving images
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    try:
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1100),
             tf.config.LogicalDeviceConfiguration(memory_limit=1100)])

        logical_devices = tf.config.list_logical_devices('GPU')
        assert len(logical_devices) == len(physical_devices) + 1

        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1100),
             tf.config.LogicalDeviceConfiguration(memory_limit=1100)])
    except:
        # Invalid device or cannot modify logical devices once initialized.
        pass

    from components.utils import tf_utils as tfu

    model = tfu.get_resnet50_first_layer_model(img_dim)
    reader.convolve_slices_and_save(model,  slice_height = height, slice_width = width)
    """
    """
    #Testging reader
    
    reader.set_selected_scene('bamboo_1')
    loaded_imgs = reader.get_selected_scene_next_files()
    counter = 1
    for im in  loaded_imgs:
        plt.subplot(1, 5, counter)

        plt.imshow(im, cmap = cm.binary)
        counter+=1"""







