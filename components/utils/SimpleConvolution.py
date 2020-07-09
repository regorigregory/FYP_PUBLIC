import numpy as np

import multiprocessing as mp


print("Number of processors: ", mp.cpu_count())

def getOne(*argv):
    return SimpleConvolution(*argv)
class SimpleConvolution():
    def __init__(self):
        self.size = 3
        self.filter = np.ones([3, 3])



    def convolve(self, img):
        height = img.shape[0]+2
        width = img.shape[1]+2

        withPadding = np.zeros([height, width])
        withPadding[1:-1, 1:-1] = img
        output = np.zeros(img.shape)
        for row in range(img.shape[0]):
            for column in range(img.shape[1]):
                slice = withPadding[row:int(row+3), column:int(column+3)]
                outputTemp = np.sum(slice * self.filter)
                output[row, column] = outputTemp
        return output

    def setFilter(self, filter):
        self.filter = filter
#In a jupyter notebook
    #Experiments - > simple averaging
        #horizontal
        #x feature
        #vertical
        #gaussian
        #L1 distance
        #merging filters
        #
        #Eucledian Distance ->note, that you have to reverse it!!!
        #merged
        #added back to the original image
        #plotting the outputs in a grid!
        #Fix parallelisation