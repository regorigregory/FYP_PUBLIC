from components.interfaces.Matcher import Matcher

# python packages
import numpy as np
import warnings
import time
import multiprocessing as mp
import concurrent
import traceback
#print("Number of processors: ", mp.cpu_count())
from components.matchers.OriginalMatcher import Wrapper as v1
class Wrapper(v1):
    def generateDisparity(self):
        try:
            scanlines = np.zeros(self.im1.shape)

            for index in range(len(self.matrices)):
                scanline = np.zeros(self.columns)

                lefts = 0
                tops = 0
                currentPixel = 0

                for direction in self.matrices[index]["tracebackPath"]:
                    if(direction == "left"):
                        lefts+=1
                    elif(direction == "top"):
                        tops += 1
                        scanline[currentPixel] = 0
                        currentPixel+=1
                    elif (direction == "diag"):
                        scanline[currentPixel]=lefts-tops #+ self.im1[index, int(currentPixel)]
                        currentPixel+=1
                    else:
                        print("Something is not right here!")
                        raise Exception
                scanlines[index]=scanline

            self.lastDisparity = np.asarray(scanlines)
        except(Exception):
            print("Unexpected error.")
            traceback.print_exc()


