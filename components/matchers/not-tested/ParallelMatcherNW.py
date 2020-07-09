from components.interfaces.Matcher import Matcher

# python packages
import numpy as np
import time
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())
from numba import jit, prange

def getOne(*argv):
    return ParallelMatcherNW(*argv)

class ParallelMatcherNW():


    def __init__(self, match, gap, egap):
        self.match = match
        self.gap = gap
        self.egap = egap
        self.matrices = None
        self.alignmentPaths = None

    def initialize(self, to, what):
        message = "Matcher is being initalized"
        #is it necessary at all?

        self.im1=to
        self.im2 = what
        self.rows = int(to.shape[0])
        self.columns = int(to.shape[1])

        self.initializeMatrices(to, what)

        message = "Matcher has been initialized."
        print(message)

    def initializeMatrices(self, img1, img2):
        assert (img1.shape == img2.shape), "The passed images don't have the same dimensions."

        self.initializeMatrixTemplate(img1[0], img2[0])
        self.matrices = []
        self.alignmentPaths = []
        for row in range(self.rows):
            self.matrices.append(self.rowMatrixTemplate.copy())
            self.alignmentPaths.append(None)
        print(len(self.matrices))


    def initializeMatrixTemplate(self, s1, s2):
        self.rowMatrixTemplate = {}
        self.rowMatrixTemplate["scores"] = np.zeros([len(s1) + 1, len(s2) + 1])
        self.rowMatrixTemplate["moves"] = np.zeros([len(s1) + 1, len(s2) + 1])
        #as it is going horizontally: 2=left
        self.rowMatrixTemplate["moves"][0, 1:] = 2

        self.fillUpFirstRows(self.rowMatrixTemplate["scores"])
        self.fillUpFirstRows(self.rowMatrixTemplate["scores"].T)

        self.rowMatrixTemplate["tracebackIndices"] = {0: [-1, 0], 1: [-1, -1], 2: [0, -1]}
        self.rowMatrixTemplate["tracebackMapping"] = {0: "top", 1: "diag", 2: "left"}

    def fillUpFirstRows(self, matrix):
        for i in range(len(matrix[0])):
            if (i == 1):
                matrix[0, i] =  self.gap
            if(i>1):
                matrix[0, i] = self.gap+self.egap*(i-1)



    def alignImagesBody(self, i):
        try:
            tic = int(round(time.time() * 1000))
            print("Aligning scanlines %d has started." % (i + 1))

            self.align(self.im1[i], self.im2[i], i)
            self.getTracebackPath(i)
            toc = int(round(time.time() * 1000))
            tictoc = toc - tic
            print("Aligning scanlines %d has finished. It has taken: %2.2f seconds." % ((i + 1), (tictoc / 1000)))
            # last alignment path is stored in:
        except IndexError:
            print("The follwing index is out of range: %d" % (i))

    def alignImages(self):
        ticMain = int(round(time.time() * 1000))

        for i in range(0, self.rows):
            # this is to ensure that superclass methods won't break - "upwards compatibility
            self.alignImagesBody(i)

        tocMain = int(round(time.time() * 1000))
        tictocMain = tocMain - ticMain
        print("Aligning scanlines has finished. It has taken: %2.2f seconds." % ((tictocMain / 1000)))
    @jit(parallel = True)
    def alignImages_jit(self):
        for i in prange(0, 375):
            # this is to ensure that superclass methods won't break - "upwards compatibility
            self.alignImagesBody(i)

    def alignImagesParalell(self):
        ticMain = int(round(time.time() * 1000))

        pool = mp.Pool(mp.cpu_count())
        pool.map(self.alignImagesBody, [row for row in range(self.rows)])
        pool.close()

        tocMain = int(round(time.time() * 1000))
        tictocMain = tocMain - ticMain
        print("Aligning scanlines has finished. It has taken: %2.2f seconds." % ((tictocMain / 1000)))


    def align(self, to, what, index):

        self.calculateMatrices(to, what, index)

        print("Alignment has been done.")
        print("Calculationg the traceback path has started.")

        self.getTracebackPath(index)

        print("Calculationg the traceback path has finished.")


    def calculateMatrices(self, to, what, currentIndex):

        i, j = 1, 1
        limit = self.columns + 1
        for i in range(1, limit):
            for j in range(1, limit):

                traceBackIndex, newValue = self.getBestScore(i, j, currentIndex)
                self.matrices[currentIndex]["moves"][i, j] = traceBackIndex
                self.matrices[currentIndex]["scores"][i, j] = newValue


    def getBestScore(self, yAxis, xAxis, currentIndex):

        #This is totally fucked up. go away.
        leftIndices, topIndices, diagIndices = self.getNeighbourIndices(yAxis, xAxis)

        p1 = int(self.im1[currentIndex, int(yAxis-1)])
        p2 = int(self.im2[currentIndex, int(xAxis-1)])

        diagScoreRaw = self.match - np.abs(p1-p2)


        fromLeftDirection = self.getDirection(leftIndices, currentIndex)
        fromTopDirection = self.getDirection(topIndices, currentIndex)

        table = self.matrices[int(currentIndex)]["scores"]
        fromLeftScore = table[leftIndices[0], leftIndices[1]]
        fromTopScore = table [topIndices[0], topIndices[1]]

        # here something dodgy is going on here, I don't remember why else self.gap and not 0.
        #perhaps because we are determining which direction would result in the maximum score?

        fromLeftScore += self.egap if fromLeftDirection == "left" else self.gap
        fromTopScore += self.egap if fromTopDirection == "top" else self.gap

        fromDiagScore = self.matrices[currentIndex]["scores"][diagIndices[0], diagIndices[1]] + diagScoreRaw

        results = np.array([fromTopScore, fromDiagScore, fromLeftScore])

        # to keep track of the direction

        max_element_index = np.argmax(results)

        # to update the score of the currently examined cell.

        max_element_value = results[max_element_index]

        return max_element_index, max_element_value

    def getDirection(self, coordinates, currentIndex):
        try:
            denaryDirectionx = self.matrices[int(currentIndex)]
            denaryDirectiony = denaryDirectionx["moves"]
            denaryDirectionz = denaryDirectiony[coordinates[0], coordinates[1]]
            x = self.matrices[currentIndex]
            y = x["tracebackMapping"]
            z = y[int(denaryDirectionz)]
            return z
        except KeyError:
            print(currentIndex)



    def getNeighbourIndices(self, i, j):
        assert (i >= 1 and j >= 1), "The indices have to be greater than or equal to 1."
        leftIndices = [i, int(j - 1)]
        topIndices = [int(i - 1), j]
        diagIndices = [int(i - 1), int(j - 1)]

        return leftIndices, topIndices, diagIndices

        ### Traceback functions #####################


    def getTracebackPath(self, currentIndex):
        curY, curX = self.getTracebackStart(currentIndex)

        moves = list()
        while (curY != 0 and curX != 0):
            previousMove = int(self.matrices[currentIndex]["moves"][curY, curX])
            try:
                nexCoordinates = self.matrices[currentIndex]["tracebackIndices"][previousMove]
            except:
                print("There has been an error with the following previous move:")
                print(previousMove)

            curY += nexCoordinates[0]
            curX += nexCoordinates[1]
            #message = "Traceback starging indices: %d %d"%(curX, curY)
            #warnings.warn(message)
            moves.append(self.matrices[currentIndex]["tracebackMapping"][previousMove])

        self.alignmentPaths[currentIndex] = list(reversed(moves))

    def getTracebackStart(self, currentIndex):

        yMaxIndex = np.argmax(self.matrices[currentIndex]["scores"][:, self.columns])
        xMaxIndex = np.argmax(self.matrices[currentIndex]["scores"][self.rows, :])

        yMaxValue = self.matrices[currentIndex]["scores"][yMaxIndex, self.columns]
        xMaxValue = self.matrices[currentIndex]["scores"][self.rows, xMaxIndex]

        return (yMaxIndex, self.columns) if (yMaxValue > xMaxValue) else (self.rows, xMaxIndex)

    def generateDisparity(self):

        scanlines = np.zeros(self.im1.shape)


        for index, path in enumerate(self.alignmentPaths):
            scanline = np.zeros(self.columns)
            lefts = 0
            tops = 0
            currentPixel = 0
            for direction in path:
                if(direction == "left"):
                    lefts+=1
                elif(direction == "top"):
                    tops += 1
                    scanline[currentPixel] = 0
                    currentPixel+=1
                else:
                    scanline[currentPixel]=(np.abs(tops-lefts))
                    currentPixel+=1
            scanlines[index]=scanline

        self.lastDisparity = np.asarray(scanlines)