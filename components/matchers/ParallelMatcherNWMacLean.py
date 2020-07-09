from components.interfaces.Matcher import Matcher
from components.interfaces.SequentialMatcher import SequentialMatcher
from components.interfaces.ParallelMatcher import ParallelMatcher

# python packages
import numpy as np
import time
import concurrent
import sys

def getOne(*argvs, **kwargs):
    return Wrapper(*argvs, **kwargs)


class Wrapper(ParallelMatcher, SequentialMatcher, Matcher):

    def __init__(self, match, gap, egap, dmax=64):
        self.match = match
        self.gap = gap
        self.egap = egap
        self.matrices = None
        self.dmax = dmax

    def initialize(self, to, what):
        message = "Matcher is being initalized"
        # is it necessary at all?

        self.im1 = what
        self.im2 = to
        self.rows = int(to.shape[0])
        self.columns = int(to.shape[1])

        self.initializeMatrices(self.im1, self.im2)

        message = "Matcher has been initialized."
        print(message)

    def initializeMatrices(self, img1, img2):
        assert (img1.shape == img2.shape), "The passed images don't have the same dimensions."

        self.initializeMatrixTemplate(img1[0], img2[0])
        self.matrices = []
        for row in range(self.rows):
            self.matrices.append(self.rowMatrixTemplate.copy())
        #print(len(self.matrices))


    def initializeMatrixTemplate(self, s1, s2):

        self.rowMatrixTemplate = {}
        self.rowMatrixTemplate["scores"] = np.zeros([len(s1) + 1, len(s2) + 1])
        # self.rowMatrixTemplate["scores"] .fill(np.finfo("d").min)
        self.rowMatrixTemplate["moves"] = np.zeros([len(s1) + 1, len(s2) + 1])
        # as it is going horizontally: 2=left
        self.maclean_moves(self.rowMatrixTemplate["moves"])
        self.maclean_init(self.rowMatrixTemplate["scores"])
        # self._first_rows_fill_func(self, self.rowMatrixTemplate["scores"].T)

        self.rowMatrixTemplate["tracebackPath"] = None
        self.rowMatrixTemplate["tracebackIndices"] = {0: [-1, 0], 1: [-1, -1], 2: [0, -1]}
        self.rowMatrixTemplate["tracebackMapping"] = {0: "top", 1: "diag", 2: "left"}


    def maclean_init(self, matrix):
        for i in range(0, matrix.shape[0]):
            matrix[i:, i] = np.array([(i) * self.gap for i in range(i, matrix.shape[0])]).T
            matrix[i, i:] = np.array([(i) * self.gap for i in range(i, matrix.shape[0])])

    def maclean_moves(self, matrix):
        for i in range(0, matrix.shape[0]):
            matrix[i:, i] = 0
            matrix[i, i:] = 2

    def alignImages(self):

        for i in range(0, self.rows):
            # this is to ensure that superclass methods won't break - "upwards compatibility

            self.alignImagesBody(i)

    def alignImagesBody(self, i):

        try:
            tic = int(round(time.time() * 1000))
            # print("Aligning scanlines %d has started." % (i + 1))

            self.align(self.im1[i], self.im2[i], i)
            toc = int(round(time.time() * 1000))
            tictoc = toc - tic
            # print("Aligning scanlines %d has finished. It has taken: %2.2f seconds." % ((i + 1), (tictoc / 1000)))
            # last alignment path is stored in:
        except IndexError:
            print("The follwing index is out of range: %d" % (i))
        return {i: self.matrices[i]}

    def align(self, to, what, index):

        self.calculateMatrices(to, what, index)

        self.getTracebackPath(index)

    def alignImagesParallel(self):
        ticMain = int(round(time.time() * 1000))
        myRange = [row for row in range(self.rows)]
        # print(myRange)
        self.myRange = myRange
        results = None
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.alignImagesBody, myRange)
        tocMain = int(round(time.time() * 1000))
        tictocMain = tocMain - ticMain
        print("Aligning scanlines has finished. It has taken: %2.2f seconds." % ((tictocMain / 1000)))
        return results



    def calculateMatrices(self, to, what, currentIndex):

        limit: int = self.columns + 1
        for i in range(1, limit):
            starting_index = 1 if i <= self.dmax + 1 else i - self.dmax
            for j in range(starting_index, i+1):
                traceBackIndex, newValue = self.getBestScore(i, j, currentIndex)
                self.matrices[currentIndex]["moves"][i, j] = traceBackIndex
                self.matrices[currentIndex]["scores"][i, j] = newValue


    def getBestScore(self, yAxis, xAxis, currentIndex):

        # This is totally fucked up. go away.
        leftIndices, topIndices, diagIndices = self.getNeighbourIndices(yAxis, xAxis)

        p1 = (self.im1[currentIndex, int(xAxis - 1)]).astype(np.float64)
        p2 = (self.im2[currentIndex, int(yAxis - 1)]).astype(np.float64)

        diagScoreRaw = self.match - np.abs(p1 - p2)
        fromDiagScore = self.matrices[currentIndex]["scores"][diagIndices[0], diagIndices[1]] + diagScoreRaw

        fromLeftDirection = self.getDirection(leftIndices, currentIndex)
        fromTopDirection = self.getDirection(topIndices, currentIndex)

        table = self.matrices[int(currentIndex)]["scores"]
        fromLeftScore = table[leftIndices[0], leftIndices[1]]
        fromTopScore = table[topIndices[0], topIndices[1]]

        # here something dodgy is going on here, I don't remember why else self.gap and not 0.
        # perhaps because we are determining which direction would result in the maximum score?

        fromLeftScore += self.egap if fromLeftDirection == "left" else self.gap
        fromTopScore += self.egap if fromTopDirection == "top" else self.gap

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
            previousMove = self.matrices[currentIndex]["moves"][curY, curX]
            try:
                nexCoordinates = self.matrices[currentIndex]["tracebackIndices"][previousMove]
            except:
                print("There has been an error with the following previous move:")
                print(previousMove)

            curY += nexCoordinates[0]
            curX += nexCoordinates[1]
            # message = "Traceback starging indices: %d %d"%(curX, curY)
            # warnings.warn(message)
            moves.append(self.matrices[currentIndex]["tracebackMapping"][previousMove])

        self.matrices[currentIndex]["tracebackPath"] = list(reversed(moves))

    def recompileObject(self, generator):
        res = [r for r in generator]
        arr = [i for i in range(self.im1.shape[0])]
        for r in res:
            index = [k for k in r.keys()]
            index = index[0]
            self.matrices[index] = r[index]

    def getTracebackStart(self, currentIndex):
        lastColumn = self.matrices[currentIndex]["scores"][:, self.columns]
        lastRow = self.matrices[currentIndex]["scores"][self.columns, :]
        maxValY = np.max(lastColumn)
        maxValX = np.max(lastRow)

        yMaxIndex = np.argmax(lastColumn)
        xMaxIndex = np.argmax(lastRow)

        yMaxValue = self.matrices[currentIndex]["scores"][yMaxIndex, self.columns]
        xMaxValue = self.matrices[currentIndex]["scores"][self.columns, xMaxIndex]

        return (yMaxIndex, self.columns) if (yMaxValue > xMaxValue) else (self.columns, xMaxIndex)

    def generateDisparity(self):
        try:
            scanlines = np.zeros(self.im1.shape)

            for index in range(self.rows):
                scanline = np.zeros(self.columns)
                lefts = 0
                tops = 0
                currentPixel = 0
                x = len(self.matrices[index]["tracebackPath"])

                for direction in self.matrices[index]["tracebackPath"]:
                    if (direction == "left"):
                        lefts += 1
                    elif (direction == "top"):
                        tops += 1
                        scanline[currentPixel] = 0
                        currentPixel += 1
                    elif (direction == "diag"):
                        scanline[currentPixel] = np.abs(tops - lefts)
                        currentPixel += 1
                    else:
                        print("Something is not right here!")
                        raise Exception
                    #print(scanline.shape)
                scanlines[index] = scanline

            self.lastDisparity = np.asarray(scanlines)
        except(Exception):
            print("Unexpected error:", sys.exc_info()[0])