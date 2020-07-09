from components.interfaces.Matcher import Matcher

# python packages
import numpy as np
import time

def getOne(*argv):
    return NWImageMatcher(*argv)

class NWImageMatcher():


    def __init__(self, match, gap, egap):
        self.match = match
        self.gap = gap
        self.egap = egap
        self.matrix3D = []
        self.lastAlignmentPaths = []

    def initialize(self, to, what):
        message = "Matcher is being initalized"
        self.initialize3DMatrices(to, what)
        #is it necessary at all?
        self.s1 = to[0]
        self.s2 = what[0]
        self.im1=to
        self.im2 = what
        self.initialize3DMatrices(to, what)
        message = "Matcher has been initialized."
        print(message)

    def initialize3DMatrices(self, img1, img2):
        assert (img1.shape == img2.shape), "The passed images don't have the same dimensions."
        rows = img1.shape[0]
        columns = img1.shape[1]
        self.initializeMatricesWithEgap(img1[0], img2[0])


    def initializeMatricesWithEgap(self, row1, row2):
        self.initializeMatrices(row1, row2)
        #calls suoer class's initializeMatrices, but fillUpFirstRows is implemented in this class, hence egap is done
        self.rowMatrixTemplate = self.matrices.copy()
        # todo Then, update the scoring function, then juhuu!!!
        # todo Then, based on the insertion, generate disparity

    def initializeMatrices(self, s1, s2):
        self.matrices = {}
        self.matrices["scores"] = np.zeros([len(s1) + 1, len(s2) + 1])
        self.matrices["moves"] = np.zeros([len(s1) + 1, len(s2) + 1])
        #as it is going horizontally: 2=left
        self.matrices["moves"][0, 1:] = 2

        self.fillUpFirstRows(self.matrices["scores"])
        self.fillUpFirstRows(self.matrices["scores"].T)

        self.matrices["tracebackIndices"] = {0: [-1, 0], 1: [-1, -1], 2: [0, -1]}
        self.matrices["tracebackMapping"] = {0: "top", 1: "diag", 2: "left"}

    def fillUpFirstRows(self, matrix):
        for i in range(len(matrix[0])):
            if (i == 1):
                matrix[0, i] =  self.gap
            if(i>1):
                matrix[0, i] = self.gap+self.egap*(i-1)



    def getBestScore(self, i, j):
        leftIndices, topIndices, diagIndices = self.getNeighbourIndices(i, j)

        p1 = int(self.s1[int(j-1)])
        p2 = int(self.s2[int(j-1)])

        diagScoreRaw = self.match - np.abs(p1-p2)

        #todo modify the lines below, so that it checks wheter the previous cell was a gap or not
        # if so, add e_gap reward
        # how to: check the previous cell's direction. if it is top or left, then it was a gap inserted
        # should egap be applied if there is a direction change, such as from horizontal to vertical
        fromLeftDirection = self.getDirection(leftIndices)
        fromTopDirection = self.getDirection(topIndices)


        fromLeftScore = self.egap if fromLeftDirection == "left" else self.gap
        fromTopScore = self.egap if fromTopDirection == "top" else self.gap

        fromDiagScore = self.matrices["scores"][diagIndices[0], diagIndices[1]] + diagScoreRaw

        results = np.array([fromTopScore, fromDiagScore, fromLeftScore])
        # to keep track of the direction
        max_element_index = np.argmax(results)
        # to update the score of the currently examined cell.

        max_element_value = results[max_element_index]
        # print(max_element_value)
        return max_element_index, max_element_value

    def getDirection(self, coordinates):
        denaryDirection = self.matrices["moves"][coordinates[0], coordinates[1]]
        return self.matrices["tracebackMapping"][denaryDirection]





    def getNeighbourIndices(self, i, j):
        assert (i >= 1 and j >= 1), "The indices have to be greater than or equal to 1."
        leftIndices = [i, int(j - 1)]
        topIndices = [int(i - 1), j]
        diagIndices = [int(i - 1), int(j - 1)]

        return leftIndices, topIndices, diagIndices

        ### Traceback functions #####################

    def alignImages(self):
        ticMain = int(round(time.time() * 1000))
        for i in range(0, (self.im1.shape[0])):
            # this is to ensure that superclass methods won't break - "upwards compatibility

            self.s1 = self.im1[i]
            self.s2 = self.im2[i]
            try:
                tic = int(round(time.time() * 1000))
                print("Aligning scanlines %d has started."%(i+1))
                self.align(self.s1, self.s2)
                self.matrix3D.append(self.matrices.copy())
                toc = int(round(time.time() * 1000))
                tictoc = toc-tic
                print("Aligning scanlines %d has finished. It has taken: %2.2f seconds." % ((i + 1), (tictoc/1000)))


                self.lastAlignmentPaths.append(self.lastAlignmentPath.copy())
                # last alignment path is stored in:
            except IndexError:
                print("The follwing index is out of range: %d"%(i))

            self.matrices = self.rowMatrixTemplate.copy()
        tocMain = int(round(time.time() * 1000))
        tictocMain = tocMain-ticMain
        print("Aligning scanlines has finished. It has taken: %2.2f seconds." % ((tictocMain / 1000)))

    def align(self, to, what):
        self.s2 = to
        self.s1 = what

        self.calculateMatrices(to, what)
        print("Alignment has been done.")
        print("Calculationg the traceback path has started.")
        self.getTracebackPath()
        print("Calculationg the traceback path has finished.")


    def calculateMatrices(self, to, what):
        i, j = 1, 1
        for i in range(1, len(to) + 1):
            for j in range(1, len(what) + 1):
                traceBackIndex, newValue = self.getBestScore(i, j)
                self.matrices["moves"][i, j] = traceBackIndex
                self.matrices["scores"][i, j] = newValue


    def getTracebackPath(self):
        curY, curX = self.getTracebackStart()

        moves = list()
        while (curY != 0 and curX != 0):
            previousMove = int(self.matrices["moves"][curY, curX])
            try:
                nexCoordinates = self.matrices["tracebackIndices"][previousMove]
            except:
                print("There has been an error with the following previous move:")
                print(previousMove)

            curY += nexCoordinates[0]
            curX += nexCoordinates[1]
            #message = "Traceback starging indices: %d %d"%(curX, curY)
            #warnings.warn(message)
            moves.append(self.matrices["tracebackMapping"][previousMove])
        self.moves = moves
        self.lastAlignmentPath = list(reversed(moves))

    def getTracebackStart(self):

        yMaxIndex = np.argmax(self.matrices["scores"][:, len(self.s2)])
        xMaxIndex = np.argmax(self.matrices["scores"][len(self.s1), :])

        yMaxValue = self.matrices["scores"][yMaxIndex, len(self.s2)]
        xMaxValue = self.matrices["scores"][len(self.s1), xMaxIndex]

        return (yMaxIndex, len(self.s2)) if (yMaxValue > xMaxValue) else (len(self.s1), xMaxIndex)

    def getAlignedSequences(self):

        what = self.s2
        to = self.s1

        toIndex = -1
        whatIndex = -1

        toAligned = ""
        whatAligned = ""

        for move in self.lastAlignmentPath:
            if (move == "diag"):
                toIndex += 1
                whatIndex += 1

                toAligned += to[toIndex]
                whatAligned += what[whatIndex]


            elif (move == "top"):

                whatAligned += "_"
                toIndex += 1
                toAligned += to[toIndex]
            else:
                toAligned += "_"
                whatIndex += 1
                whatAligned += what[whatIndex]

        self.s1Aligned = toAligned
        self.s2Aligned = whatAligned

    def generateDisparity(self):

        scanlines = np.zeros(self.im1.shape)

        for index, path in enumerate(self.lastAlignmentPaths):
            scanline = np.zeros(self.im1.shape[1])
            lefts = 0
            tops = 0
            currentPixel = 0
            for direction in path:
                if (direction == "left"):
                    lefts += 1
                elif (direction == "top"):
                    tops += 1
                else:
                    scanline[currentPixel] = (np.abs(tops - lefts))
                    currentPixel += 1
            scanlines[index] = scanline

        self.lastDisparity = np.asarray(scanlines)
