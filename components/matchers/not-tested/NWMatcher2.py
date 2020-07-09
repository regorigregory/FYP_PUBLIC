import numpy as np
from components.interfaces.Matcher import Matcher

def getOne(*argv):
    return NWMatcher2(*argv)

class NWMatcher2(Matcher):

    ### Initializing functions #################

    def __init__(self, match, mu, delta):
        # mu = mismatch
        # delta = gap
        self.match = match
        self.mismatch = mu
        self.gap = delta
        #This dictionary is used to support and debug the traceback
        self.matrices = {}

    ### Main pipeline
    def align(self, to, what):
        self.s2 = to
        self.s1 = what
        self.initializeMatrices()
        self.calculateMatrices()
        self.getTracebackPath()
        self.getAlignedSequences()
        return self.s1Aligned, self.s2Aligned


    def initializeMatrices(self,):

        self.matrices = {}
        self.matrices["scores"] = np.zeros([len(self.s1)+1, len(self.s2)+1])
        self.matrices["moves"] = np.zeros([len(self.s1)+1, len(self.s2)+1])
        self.matrices["moves"][0, 1:] = 2
        self.fillUpFirstRows(self.matrices["scores"])
        self.fillUpFirstRows(self.matrices["scores"].T)

        self.matrices["tracebackIndices"] = {0: [-1, 0], 1: [-1, -1], 2: [0, -1]}
        self.matrices["tracebackMapping"] = {0: "top", 1: "diag", 2: "left"}

    def fillUpFirstRows(self, matrix):
        for i in range(len(matrix[0])):
            if (i != 0):
                matrix[0, i] = matrix[0, int(i - 1)] + self.gap

### Calculating the matrix ####################

    def calculateMatrices(self):
        i, j = 1, 1
        for i in range(1, len(self.s1)+1):
            for j in range(1, len(self.s2) + 1):
                traceBackIndex, newValue =self.getBestScore(i,j)
                self.matrices["moves"][i, j] = traceBackIndex
                self.matrices["scores"][i, j] = newValue

    def getBestScore(self, i, j):
        leftIndices, topIndices, diagIndices = self.getNeighbourIndices(i, j)
        diagScoreRaw = self.match if (self.isMatch(i, j)) else self.mismatch

        fromLeftScore = self.matrices["scores"][leftIndices[0], leftIndices[1]] + self.gap
        fromTopScore = self.matrices["scores"][topIndices[0], topIndices[1]] + self.gap
        fromDiagScore = self.matrices["scores"][diagIndices[0], diagIndices[1]]+diagScoreRaw

        results = np.array([fromTopScore, fromDiagScore, fromLeftScore])
        #to keep track of the direction
        max_element_index = np.argmax(results)
        #to update the score of the currently examined cell.

        max_element_value = results[max_element_index]
        #print(max_element_value)
        return max_element_index, max_element_value

    def isMatch(self, row, column):
        assert (row > 0), "row error: The passed indices have to be greater than 0"
        assert (column > 0), "column error: The passed indices have to be greater than 0"
        assert (column <= len(
            self.s2)), "column error: The passed indices cannot be greater than the length of s1: isMatch(self, row, column)."
        assert (row <= len(
            self.s1)), "row error: The passed indices cannot be greater than the length of s2: isMatch(self, row, column)."
        # This is necessary as the matrices are starting at 0.
        column, row = column - 1, row - 1

        return self.s1[int(row)] == self.s2[int(column)]

    def getNeighbourIndices(self, i, j):
        assert(i>=1 and j>=1), "The indices have to be greater than or equal to 1."
        leftIndices = [i, int(j - 1)]
        topIndices = [int(i - 1), j]
        diagIndices = [int(i - 1), int(j - 1)]

        return leftIndices, topIndices, diagIndices

### Traceback functions #####################

    def getTracebackPath(self):
        curY, curX = self.getTracebackStart()
        moves = list()
        while (curY!=0 and curX!=0):
            previousMove = int(self.matrices["moves"][curY, curX])
            try:
                nexCoordinates = self.matrices["tracebackIndices"][previousMove]
            except:
                print("There has been an error with the following previous move:")
                print(previousMove)

            curY += nexCoordinates[0]
            curX += nexCoordinates[1]
            moves.append(self.matrices["tracebackMapping"][previousMove])
        #moves
        self.lastAlignmentPath =  reversed(moves)


    def getTracebackStart(self):

        yMaxIndex = np.argmax(self.matrices["scores"][:, len(self.s2)])
        xMaxIndex = np.argmax(self.matrices["scores"][len(self.s1), :])

        yMaxValue = self.matrices["scores"][ yMaxIndex, len(self.s2)]
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