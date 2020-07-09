from components.matchers.ParallelMatcherNW2 import Wrapper as ParallelMatcherNW2


# python packages
import numpy as np


def getOne(*argvs, **kwargs):
    return ParallelMatcherNW3(*argvs, **kwargs)


class ParallelMatcherNW3(ParallelMatcherNW2):

    def initialize(self, to, what):

        self.im1 = what
        self.im2 = to

        self.im1_padded = np.pad(self.im1, (1,1), mode="constant")
        self.im2_padded = np.pad(self.im2, (1,1), mode="constant")

        self.rows = int(to.shape[0])
        self.columns = int(to.shape[1])

        self.initializeMatrices(self.im1, self.im2)

        message = "Matcher has been initialized."
        print(message)

    def getBestScore(self, yAxis, xAxis, currentIndex):

        # This is totally fucked up. go away.
        leftIndices, topIndices, diagIndices = self.getNeighbourIndices(yAxis, xAxis)

        p1 = (self.im1[currentIndex, int(xAxis - 1)]).astype(np.float64)
        p2 = (self.im2[currentIndex, int(yAxis - 1)]).astype(np.float64)

        diagPathAbsValue = self.getPatchAbsoluteDifference(currentIndex, int(xAxis - 1), int(yAxis - 1))
        diagScoreRaw = self.match - diagPathAbsValue
        fromDiagScore = self.matrices[currentIndex]["scores"][diagIndices[0], diagIndices[1]] + diagScoreRaw

        fromLeftDirection = self.getDirection(leftIndices, currentIndex)
        fromTopDirection = self.getDirection(topIndices, currentIndex)

        table = self.matrices[int(currentIndex)]["scores"]
        fromLeftScore = table[leftIndices[0], leftIndices[1]]
        fromTopScore = table[topIndices[0], topIndices[1]]

        # something dodgy is going on here, I don't remember why else self.gap and not 0.
        # perhaps because we are determining which direction would result in the maximum score?

        fromLeftScore += self.egap if fromLeftDirection == "left" else self.gap
        fromTopScore += self.egap if fromTopDirection == "top" else self.gap

        results = np.array([fromTopScore, fromDiagScore, fromLeftScore])

        # to keep track of the direction

        max_element_index = np.argmax(results)

        # to update the score of the currently examined cell.

        max_element_value = results[max_element_index]

        return max_element_index, max_element_value

    def getPatchAbsoluteDifference(self, currentIndex, im1_pixel_index, im2_pixel_index):
        startRow = currentIndex
        endRow = currentIndex+3 #slicing is not inclusive


        #what if you are at the edges? you need to handle it
        img1_start_column = im1_pixel_index
        img1_end_column = im1_pixel_index+3
        patch1 = np.asarray(self.im1_padded[startRow:endRow, img1_start_column:img1_end_column])

        img2_start_column = im2_pixel_index
        img2_end_column = im2_pixel_index + 3
        patch2 = np.asarray(self.im2_padded[startRow:endRow, img2_start_column:img2_end_column])
        absolute_difference = np.abs(patch1-patch2)
        sum_of_absolute_difference = np.sum(absolute_difference)
        return sum_of_absolute_difference

