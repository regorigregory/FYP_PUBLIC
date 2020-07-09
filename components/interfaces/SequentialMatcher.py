import abc
import numpy as np
import traceback

class SequentialMatcher(abc.ABC):

    ###################################################################
    # Default functions ###############################################
    ###################################################################
    def initialize(self, to, what):
        message = "Matcher is being initalized"
        # is it necessary at all?

        self.im1 = to
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
        for row in range(self.rows):
            self.matrices.append(self.rowMatrixTemplate.copy())
        print(len(self.matrices))

    def initializeMatrixTemplate(self, s1, s2):

        self.rowMatrixTemplate = {}
        self.rowMatrixTemplate["scores"] = np.zeros([len(s1) + 1, len(s2) + 1])
        # self.rowMatrixTemplate["scores"] .fill(np.finfo("d").min)
        self.rowMatrixTemplate["moves"] = np.zeros([len(s1) + 1, len(s2) + 1])
        # as it is going horizontally: 2=left
        self.maclean_moves(self.rowMatrixTemplate["moves"])
        self._first_rows_fill_func(self, self.rowMatrixTemplate["scores"])
        # self._first_rows_fill_func(self, self.rowMatrixTemplate["scores"].T)

        self.rowMatrixTemplate["tracebackPath"] = None
        self.rowMatrixTemplate["tracebackIndices"] = {0: [-1, 0], 1: [-1, -1], 2: [0, -1]}
        self.rowMatrixTemplate["tracebackMapping"] = {0: "top", 1: "diag", 2: "left"}

    def fillUpFirstRows(self, matrix):
        for i in range(len(matrix[0])):
            if (i == 1):
                matrix[0, i] =  self.gap
            if(i>1):
                matrix[0, i] = self.gap+self.egap*(i-1)

    def fillUpFirstRows2(self, matrix):
        for i in range(len(matrix[0])):
            matrix[0, i] = self.gap * (i)

    def fillUpFirstRows3(self, matrix):
        for i in range(len(matrix[0])):
                matrix[0, i] = self.gap*self.columns

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
            self.alignImagesBody(i)
        self.generateDisparity()

    def getTracebackStart(self, currentIndex):

        yMaxIndex = np.argmax(self.matrices[currentIndex]["scores"][:, self.columns])
        xMaxIndex = np.argmax(self.matrices[currentIndex]["scores"][self.columns, :])

        yMaxValue = self.matrices[currentIndex]["scores"][yMaxIndex, self.columns]
        xMaxValue = self.matrices[currentIndex]["scores"][self.columns, xMaxIndex]

        return (yMaxIndex, self.columns) if (yMaxValue > xMaxValue) else (self.columns, xMaxIndex)

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
            # message = "Traceback starging indices: %d %d"%(curX, curY)
            # warnings.warn(message)
            moves.append(self.matrices[currentIndex]["tracebackMapping"][previousMove])

        self.matrices[currentIndex]["tracebackPath"] = list(reversed(moves))

    def generateDisparity(self):
        try:
            scanlines = np.zeros(self.im1.shape)

            for index in range(len(self.matrices)):
                scanline = np.zeros(self.columns)

                lefts = 0
                tops = 0
                currentPixel = 0

                for direction in self.matrices[index]["tracebackPath"]:
                    if (direction == "left"):
                        lefts += 1
                    elif (direction == "top"):
                        tops += 1
                        scanline[currentPixel] = 0
                        currentPixel += 1
                    elif (direction == "diag"):
                        scanline[currentPixel] = np.abs(lefts - tops)  # + self.im1[index, int(currentPixel)]
                        currentPixel += 1
                    else:
                        print("Something is not right here!")
                        raise Exception
                scanlines[index] = scanline

            self.lastDisparity = np.asarray(scanlines)
        except(Exception):
            print("Unexpected error.")
            traceback.print_exc()

    ###################################################################
    # Functions to be overriden########################################
    ###################################################################

    @abc.abstractmethod
    def alignImagesBody(self, i):
      pass