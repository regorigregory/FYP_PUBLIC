# This algo is buggy. Only works with short sequences, otherwise it returns confusing results.
# Written months ago, will fix later on.

import numpy as np
def getOne(*argv):
    return NWMatcher(*argv)

class NWMatcher():

    def __init__(self, match, mu, delta):

        # mu = mismatch
        # delta = gap
        self.match = match
        self.mismatch = mu
        self.gap = delta

    def matchSequence(self, a, b):
        self.init_sequence(a, b)
        self.populate_matrix()
        return self.get_best_match()

    def init_sequence(self, seq1, seq2):

        self.columns = len(seq1) + 1
        self.rows = len(seq2) + 1

        self.seq1 = seq1
        self.seq2 = seq2

        self.matrix = np.zeros([self.rows, self.columns])
        self.moves = np.empty([self.rows, self.columns], dtype=str)
        self.moves[0, 0] = "o"
        penalty = 0

        for i in range(1, self.rows):
            penalty += self.gap
            self.matrix[i, 0] = penalty
            self.moves[i, 0] = "south"
        penalty = 0

        for i in range(1, self.columns):
            penalty += self.gap
            self.matrix[0, i] = penalty
            self.moves[0, i] = "east"
        self.greater = self.columns if (self.columns > self.rows) else self.rows

    # possible moves in a matrix[i,j]:
    # down [i+1]
    # right=[j+1]
    # diag = [i+i, j+1]

    def move(self, i, j):
        target = self.matrix[i, j]
        # moving from starting points to target

        down = self.matrix[i - 1, j] + self.gap
        right = self.matrix[i, j - 1] + self.gap

        diag = self.matrix[i - 1, j - 1]

        if self.seq1[j - 1] == self.seq2[i - 1]:
            diag += self.match
        else:
            diag += self.mismatch
        move_dict = {"south": down, "east": right, "diag": diag}

        return max(move_dict, key=move_dict.get), max([down, right, diag])

    def populate_matrix(self):
        for i in range(1, self.rows):
            for j in range(1, self.columns):
                # here you need to fix the non existent index issue!
                self.moves[i, j], self.matrix[i, j] = self.move(i, j)

    def get_best_match(self):

        seq_1_out = ""
        seq_2_out = ""

        # Finding the last row and column

        last_row = self.rows - 1
        last_column = self.columns - 1

        # Find the cell with the maximum value

        # in the last row
        column_max_index = np.argmax(self.matrix[last_row, :])
        # in the last column
        row_max_index = np.argmax(self.matrix[:, last_column])

        # Finding the greater one amongs the two retrieved ones

        x_max = self.matrix[last_row, column_max_index]
        y_max = self.matrix[row_max_index, last_column]

        # determining the starting cell for the traceback

        coords = [last_row, column_max_index] if (x_max > y_max) else [row_max_index, last_column]

        # down: south horizontal gap = gap in the first string
        # right: east: vertical gap = gap in the second string
        # diagonal:diag:  match/mismatch in the current characters

        seq_1_cur = len(self.seq1) - 1
        seq_2_cur = len(self.seq2) - 1

        seq_1_next_char = self.seq1[seq_1_cur]
        seq_2_next_char = self.seq2[seq_2_cur]
        cx = coords[1]
        cy = coords[0]

        while ((seq_2_cur and seq_1_cur) > -1):

            origin = self.moves[cy, cx]

            if origin == "s":
                seq_1_out = "_" + seq_1_out
                seq_2_out = seq_2_next_char + seq_2_out

                seq_2_cur -= 1
                cy -= 1

            elif origin == "e":
                seq_1_out = seq_1_next_char + seq_1_out
                seq_2_out = "_" + seq_2_out

                seq_1_cur -= 1
                cx -= 1

            else:
                seq_1_out = seq_1_next_char + seq_1_out
                seq_2_out = seq_2_next_char + seq_2_out

                seq_1_cur -= 1
                seq_2_cur -= 1

                cy -= 1
                cx -= 1

            seq_1_next_char = self.seq1[seq_1_cur]
            seq_2_next_char = self.seq2[seq_2_cur]
        return seq_1_out, seq_2_out
