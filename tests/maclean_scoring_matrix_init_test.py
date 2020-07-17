import numpy as np
def maclean_init(matrix, gap):
    for i in range(0,matrix.shape[0]):
        matrix[i:, i] = np.array([(i)*gap for i in range(i, matrix.shape[0])]).T
        matrix[i, i:] = np.array([(i)*gap for i in range(i, matrix.shape[0])])
        #return matrix
if __name__ == "__main__":
    matrix = np.zeros([10,10])
    gap = -20
    maclean_init(matrix, gap)
    print(matrix)