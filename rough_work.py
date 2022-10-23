from matplotlib.pyplot import axis
import numpy as np

matrix1 = np.array([1, 1, 1, 1])
matrix2 = np.array([2, 2, 2, 2])
matrix3 = np.array([3, 3, 3, 3])

arr = np.row_stack((matrix1, matrix2, matrix3))

arr_mean = np.mean(arr, axis=0)
print(arr_mean)