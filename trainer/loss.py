import numpy as np

# 오차제곱합(Sum of Squares for Error, SEE)
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)