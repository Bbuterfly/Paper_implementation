import numpy as np

# 오차제곱합(Sum of Squares for Error, SEE)
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 교차 엔트로피 오차(Cross Entropy Error, CEE)
def cross_entropy_error(y, t, one_hot = False):
    delta = 1e-7 # log에 0입력 시 -inf를 방지하는 수
    
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    
    if one_hot:
        return -np.sum(np.log(y[np.arrange(batch_size), t] + delta)) / batch_size
    else:
        return -np.sum(t * np.log(y + delta)) / batch_size