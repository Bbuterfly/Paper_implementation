import numpy as np

# regression task -> identity function, classification -> softmax function
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    
    return y