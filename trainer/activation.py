import numpy as np

# regression task -> identity function, classification -> softmax function
def softmax(x):
    c = np.max(x) # 오버플로우 방지, 일반적으로 입력 신호의 최댓값 사용
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x - c)
    y = exp_x / sum_exp_x
    
    return y