import numpy as np

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
# regression task -> identity function, classification -> softmax function
def softmax(x):
    c = np.max(x) # 오버플로우 방지, 일반적으로 입력 신호의 최댓값 사용
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x - c)
    y = exp_x / sum_exp_x
    
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))