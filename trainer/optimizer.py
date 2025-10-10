import numpy as np

# 중앙 차분식 수치미분법 -> f'(x) = f(x+h) - f(x-h) / 2h
# 어차피 backward를 통해서 grad를 계산하기 때문에 사용하지 않음
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / 2*h
        x[idx] = tmp_val
        
        it.iternext()
        
    return grad
    
def gradient_descent(f, init_x, lr=1e-2, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x