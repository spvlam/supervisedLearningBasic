import numpy as np 
from basic_simulation import *
# randomly generate data 
N = 2 # number of training sample 
d = 2 # data dimension 
C = 3 # number of classes 

x = np.random.randn(d, N)
y = np.random.randint(0, 3, (N,))
y_out = conver_y_lable(y,C)

def soft_max(X,W):
    """
    return A (C,N)
    """
    return ((np.exp(np.dot(W.T,X))/np.sum(np.exp(np.dot(W.T,X)),axis=0)))
def cost(X,W,Y):
    A = soft_max(X,W)
    return -np.sum(np.log(A)*Y)
def SGD(X,W,Y):
    """
    return (C,C)
    """
    A = soft_max(X,W)
    return -np.dot(X,(Y-A).T)
def numerical_gradient(X,W,Y):
    eps = 1e-6
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w_copy1 = W.copy()
            w_copy2 = W.copy()
            w_copy1[i,j] -= eps
            w_copy2[i,j] += eps
            g[i,j] = (cost(X,w_copy2,Y)-cost(X,w_copy1,Y))/(2*eps)
    return g
W_init = np.random.randn(d, C)
g1 = SGD(x,W_init,y_out)
g2= numerical_gradient(x,W_init,y_out)
print(g1.shape,g2.shape,g1,g2)
print(np.linalg.norm(g1-g2))



