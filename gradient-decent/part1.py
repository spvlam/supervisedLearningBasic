import numpy as np
import matplotlib.pyplot as plt
# data set
x = np.random.rand(1000, 1)
y = 4*x+3 + 0.2*np.random.randn(1000,1)
# using formular
x_input = np.concatenate((np.ones_like(x),x),axis=1)
b = np.dot(x_input.T,y)
w= np.dot(np.linalg.pinv(np.dot(x_input.T,x_input)),b)
x_0 = np.linspace(0,2)
y_0 = w[0] + w[1]*x_0
# print(w.shape,w)
# using gradient method
def gradient1(w_init):
    return np.dot(x_input.T,-y+np.dot(x_input,w_init))/x.shape[0]
def cost(w_real):
    return np.linalg.norm(y-np.dot(x_input,w_real),2)/2000
def numerical_gradient(w):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps 
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n))/(2*eps)
    return g 
def my_GD(leaning_rate):
    w_in = [np.array([[2], [1]],dtype=float)]
    while np.linalg.norm(numerical_gradient(w_in[-1]) )> 1e-3:
        w_new = w_in[-1]-leaning_rate*numerical_gradient(w_in[-1])
        w_in.append(w_new)
    return w_in[-1]
print(my_GD(1))



