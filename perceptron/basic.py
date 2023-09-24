import numpy as np
import matplotlib.pyplot as plt
# generate data
means = [[2,2],[4,2]]
cov =[[0.3,0.2],[0.2,0.3]]
N=10
x0 = np.random.multivariate_normal(means[0],cov,N).T
x1 = np.random.multivariate_normal(means[1],cov,N).T
X= np.concatenate((x0,x1),axis=1)
# y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
y = np.array([1]*N+[-1]*N)
x_bas = np.concatenate((np.array([[1]*2*N]),X),axis=0)
# X2 = np.concatenate((np.ones((1, 2*N)), X), axis = 0)
#Perceptron learning 
def h_sign(x,w):
    return np.sign(np.dot(w.T,x))
def convergence(y1,y2):
    return np.array_equal(y1,y2)
def PLA(x_in,y_in,lea,w_ini):
    w_init=[w_ini]
    D=x_in.shape[0]
    while True:
        for i in range(2*N):
            x_i =  x_in[:,i].reshape(D,1)
            y_i = y_in[i]
            if h_sign(x_i,w_init[-1])[0]!=y_i:
                w_new = w_init[-1] + lea*np.dot(x_i,y_i)
                w_init.append(w_new)
        if convergence(h_sign(x_in,w_init[-1]).reshape(20),y_in):
            break
    return w_init[-1]
w_init = np.random.randn(3, 1)
print(w_init)
result = PLA(x_bas,y,1,w_init)

# plot the result

x3 = np.linspace(0,10)

y3 = (result[0]+result[1]*x3)/(-result[2])

plt.scatter(x0[0],x0[1],c='b',marker='o')
plt.scatter(x1[0],x1[1],c='r',marker='x')
plt.plot(x3,y3)
plt.show()






