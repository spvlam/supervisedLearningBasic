import numpy as np
import matplotlib.pyplot as plt

# generate 3 cluster of data
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
x0 = np.random.multivariate_normal(means[0], cov, N)
x1 = np.random.multivariate_normal(means[1], cov, N)
x2 = np.random.multivariate_normal(means[2], cov, N)

y = np.asarray([0]*N+[1]*N+[2]*N)

plt.scatter(x0[:,0],x0[:,1],c='b',marker="*")
plt.scatter(x1[:,0],x1[:,1],c='r',marker="+")
plt.scatter(x2[:,0],x2[:,1],c='g',marker=">")
# plt.show()


X_bas = np.concatenate((np.ones((3*N,1)),np.concatenate((x0,x1,x2),axis=0)),axis=1)
def conver_y_lable(y,c):
    """
    input : y =[0 1 2 0],c =3
    output : y_out =   
            [[1, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0]]
    """
    y_out = np.zeros((c,y.size))
    for i,j in enumerate(y):
        y_out[j,i] = 1
    return y_out

def soft_max(X,W):
    """
    input X (D,N) , W(D,C)
       
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
def softmax_regression(X,Y,w_ini,eps,learning_rate,count_max,size_batch):
    w = [w_ini]
   
    count=0
    d=X.shape[0]
    c=Y.shape[0]
    N1 = X.shape[1]
    while count < count_max :
        for i in range(X.shape[1]):
            x_i = X[:,i].reshape(d,1)
            y_i = Y[:,i].reshape(c,1)
            a_i = soft_max(x_i,w[-1])
            w_new = w[-1] + np.dot(x_i,(y_i-a_i).T)*learning_rate
            count+=1
            if count % size_batch ==0:
                if np.linalg.norm(w_new-w[-1]) < eps:
                    return w[-1]
            w.append(w_new)
    return w[-1]
            
w_init = np.random.randn(3,3)
w = softmax_regression(X_bas.T,conver_y_lable(y,3),w_init,1e-6,0.05,10000,20)


