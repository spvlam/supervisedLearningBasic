import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
def handle(x):
    return np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
x_bas = handle(X)
def signmoid(z):
    return 1/(1+np.exp(-z))
def logistic_gression(x,y,w_init,leaning_rate,count_loop,eps):
    count=0
    check_w_after = x.shape[0]
    w = [ w_init]
    while  count < count_loop : 
        for i in range(x.shape[0]):
            xi = x[i,:].reshape(1,x.shape[1])
            w_new = w[-1] + leaning_rate*xi *(y[i]-signmoid(np.dot(xi,w[-1].T )))
           
            count+=1
            if count % check_w_after ==0:
                if np.linalg.norm(w_new-w[-check_w_after]) < eps:
                    return w
            w.append(w_new)

    return w
w_int = np.random.randn(1,x_bas.shape[1])
a = logistic_gression(x_bas,y,w_int,0.05,10000,1e-4)
print(a[-1])

# plot
xo = X[y==0]
yo = y[y==0]
x1 =X[y==1]
y1= y[y==1]
xx = np.linspace(0,6,1000)
plt.plot(xo,yo,"ro")
plt.plot(x1,y1,"b^")
yy= signmoid(a[-1][0,0] + a[-1][0,1]*xx)
plt.plot(xx,yy)
plt.show()

            
    

