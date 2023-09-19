import numpy as np
import matplotlib.pyplot as plt
# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Visualize data v
plt.plot(X,y,"ro")
plt.axis([140,190,45,75])
plt.xlabel("height")
plt.ylabel("weight")

# the result of weight
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one,X),axis=1)
A=np.dot(Xbar.T,Xbar)
w=np.dot(np.linalg.pinv(A),np.dot(Xbar.T,y))
print(w.T)

from sklearn import linear_model
#  USING AVAILABLE MODEL
line_re = linear_model.LinearRegression(fit_intercept=False)
line_re.fit(Xbar,y)
print(line_re.coef_)