import numpy  as np
from mnist import MNIST
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from display_network import *
# load all training and testing data

mntrain = MNIST("/home/lamnv/Documents/my carier/Marchine learning/supervise learning/data/MNIST")
mntrain.load_training()
xtrain_all = np.asarray(mntrain.train_images)
ytrain_all = np.array(mntrain.train_labels.tolist())

mntest = MNIST("/home/lamnv/Documents/my carier/Marchine learning/supervise learning/data/MNIST")
mntest.load_testing()
xtest_all = np.asarray(mntest.test_images)
ytest_all = np.array(mntest.test_labels.tolist())
# the main purpose of extract feature is to get the nescessary feature
def extract_feature(x,y,clas):
    y_index = np.array([])
    y_res = []
    len1=0
    for i in clas:
        for j in i:
            y_index = np.concatenate((y_index,np.where(y==j)[0]),axis=0)
            for k in range(len(y_index)-len1):
                y_res.append(j)
            len1=len(y_index)
            print(len(y_index))
    y_res = np.asarray(y_res)
    x_res = x[y_index.astype(int),:] # to convert into range of 0 and 1
    return (x_res,y_res)

clas =[[0],[1]]
(x_train,y_train) = extract_feature(xtrain_all,ytrain_all,clas)
(x_test, y_test) = extract_feature(xtest_all, ytest_all, clas)
logres = linear_model.LogisticRegression().fit(x_train,y_train)
y_red = logres.predict(x_test)
print(f"accurancy {accuracy_score(y_red.tolist(),y_test)}")

mis_id = np.where((y_red-y_test)!=0)[0]
x_mis = x_test[mis_id,:]
plt.axis('off')
A = display_network(x_mis.T)
f2 = plt.imshow(A, interpolation='nearest' )
plt.gray()
plt.show()
