import numpy  as np
from mnist import MNIST
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import accuracy_score
# load all training and testing data

mntrain = MNIST("/home/lamnv/Documents/my carier/Marchine learning/supervise learning/dataMNIST/MNIST")
mntrain.load_training()
xtrain_all = np.asarray(mntrain.train_images)
ytrain_all = np.array(mntrain.train_labels.tolist())

mntest = MNIST("/home/lamnv/Documents/my carier/Marchine learning/supervise learning/dataMNIST/MNIST")
mntest.load_testing()
xtest_all = np.asarray(mntest.test_images)
ytest_all = np.array(mntest.test_labels.tolist())
def extract_feature(x,y,clas):
    y_index = np.array([])
    y_res = []
    for i in clas:
        for j in i:
            y_index = np.concatenate((y_index,np.where(y==j)[0]),axis=0)
            y_res.append(j)
    y_res = np.asarray(y_res)
    x_res = x[y_index.astype(int),:] # to convert into range of 0 and 1
    return (x_res,y_res)

clas =[[0],[1]]
(x_train,y_train) = extract_feature(xtrain_all,ytrain_all,clas)
(x_test, y_test) = extract_feature(xtest_all, ytest_all, clas)
logres = linear_model.LogisticRegression().fit(x_train,y_train)
# # y_red = logres.predict(x_test)
# print(y_red)
# print(type(y_red))
# print(f"accurancy {accuracy_score(y_red.tolist(),y_test)}")


