import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split

# load data set 
iris = datasets.load_iris()
iris_data = iris.data
iris_target = iris.target
# devide into training and testing set
x_train, x_test, y_train, y_test = train_test_split(iris_data,iris_target,test_size=50)
k_nearest = neighbors.KNeighborsClassifier(n_neighbors=1,p=2,weights='distance').fit(x_train,y_train)
