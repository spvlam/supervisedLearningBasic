import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

# load data set 
iris = datasets.load_iris()
iris_data = iris.data
iris_target = iris.target
# devide into training and testing set
print(iris_data[iris_target==0,:2])