import numpy as np

# a = np.array([])
# b=[1,2]
# c = np.where(b==1)
# print(a.ndim,)

import imageio
imagine = imageio.imread("/home/lamnv/Documents/my carier/Marchine learning/supervise learning/girl3.jpg")
# print(imagine)
# print(type(imagine))
# print(imagine.shape)
ex = np.array([[[1,2,3],[2,3,4]],[[3,2,3],[9,3,4]]])
print(ex[:,:,0])