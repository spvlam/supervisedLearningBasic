import numpy as np
w_in = [np.array([[2], [1]])]
w_in[-1][1] += 1e-1
print(w_in[-1][1])