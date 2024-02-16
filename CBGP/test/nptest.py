import numpy as np 




error_matrix = np.stack([np.array([1,2,3]), np.array([2,2,2]), np.array([3,4,5])])
print(error_matrix)
weights = np.array([1,1,2])
print("error x weights")
print(error_matrix*weights)
print("sum")
print((error_matrix*weights).sum(axis=1))
print("argmin")
print((error_matrix*weights).sum(axis=1).argmin())
elite = (error_matrix*weights).sum(axis=1).argmin()