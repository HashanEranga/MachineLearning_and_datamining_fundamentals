# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Creation

# %%
#importing module
import numpy as np


# %%
#creating 1D array
array1 = np.array([1,2,3])


# %%
#returning the datatype of the array
array1.dtype


# %%
#
matrix = np.array([np.arange(3), [i for i in range(1,4)], [6,7,8]])
print(matrix)

# %% [markdown]
# # Initialization

# %%
#define an array of zeros having the datatype float
np.zeros((5,2,2), dtype=float)


# %%
#define an array full of ones
np.ones((4,5), dtype=int)


# %%
#creating an array of 3x4 matrix
#having random data
np.empty([3,4])


# %%
#arrange with evenly spaced values
np.arange(2,10,2)
#np.arange(start_val, end_val, period)


# %%
#rearrangeing the size of the array
np.arange(2,10,1).reshape(4,2)


# %%
#creating an array with constants 2x3 matrix including the value 4
np.full([2,3], 4)


# %%
#identity matrix created 3x3
np.eye(3)


# %%
np.identity(3)


# %%
#creating an array 5 splits from 2 to 3 including those values
np.linspace(2,3,5)

# %% [markdown]
# # Copying, Sorting, Shifting

# %%
#Return the copy of the defined matrix called matrix
np.copy(matrix)


# %%
#matrix deep copy function
matrix.copy()


# %%
#shallow copy 
matrix.view()


# %%
#sorting values in assending order
matrix.sort()
print(matrix)


# %%
#sort values along the specified axis
matrix.sort(axis=0)


# %%
matrix.sort(axis=1)


# %%
mat = np.array( [[1,2,3], [3,4,5], [7,4,2], [9,6,4]])
print(mat)


# %%
#values are sorted along the column wise
mat.sort(axis=0)
mat


# %%
#values are sorting along the row
mat.sort(axis=1)
mat


# %%
#indexing starting from 0 
matrix[0:,:1]


# %%
matrix


# %%
matrix[:1, :]


# %%
matrix[:2, 0:2]


# %%
matrix[:1, :]

# %% [markdown]
# # Try Out

# %%
#change the specific value in the array
matrix[0,0] = 45
matrix


# %%



# %%
#print the 1st raw first value
matrix[0,0]


# %%
#making changes to all the values in the first raw
matrix[0] = 42
matrix


# %%
#interpret all the values from the 1st raw
matrix[1:]


# %%
#show values 
matrix[0:2]


# %%
matrix[:]


# %%
matrix[:1, :2]


# %%
matrix[1: , 1:]


# %%
matrix[:2, 1:]


# %%
#using for flatten the arrays like 2 dimetional and multi dimentional
matrix.ravel()


# %%
matrix[:, 1].copy()


# %%
matrix[0].tolist()


# %%
matrix.reshape(9)


# %%
#matrix multiplication
matA = np.array([[2,4,7],[5,3,9], [2,1,7]])
matB = np.random.randint(-3,3,(3,7))


# %%
matC = np.matmul(matA, matB)
matC


# %%
import time
start_time = time.time()
matD = np.matmul(matA, matB)
end_time = time.time()
duration = start_time-end_time

print('Duration =', duration)


# %%
print(matD)
