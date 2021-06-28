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

# %% [markdown]
# # Pandas

# %%
#import pandas library
import pandas as pd


# %%
#creating pandas dictionaries using series
a = pd.Series([1,4,-2,'string'], index=['a','b','c','d'])
a


# %%
#creating a dataframe using pandas
data = {'polulation' : [12,24,56,234,78,12], 'animals': ['dog', 'cat', 'wold', 'parrot', 'fox', 'snake']}
data
#passing the dataset to the dataframe
df = pd.DataFrame(data, index=[1,2,3,4,5,6])
df


# %%
dataset = pd.read_csv('Dataset/sampleDataset.csv')
dataset


# %%
dataset.isnull()


# %%
# Create a Series with a List
s = pd.Series([1,4,-2,'home'], index=['a','b','c','d'])

# %% [markdown]
# # TODO 1 : What is the type of s ? Can it be changed ?
# 
# The data type of the s Serias can be viewed as follows (Refer code 1 below). The type of these are object type. If all the data in the same type then those data can can be converted into a particular data type.
# The data type of a series can be changed according to the code given below (Refer code 2 below)
# 

# %%
# code 1
print(type(s))
print(pd.api.types.is_numeric_dtype(s))


# %%
# code 2
dataSeries = pd.Series(['1','2', '3','4'], index=['a','b','c','d'])
print(dataSeries)
convertedDataSeries = dataSeries.astype('int32')
print(convertedDataSeries)
print(convertedDataSeries.dtype)

# %% [markdown]
# # Accessing and Modyfying
# 

# %%
# call all the data in the range 1 to 3 except 3rd index
s[1:3]

# Call the first index in the data series
s[0]

# find the data which have the index value 'd'
s['d']

# return all the values are stored in the data series from 2
s.values[2:]

# %% [markdown]
# # Creating a data frame using a dictionary

# %%
data = {'population' : [1.5, 1.2, 2.0, 1.4, 0.8], 'state' : ['Nevada', 'Florida', 'Ohio', 'Texas', 'Florida'], 'year' : [2003, 2000, 2004, 1990, 1994]}

df = pd.DataFrame(data, index=['one', 'two', 'three', 'four', 'five'], columns=['year', 'state', 'population', 'debt'])


# %%
# show in a table view the population and state
df[['population', 'state']]


# %%
# shows the population of the dataframe
df.population


# %%
# show 1st index of the data to the end
df.iloc[1:]


# %%
# show a chosen set from the data frame 
df.iloc[2:4:, 2:5]


# %%
# give all the attribute values of the record indexed as 'one'
df.loc['one']


# %%
# Assign all the debt column value 34.67
df.debt = 34.67
print(df)


# %%
df.debt = [df.iloc[:,2][i]*5 for i in range(0, df.shape[0])]
print(df)


# %%
# give details about 5 data if just the head was asked
df.head


# %%
# only show some sample 3 records
df.head(3)


# %%
# represent the last two records of the data
df.tail(2) 


# %%
# show sample recods 
df.sample(n=3)


# %%
# adding a new column to the data frame and assign random values
df['newColumn'] = pd.Series(np.random.randn(df.shape[0]), index= df.index)
print(df)


# %%
# remove the duplicate values presence in the state column
df.drop_duplicates('state')


# %%
df.state

# %% [markdown]
# # Loading Data From CSV Files

# %%
df = pd.read_csv('Dataset\sampleDataSet.csv')
df.head(5)

# %% [markdown]
# # TODO 2 : Comment on the shape of the data frame with and without setting names
# 
# %% [markdown]
# ## Dealing with the missing values
# 

# %%
# check for missing data values in the given dataset
df.isnull()


# %%
df.isnull().sum


# %%
# remove all the null data
df = df[df.isnull() != True]
print(df)


# %%
df.dropna(axis=0).isnull().sum()


# %%
df.dropna(axis=1)


# %%
df.dropna(axis=1, how='all')


# %%
df.dropna(axis=1, thresh=1)


# %%
# df.drop('i', axis=1)


# %%
df.fillna(899)


# %%
df.fillna(method='ffill')


# %%
df.replace(6.3,600)


# %%
df.replace('.',np.nan)


# %%
df[np.random.randint(df.shape[0] > 0.5)] = 1.5
print(df)

# %% [markdown]
# ## Apply functions can be written using lambda expression or using ordinary function definition

# %%
f = lambda df: df.max()-df.min()
def f(x):
    return x.max()-x.min()
df.iloc[:, 3:5].apply(f)

# %% [markdown]
# ## Group Operations
# 
# 

# %%
dataRequired = {'Names' : [1, 2, 3, 4, 1], 'Team' : ['A', 'B', 'C', 'D', 'E']}
df = pd.DataFrame(dataRequired, index=['a','b','c','d','e'])
grouped = df[['Names']].groupby(df['Team'])
grouped.mean()


# %%
grouped.mean()


# %%
grouped = df[['Names']].groupby(df['Team']).mean()


# %%
grouped.unstack()

# %% [markdown]
# ## Data Summarizing
# 

# %%
df['Names'].nunique()


# %%
df['Names'].value_counts()


# %%
df.describe


# %%
df.mean


# %%
df.sort_index().head()

# %% [markdown]
# ## Data Visualization

# %%
df.plot(kind='hist')


# %%
df.plot(kind='bar')


# %%
df.boxplot()

# %% [markdown]
# # Try Out

# %%
import pandas as pd
import numpy as np

# import the data set and assign column values given below
dataSet = pd.read_csv('DataSet\Lab01Exercise01.csv', names=['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5'])

# making sure the data columns are assigned properly
dataSet.head(2)


# %%
dataSet.mean()


# %%
# replace all the Nan and NaN values using the mean of each columns
filledDataSet = dataSet.fillna(dataSet.mean())


# %%
from pandas.plotting import scatter_matrix
scatter_matrix(filledDataSet, alpha=0.2, figsize=(6,6), diagonal='kde')


# %%
#developed the condition required
condition = [((filledDataSet['Channel1'] + filledDataSet['Channel5'])/2) < ((filledDataSet['Channel2'] + filledDataSet['Channel3'] + filledDataSet['Channel4'])/3), ((filledDataSet['Channel1'] + filledDataSet['Channel5'])/2) >= ((filledDataSet['Channel2'] + filledDataSet['Channel3'] + filledDataSet['Channel4'])/3)]

# assign the values if the answer is true <= 1 else False <= 0 
value = [1,0]

# assigned the values for each of the corresponding 
filledDataSet['class'] = np.select(condition, value)
filledDataSet.describe


# %%
filledDataSet.describe

# %% [markdown]
# # Try Out Random Walk
# 

# %%
import numpy as np
# efine Random walk Class
class RandomWalk:
    # create the constructor 
    def __init__(self, position):
        self.position = position

    # walking method defined
    def RandWalk(self):
        path_taken = [] # Empty list for records the path taken
        path_taken.append(self.position)

        # 500 steps walk
        for i in range(500):
            # generate a random integer
            random_input = np.random.randint(0,2)
            # increase the step if input == 1 otherwise decrease the step
            if random_input == 1:
                self.position += 1
            else:
                self.position -= 1
            path_taken.append(self.position)
        # return the taken path
        return path_taken

# create the object 
man1 = RandomWalk(100).RandWalk()

print(man1)
     


# %%
from matplotlib import pyplot as plt
# i dimetional random walk in a plot
plt.plot(man1)


# %%



