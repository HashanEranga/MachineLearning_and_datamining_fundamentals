# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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



