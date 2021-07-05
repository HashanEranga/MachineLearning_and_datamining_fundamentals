# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([2,5,7,9,11,14,15,17,19])

n = np.size(x)

mx = np.mean(x)
my = np.mean(y)

ssxy = np.sum(y*x)
ssxx = np.sum(x*x)

b1 = ssxy/ssxx
b0 = my - b1*mx
 
plt.scatter(x,y, color = 'b', marker = "*", s = 60)
plt.title('Simple Linear Regression')
plt.xlabel('Independant Variable')
plt.ylabel('Dependant Variable')

y_pred = b0 + b1*x

plt.plot(x, y_pred, color = 'r')
plt.show()


# %%
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# %%
wine_dataset = datasets.load_wine()
x = wine_dataset['data']
y = wine_dataset['target']


# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)


# %%
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)


# %%
predictions = log_reg.predict(x_test)


# %%
print(accuracy_score(y_test, predictions))


# %%



