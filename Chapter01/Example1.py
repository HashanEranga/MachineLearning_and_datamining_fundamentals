import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

data_root = "https://raw.githubusercontent.com/ageron/data/main/lifesat/lifesat.csv"
lifesat = pd.read_csv(data_root)
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat["Life satisfaction"].values

lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction", label="Country (each dot)")
plt.axis([23_500, 62_500, 4, 9])

for _, row in lifesat.iterrows():
    plt.annotate(row["Country"], (row["GDP per capita (USD)"], row["Life satisfaction"]),
                 textcoords="offset points", xytext=(4, 4), fontsize=8)

plt.legend()
plt.show()

# model = LinearRegression()
model = KNeighborsRegressor(n_neighbors=3)

model.fit(X, y)

X_new = [[37_655.2]]
print(model.predict(X_new))