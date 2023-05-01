import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error





data = pd.read_csv('head.csv')
print(data.head(10))
data.shape
data.info()
data.describe()





# collecting X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values






# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# total number of values
n = len(X)

# using the formula to calculate b1 and b2
numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer/denom
b0 = mean_y - (b1 * mean_x)  

# print the coefficients
print(b0, b1)







max_x = np.max(X) + 100
min_x = np.min(X) - 100

x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

plt.plot(x, y, color = '#ef5423', label = 'Regression Line')
plt.scatter(X, Y, color = '#58b970', label = 'Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()







# using R squared
ss_t = 0
ss_r = 0
for i in range(n):
  y_pred = b0 + b1 * X[i]
  ss_t += (Y[i] - mean_y) ** 2
  ss_r += (Y[i] - y_pred) ** 2 

r2 = 1 - (ss_r/ss_t)
print(r2)






# using scikit learn
X  = X.reshape((n,1))
# creating model
reg = LinearRegression()
# fitting training data
reg = reg.fit(X, Y)
# Y prediction
y_pred = reg.predict(X)
mse = mean_squared_error(Y,y_pred)
rmse = np.sqrt(mse)
# calculating r score
r2_score = reg.score(X,Y)
print(rmse)
print(r2_score)