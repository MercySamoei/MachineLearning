import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = sns.load_dataset('iris')
iris.head()

sns.pairplot(iris, hue='species');

# extract the features matrix and target array from the DataFrame.
X_iris = iris.drop('species', axis=1)
X_iris.shape

# this will give us only the specific matrix 
y_iris = iris['species']
y_iris.shape

X = iris.data
y = iris.target

# choose the model and set the hyperparameters for the model.
model = KNeighborsClassifier(n_neighbors=1)

# train the model, and use it to predict labels for data we already know:
model.fit(X, y)
y_model = model.predict(X)

# find the accuracy score: how correct is the model
accuracy_score(y, y_model)

# using a holdout set to check the model performance. 
# The set divides data, trains it and tests its accuracy using data that has not been exposed to the model
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)

model.fit(X1, y1)

y2_model = model.predict(X2)
accuracy_score(y2, y2_model)