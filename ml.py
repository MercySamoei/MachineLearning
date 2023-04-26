import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


music = pd.read_csv('music.csv')
X = music.drop(columns = 'genre')
y = music['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
prediction

score = accuracy_score(y_test, prediction)
score