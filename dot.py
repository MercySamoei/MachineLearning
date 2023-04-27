# This code exports a decision tree (binary) as a graphviz file named music-rec.dot using the export_graphviz() function from the tree module.
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

music = pd.read_csv('music.csv')
X = music.drop(columns = ['genre'])
y = music['genre']

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# joblib.load('music-recommender.joblib')

# predictions = model.predict([[22,1],[24,0]])
# predictions
tree.export_graphviz(model, out_file='music-rec.dot',
                     feature_names=['age','gender'],
                     class_names=sorted(y.unique()),
                     label='all',
                     filled=True,
                     rounded=True)