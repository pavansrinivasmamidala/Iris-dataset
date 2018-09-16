import pandas as pd
import numpy as np
from sklearn import model_selection,tree
import tensorflow as tf


df=pd.read_csv('iris.csv')
x=df.ix[:,(0,1,2,3)].values
y=df.ix[:,(4)].values


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.20,random_state=415)


#decisiontree

# Creating the classifier object
clf_gini = tree.DecisionTreeClassifier(criterion = "gini",max_depth=4, min_samples_leaf=1)

# Performing training
clf_gini.fit(x_train, y_train)

clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy",max_depth = 4, min_samples_leaf = 1)

# Performing training
clf_entropy.fit(x_train, y_train)
accuracy=clf_entropy.score(x_test,y_test)
y_pred = clf_entropy.predict(x_test)
print("Predicted values:")
print(y_pred)
print(accuracy)
