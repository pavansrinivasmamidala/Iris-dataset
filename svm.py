import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.svm import SVC
import tensorflow as tf


df=pd.read_csv('iris.csv')
x=df.ix[:,(0,1,2,3)].values
y=df.ix[:,(4)].values
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.20,random_state=415)


clf = SVC(kernel='linear')  
clf.fit(x_train, y_train)
accuracy=clf.score(x_test,y_test)
y_pred = clf.predict(x_test)
print("Predicted values:")
print(y_pred)
print(accuracy)
