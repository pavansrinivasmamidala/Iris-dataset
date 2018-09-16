import pandas as pd
import numpy as np
from sklearn import neighbors,model_selection


df=pd.read_csv('iris.csv')
x=df.ix[:,(0,1,2,3)].values
y=df.ix[:,(4)].values


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.20,random_state=415)

#knn

c=neighbors.KNeighborsClassifier()
c.fit(x_train,y_train)
accuracy=c.score(x_test,y_test)
y_r=c.predict(x_test)
print("accuracy=",accuracy)
print(y_r)
