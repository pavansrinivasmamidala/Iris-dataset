import pandas as pd
import numpy as np
from sklearn import neighbors,model_selection,preprocessing,tree
import tensorflow as tf


df=pd.read_csv('iris.csv')
df=pd.get_dummies(df,columns=['label'])
values=list(df.columns.values)
y = df[values[-3:]]
y = np.array(y, dtype='float32')
X = df[values[0:-3]]
X = np.array(X, dtype='float32')

x_train,x_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.20,random_state=415)

#neural network
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

n_nodes_hl1=40
n_nodes_hl2=40
n_nodes_hl3=40

n_input=4
n_classes=3
training_epochs=500

x=tf.placeholder('float',[None,n_input])
y=tf.placeholder('float',[None,n_classes])

def multilayerperceptron(x):
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([n_input,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1=tf.add(tf.matmul(x,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1=tf.nn.relu(l1)

    l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2=tf.nn.relu(l2)

    l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3=tf.nn.relu(l3)

    out=tf.add(tf.matmul(l3,output_layer['weights']),output_layer['biases'])
    return out
predictions=multilayerperceptron(x)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions,labels=y))
optimizer=tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost=0.0
        _,c=sess.run([optimizer,cost],feed_dict={x:x_train,y:y_train})
        avg_cost+=c
        print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))
    sess.close()
