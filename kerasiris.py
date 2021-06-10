import tensorflow.compat.v1 as tf #symbolic math library used for machine learning
tf.disable_v2_behavior()
import pandas as pd #data manipulation and analysis
import sklearn.model_selection #machine learning library for python
import numpy as np #converts things into arrays easier to understand
from tensorflow import keras
df=pd.read_csv('/Users/sasankgamini/Desktop/iris.csv')
df.replace('?',-99999, inplace=True)
inputs=df.drop('class',1) #columns is 1 and 0 is rows
inputs=np.array(inputs) #using numpy so it arranges the data so it's easier to read
outputs=df['class']
outputs=np.array(outputs)
labels=['iris-setosa','iris-versicolor','iris-virginica']  #labels means Y
 
Xtraindata, Xtestdata, Ytraindata, Ytestdata = sklearn.model_selection.train_test_split(inputs,outputs, test_size = 0.2) #splitting into testing and training, need more training so we chose test size to be 20%
##print(Ytraindata)
##print('after conversion')
Ytraindata[Ytraindata=='Iris-setosa']=0
Ytraindata[Ytraindata=='Iris-versicolor']=1
Ytraindata[Ytraindata=='Iris-virginica']=2
Ytestdata[Ytestdata=='Iris-setosa']=0
Ytestdata[Ytestdata=='Iris-versicolor']=1
Ytestdata[Ytestdata=='Iris-virginica']=2
##print(Ytraindata)
##print(Xtraindata)
##print(Ytraindata)
##print(Xtestdata)
##print(Ytestdata)
##print(Xtraindata.shape)
##print(Xtestdata.shape)
##print(Xtestdata)
##print(Ytestdata)
##print(Ytestdata.shape)

#building the model
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(4,)),
    keras.layers.Dense(130,activation='relu'),
    keras.layers.Dense(3,activation='softmax')
    ])

#Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  #sparse means low
              metrics=['accuracy']  #metric to make sure there is progress
              )
#Train the model
model.fit(Xtraindata,Ytraindata,epochs=50)

#Test the model
test_loss, test_acc = model.evaluate(Xtestdata, Ytestdata)
print(test_acc)


              
              









