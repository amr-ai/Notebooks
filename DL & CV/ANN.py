                                                #ANN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,datasets

# load data
(x_train_raw,y_train_raw),(x_test_raw,y_test_raw)=datasets.mnist.load_data()

y_train=keras.utils.to_categorical(y_train_raw)
y_test=keras.utils.to_categorical(y_test_raw)

x_train=x_train_raw.reshape(60000,-1)
x_test=x_test_raw.reshape(10000,-1)

x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255

num_classes=10
model=keras.Sequential()
model.add(layers.Dense(512, activation='relu',input_dim=784))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(124,activation='relu'))
model.add(layers.Dense(num_classes,activation='softmax'))
model.summary()

Optimizer=optimizers.Adam(0.001)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Optimizer, metrics=['accuracy'])


model.fit(x_train,y_train,batch_size=128,epochs=10,verbose=0)

score=model.evaluate(x_test,y_test)