                                        #CNN
from tensorflow.keras import optimizers, datasets, layers
import tensorflow as tf


((train_x, train_y),(test_x, test_y)) = datasets.mnist.load_data()

train_x=train_x.reshape(60000,28,28,1)
test_x=test_x.reshape(10000,28,28,1)

train_x=train_x.astype('float32')/255
test_x=test_x.astype('float32')/255

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

model1 = tf.keras.Sequential()
model1.add(layers.Conv2D(filters= 32,kernel_size= 5, strides= 1, padding= 'same',
    activation= 'relu', input_shape= (28,28,1)))
model1.add(layers.MaxPool2D(pool_size=(2,2), strides= (2,2), padding= 'valid'))
model1.add(layers.Conv2D(filters= 64,kernel_size= 3, strides= (1,1), padding= 'same',
    activation= 'relu'))
model1.add(layers.MaxPool2D(pool_size=(2,2), strides= (2,2), padding= 'valid'))
model1.add(layers.Dropout(0.25))
model1.add(layers.Flatten())
model1.add(layers.Dense(units= 128, activation= tf.nn.relu))
model1.add(layers.Dropout(0.5))
model1.add(layers.Dense(units= 10, activation= tf.nn.softmax))


model1.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])
model1.fit(x= train_x, y= train_y, validation_data = (test_x, test_y),epochs= 10, batch_size= 128)