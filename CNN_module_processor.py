import pickle
import time
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import CNN_classifier as CNN


NAME = "chungus-CNN-64x3-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

pickle_in = open("X_train.pickle", "rb")
CNN.X_train = pickle.load(pickle_in)
CNN.X_train = pickle.load(open("X_train.pickle", "rb"))
CNN.y = pickle.load(open("y.pickle", "rb"))

CNN.X_train = CNN.X_train / 255.0

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=CNN.X_train.shape[1:]))  # throws in the input shape
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

model.fit(CNN.X_train, CNN.y, batch_size=32, epochs=3, validation_split=0.05)

# create function that occludes the images to increase the data set diversity and size

