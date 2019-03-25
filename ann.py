import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
x_train = np.random.random((1000, 12))
y_train = keras.utils.to_categorical(np.random.randint(3, size=(1000, 1)), num_classes=3)
x_test = np.random.random((100, 12))
y_test = keras.utils.to_categorical(np.random.randint(3, size=(100, 1)), num_classes=3)

# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=12))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              #optimizer=sgd,
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=200,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)

