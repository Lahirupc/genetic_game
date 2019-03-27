import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import ga_game as ga

# # Generate dummy data
x_train = np.random.random((1000, 12))
y_train = keras.utils.to_categorical(np.random.randint(3, size=(1000, 1)), num_classes=3)
# x_test = np.random.random((100, 12))
# y_test = keras.utils.to_categorical(np.random.randint(3, size=(100, 1)), num_classes=3)


def init_ann(weights):
    w1 = np.reshape(weights[0][:168], (12,14))
    w2 = np.reshape(weights[0][168:], (14,3))
    
    model = Sequential()
    layer1 = Dense(14, activation='relu', weights=[w1,w1[0]], input_dim=12)
    model.add(layer1)
    output_layer = Dense(3, activation='softmax', weights=[w2,w2[0]])
    model.add(output_layer)
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              #optimizer=sgd,
              optimizer='rmsprop',
              metrics=['accuracy'])

    model.summary()
    return model

# Inputs should be given here.
def model_train(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=20, batch_size=64)

# def model_test():
#     score = model.evaluate(x_test, y_test, batch_size=128)

def model_predict(model, x):
    model.predict(x)

sample_population = []
sample_fitness = np.random.uniform(low=0.2, high=1.0, size=(10,))

for i in range(0, 10):
    sampl = np.random.uniform(low=0.2, high=1.0, size=(210,))
    sample_population.append(sampl)

weights = ga.createNewPopulation(sample_population, sample_fitness)
print(weights[0][:168])
weightz = np.reshape(weights[0][:168], (12,14))
print(np.shape(weightz))
print(np.shape(weightz[0]))
model = init_ann(weights)
model_train(x_train, y_train)


