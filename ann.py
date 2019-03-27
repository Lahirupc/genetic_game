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
    model = Sequential()
    layer1 = Dense(14, activation='relu', weights=[weights[1:],weights[0]], input_dim=12)
    
    #weights_layer1 = np.reshape(weights[:168],(12,14))
    #layer1.set_weights(weights_layer1)
    model.add(layer1)
    #model.add(Dropout(0.5))
    output_layer = Dense(3, activation='softmax')
    # weights_output_layer = weights[168:]
    # output_layer.set_weights(weights_output_layer)
    model.add(output_layer)
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              #optimizer=sgd,
              optimizer='rmsprop',
              metrics=['accuracy'])

    print(np.shape(layer1.get_weights()))
    print(np.shape(np.reshape(weights[1:169],(12,14))))
    model.summary()
    return model


# Inputs should be given here.
def model_train(x_train, y_train):
    model.fit(x_train, y_train, epochs=20, batch_size=64)

# def model_test():
#     score = model.evaluate(x_test, y_test, batch_size=128)

def model_predict(x_data_point):    
    prediction = model.predict(x_data_point)
    print(prediction)

sample_population = []
sample_fitness = np.random.uniform(low=0.2, high=1.0, size=(10,))

for i in range(0, 10):
    sampl = np.random.uniform(low=0.2, high=1.0, size=(210,))
    sample_population.append(sampl)

weights = ga.createNewPopulation(sample_population, sample_fitness)

model = init_ann(np.reshape(weights[0][:169], (12,14)))

model_train(x_train, y_train)


