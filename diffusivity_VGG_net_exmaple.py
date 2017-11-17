##### implement VGG net in keras
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, adam
import h5py

## generate dummy data

#x_train = np.random.random((100, 100, 100, 3))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
#x_test = np.random.random((20, 100, 100, 3))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

# import training and test data
diffu_keras = h5py.File('diffusivity_keras.h5py', 'r')
x_train = diffu_keras["X_train_reshaped"][:]
y_train = diffu_keras["Y_train_reshaped"][:]
x_test = diffu_keras["X_test_reshaped"][:]
y_test = diffu_keras["Y_test_reshaped"][:]
print("shapes of train and test data: {}, {}, {}, {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

model = Sequential()
# input: (100, 100, 3) images as input tensor
model.add(Conv2D(16, (3, 3), activation='relu', input_shape= (100, 100, 1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

### for regression process
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr= 0.0316, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

#adam = adam(lr = 0.01)
#model.compile(loss='mean_squared_error', optimizer=adam)


model.fit(x_train, y_train, batch_size=256, epochs=50)

y_prediction = model.predict(x_test, batch_size=128, verbose=0)
error = np.mean(np.abs((y_prediction - y_test) / y_test))

print("test error of convolutional neural network: {}%".format((1 - error) * 100))