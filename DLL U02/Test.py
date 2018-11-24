import logging

logging.basicConfig(level=logging.WARNING)

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import RandomSearch

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import argparse

# <JAB>

import ConfigSpace.hyperparameters as CSH

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.datasets import mnist

from cnn_mnist import mnist

from hpbandster.core.result import json_result_logger as jlog




x_train, y_train, x_valid, y_valid, x_test, y_test = mnist("./")



#{'learning_rate': 0.3454628959392778, 'num_filters': 10, 'batch_size': 41, 'filter_size': 3}

lr =  0.3454628959392778
num_filters = 10
batch_size = 41
filter_size = 3
epochs = 12

model = Sequential()
model.add(Conv2D(num_filters,
                 kernel_size=filter_size,
                 activation='relu',
                 input_shape=(28, 28, 1),
                 padding='same'
                 ))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(num_filters,
                 kernel_size=filter_size,
                 activation='relu',
                 padding='same'
                 ))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

optimizer = SGD(lr=lr)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the Model
print('\n\n*Starting Training:')
train_his = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose=2,
          use_multiprocessing=True)

#print('\n\n*Training Evaluation:')
#train_score = model.evaluate(self.x_train, self.y_train, verbose=0)
#print('\n\n*Validation Evaluation:')
#val_score = model.evaluate(x_valid, y_valid, verbose=0)
#print('\n\n*Test Evaluation:')
#test_score = model.evaluate(x_test, y_test, verbose=0)


import json

fh = open('./results/U02.4/learning_curves_1.json', "w")

json.dump(train_his.history, fh)
fh.close()