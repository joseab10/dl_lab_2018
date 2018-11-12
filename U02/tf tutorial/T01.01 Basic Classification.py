
# --------------------------------------------------------------------------------
#  IMPORTS
# --------------------------------------------------------------------------------

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

# Helper Libraries
import numpy as np
import matplotlib.pyplot as plt

print('TensorFlow version ', tf.__version__)


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

devices = sess.list_devices()
print(devices)


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")



# 1.0 Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/Top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle Boot']



# 1.1 Explore the Data
print(train_images.shape)
print(len(train_labels))

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(8, 8))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()


# 2.0 Build the Model

# 2.1 Setup the Layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
)

# 2.2 Compile the Model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 2.3 Train the Model
print('\n*Training the Model:')
model.fit(train_images, train_labels, epochs=5)

# 2.4 Evaluate Accuracy
print('\n*Evaluating the Model:')
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test Accuracy: ', test_acc * 100, '%')