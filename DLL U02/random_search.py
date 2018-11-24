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

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

devices = sess.list_devices()


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
# </JAB>


class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # <JAB>
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = mnist("./")

        #(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        #self.x_train = train_images[:50000]
        #self.y_train = train_labels[:50000]
        #self.x_valid = train_images[50000:]
        #self.y_valid = train_labels[50000:]
        #self.x_test = test_images
        #self.y_test = test_labels
#
#
        print('Training Data Shapes (x, y): (', self.x_train.shape, ', ', self.y_train.shape, ')')
        print('Validation Data Shapes (x, y): (', self.x_valid.shape, ', ', self.y_valid.shape, ')')
        print('Test Data Shapes (x, y): (', self.x_test.shape, ', ', self.y_test.shape, ')')
        # </JAB>


    def compute(self, config, budget, **kwargs):
        """
        Evaluates the configuration on the defined budget and returns the validation performance.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        lr = config["learning_rate"]
        num_filters = config["num_filters"]
        batch_size = config["batch_size"]
        filter_size = config["filter_size"]

        epochs = budget

        # TODO: train and validate your convolutional neural networks here
        # <JAB>

        # Define the model
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
        train_his = model.fit(self.x_train, self.y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(self.x_test, self.y_test),
                  verbose=2,
                  use_multiprocessing=True)

        #print('\n\n*Training Evaluation:')
        #train_score = model.evaluate(self.x_train, self.y_train, verbose=0)
        print('\n\n*Validation Evaluation:')
        val_score = model.evaluate(self.x_valid, self.y_valid, verbose=0)
        print('\n\n*Test Evaluation:')
        test_score = model.evaluate(self.x_test, self.y_test, verbose=0)

        # </JAB>
        # TODO: We minimize so make sure you return the validation error here

        from tensorflow.keras.initializers import glorot_uniform  # Or your initializer of choice

        initial_weights = model.get_weights()
        new_weights = [glorot_uniform()(w.shape).eval() for w in initial_weights]
        model.set_weights(new_weights)

        return ({
            'loss': val_score[0],  # this is the a mandatory field to run hyperband
            'info': {'fit_lc' : train_his.history,
                     'test_score': test_score}  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace(exercise):
        config_space = CS.ConfigurationSpace()

        # TODO: Implement configuration space here. See https://github.com/automl/HpBandSter/blob/master/hpbandster/examples/example_5_keras_worker.py  for an example
        # <JAB>

        # Define the exercise to run from the exercise sheet

        # Depending on the exercise, define the configuration space accordingly
        # These are the hyperparameters that need to be set:
        # config["learning_rate"]
        # config["num_filters"]
        # config["batch_size"]
        # config["filter_size"]

        if exercise == 1:
            # All values will be fixed to test that the CNN works propperly
            config_space.add_hyperparameters([
                CSH.CategoricalHyperparameter('learning_rate', [0.1]),
                CSH.CategoricalHyperparameter('batch_size', [64]),
                CSH.CategoricalHyperparameter('num_filters', [16]),
                CSH.CategoricalHyperparameter('filter_size', [3]),
            ])
        elif exercise == 2:
            config_space.add_hyperparameters([
                CSH.CategoricalHyperparameter('learning_rate', [0.1, 0.01, 0.001, 0.0001]),
                CSH.CategoricalHyperparameter('batch_size', [64]),
                CSH.CategoricalHyperparameter('num_filters', [16]),
                CSH.CategoricalHyperparameter('filter_size', [3]),
            ])
        elif exercise == 3:
            config_space.add_hyperparameters([
                CSH.CategoricalHyperparameter('learning_rate', [0.1]),
                CSH.CategoricalHyperparameter('batch_size', [64]),
                CSH.CategoricalHyperparameter('num_filters', [16]),
                CSH.CategoricalHyperparameter('filter_size', [1, 3, 5, 7]),
            ])
        elif exercise == 4:
            config_space.add_hyperparameters([
                CSH.UniformFloatHyperparameter('learning_rate', lower=10e-4, upper=10e-1, default_value=10e-1, log=True),
                CSH.UniformIntegerHyperparameter('batch_size', lower=16, upper=128, default_value=64, log=True),
                CSH.UniformIntegerHyperparameter('num_filters', lower=2 ** 3, upper=2 ** 6, default_value=2 ** 4, log=True),
                CSH.CategoricalHyperparameter('filter_size', [3, 5]),
            ])
        # </JAB>

        return config_space


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--budget', type=float,
                    help='Maximum budget used during the optimization, i.e the number of epochs.', default=5)#default=12)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=5)#default=20)
args = parser.parse_args()


# <JAB>

for exercise in [2]: #range(1, 5):


    print('\n\n################################################################################')
    print('Solving Exercise ', exercise)
    print('################################################################################\n\n')


    iterations = 1
    budget = 12
    if exercise > 1:
        iterations = 8
        budget = 12
    if exercise == 4:
        iterations = 50 #args.n_iterations
        budget = 6

    # Step 1: Start a nameserver
    # Every run needs a nameserver. It could be a 'static' server with a
    # permanent address, but here it will be started for the local machine with the default port.
    # The nameserver manages the concurrent running workers across all possible threads or clusternodes.
    # Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
    NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
    NS.start()

    import os
    os.makedirs('./results/U02.' + str(exercise), exist_ok=True)
    logger = jlog('./results/U02.' + str(exercise), overwrite=True)


    # Step 2: Start a worker
    # Now we can instantiate a worker, providing the mandatory information
    # Besides the sleep_interval, we need to define the nameserver information and
    # the same run_id as above. After that, we can start the worker in the background,
    # where it will wait for incoming configurations to evaluate.
    w = MyWorker(nameserver='127.0.0.1', run_id='example1')
    w.run(background=True)

    # Step 3: Run an optimizer
    # Now we can create an optimizer object and start the run.
    # Here, we run RandomSearch, but that is not essential.
    # The run method will return the `Result` that contains all runs performed.


    rs = RandomSearch(configspace=w.get_configspace(exercise),
                      run_id='example1', nameserver='127.0.0.1',
                      min_budget=int(budget), max_budget=int(budget), result_logger=logger)
    res = rs.run(n_iterations=iterations)

    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    rs.shutdown(shutdown_workers=True)



    NS.shutdown()

    # Step 5: Analysis
    # Each optimizer returns a hpbandster.core.result.Result object.
    # It holds information about the optimization run like the incumbent (=best) configuration.
    # For further details about the Result object, see its documentation.
    # Here we simply print out the best config and some statistics about the performed runs.
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])


    # Plots the performance of the best found validation error over time
    all_runs = res.get_all_runs()
    # Let's plot the observed losses grouped by budget,
    import hpbandster.visualization as hpvis

    hpvis.losses_over_time(all_runs)

    import matplotlib.pyplot as plt
    plt.savefig('./results/U02.' + str(exercise) + "/random_search.png")

    lc = []
    ev = []

    fig = plt.figure()
    loss_fig = fig.add_subplot(2,1,1)
    loss_fig.set_title('Loss')
    loss_fig.set_ylabel('loss')
    loss_fig.set_xlabel('epochs')
    loss_fig.grid(True)

    acc_fig = fig.add_subplot(2,1,2)
    acc_fig.set_title('Accuracy')
    acc_fig.set_ylabel('accuracy')
    acc_fig.set_xlabel('epochs')
    acc_fig.grid(True)


    for id, result in res.data.items():
        learning_curve = result.results[budget]['info']['fit_lc']
        evaluation_results = result.results[budget]['info']['test_score']
        lc.append({str(id): learning_curve})
        ev.append({str(id): evaluation_results})

        # Create labels
        #label = 'Iter: ' + str(id) # + '; '
        #for hyparam, value in id2config[id]['config'].items():
        #    label += hyparam + ': ' + str(value) + '; '
        #label = label[:-2]


        loss_fig.plot(learning_curve['loss'], label='TRAIN.: ' + str(id))
        loss_fig.plot(learning_curve['val_loss'], label='VALID.: ' + str(id))

        acc_fig.plot(learning_curve['acc'], label='TRAIN.: ' + str(id))
        acc_fig.plot(learning_curve['val_acc'], label='VALID.: ' + str(id))


    loss_fig.legend(loc='upper right', fontsize='xx-small',ncol=2)
    acc_fig.legend(loc='lower right', fontsize='xx-small', ncol=2)


    fig.savefig('./results/U02.' + str(exercise) + "/learning_curves.png")

    import json

    fh = open('./results/U02.' + str(exercise) + '/learning_curves.json', "w")

    json.dump(lc, fh)
    fh.close()

    fh = open('./results/U02.' + str(exercise) + '/evaluation_results.json', "w")
    json.dump(ev, fh)
    fh.close()


# TODO: retrain the best configuration (called incumbent) and compute the test error

# </JAB>