from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from model import Model
from utils import *
from tensorboard_evaluation import Evaluation

# <JAB>
from datetime import datetime
import argparse
# <JAB>


def read_data(datasets_dir="./data", data_file = 'data_ln.pkl.gzip', frac = 0.1, start_sample = 0, max_samples = 10000):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, data_file)
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1 - frac) * n_samples)], y[:int((1 - frac) * n_samples)]
    X_valid, y_valid = X[int((1 - frac) * n_samples):], y[int((1 - frac) * n_samples):]

    return X_train, y_train, X_valid, y_valid


# <JAB>

def plot_data(x, y, history_length = 1, rows = 10, title = ''):

    image_path = './img/'

    fig = plt.figure(1)
    fig.suptitle(title)
    plt.subplots_adjust(hspace=1)

    rows = rows
    columns = history_length

    spi = 1

    for i in range(rows):
        for j in range(columns):
            subplot = fig.add_subplot(rows, columns, spi)
            subplot.imshow(x[i, :, :, j], cmap='gray')

            subplot.set_xticks([])
            subplot.set_yticks([])

            if j == columns - 1:
                subplot.set(xlabel='A: ' + ACTIONS[action_to_id(y[i])]['log'])

            spi += 1

    fig.show()

    fig.savefig(image_path + title + '.png', dpi=300)

# </JAB>

def resampling(x, y, samples):
    batch_size = x.shape[0]

    probabilities = np.zeros(batch_size)

    actions = [LEFT,
               RIGHT,
               ACCELERATE,
               BRAKE,
               STRAIGHT]

    for action in actions:
        action_indexes = y == action
        probabilities[action_indexes] = batch_size / np.sum(action_indexes)

    # Normalize probabilities so that they sum 1
    probabilities = probabilities / np.sum(probabilities)
    index_list = np.random.choice(np.arange(batch_size), samples, replace=False, p=probabilities)

    return index_list

def preprocessing(X_train, y_train, X_valid, y_valid):

    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    # <JAB>
    # Preprocessing now done in drive_manually before storing data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], 1))
    # </JAB>
    
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr,
                model_dir="./models", tensorboard_dir="./tensorboard", history_length = 1,
                arq_file = '', ckpt_file='', net_name='JABnet'):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your neural network in model.py 
    agent = Model(history_length=history_length, name = net_name, learning_rate=lr, from_file=arq_file)

    if ckpt_file != '':
        agent.load(ckpt_file)
    
    tensorboard_eval = Evaluation(tensorboard_dir, agent.session)

    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser
    # 
    # training loop

    # Initialize Parameters
    agent.session.run(agent.init)

    # Save Architecture (Only if the network was not loaded from one)
    if arq_file == '':
        agent.save_arq()

    for i in range(n_minibatches):

        minibatch_start = np.random.randint(0, X_train.shape[0] - batch_size - history_length - 1)
        minibatch_end   = minibatch_start + batch_size

        X_minibatch = X_train[minibatch_start : minibatch_end, :, :, :]
        y_minibatch = y_train[minibatch_start: minibatch_end, :]

        agent.train(X_minibatch, y_minibatch)

        if i % 1000 == 0:
            loss_train, acc_train = agent.evaluate(X_train, y_train, max_batch_size=200)
            loss_valid, acc_valid = agent.evaluate(X_valid, y_valid, max_batch_size=200)

            print("Minibatch: ", i , " Train accuracy: ", acc_train, " Train Loss: ", loss_train, ", Test accuracy: ", acc_valid, " Test Loss: ", loss_valid)

            # Save intermediate checkpoints in case training crashes or for Early Stop
            agent.save(suffix='_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_i' + str(i) + '_TrAcc_' + "{:.4f}".format(acc_train * 100),
                       dump_architecture=False)

            eval_dict = {
                'loss': loss_train,
                'acc' : acc_train,
                'vloss' : loss_valid,
                'vacc'  : acc_valid
            }
            tensorboard_eval.write_episode_data(i, eval_dict)

    agent.save(dump_architecture=False)
    # </JAB>


if __name__ == "__main__":

    # <JAB>
    parser = argparse.ArgumentParser()

    parser.add_argument('--arq_file' , action="store", default='./models/net5_40k_CNN.narq.json',                 help='Load Architecture from file.')
    parser.add_argument('--ckpt_file', action="store", default='',                 help='Load Parameters from file.')
    parser.add_argument('--data_file', action="store", default='data_ln.pkl.gzip', help='Training data file.')
    parser.add_argument('--net_name' , action="store", default='net5_40k_CNN',           help='Model Name.')
    parser.add_argument('--lr'       , action="store", default=0.0001,             help='Learning Rate.')
    parser.add_argument('--bs'       , action="store", default=64,                 help='Batch Size.')
    parser.add_argument('--n_batch'  , action="store", default=100000,             help='Number of training batches.')
    parser.add_argument('--his_len'  , action="store", default=5,                  help='History Length for RNN.')
    parser.add_argument('--debug'    , action='store', default=0,                  help='Debug verbosity level [0-100].')


    args = parser.parse_args()

    history_length = args.his_len
    batch_size = args.bs
    n_batches = args.n_batch
    lr = args.lr
    DEBUG = args.debug

    arq_file  = args.arq_file
    ckpt_file = args.ckpt_file
    data_file = args.data_file
    net_name  = args.net_name

    X_train, y_train, X_valid, y_valid = read_data("./data", data_file=data_file)

    # <JAB>
    # Used for quicker testing
    max_train = X_train.shape[0]
    max_valid = X_valid.shape[0]

    # preprocess data
    X_train, y_train_onehot, X_valid, y_valid_onehot = preprocessing(X_train[:max_train], y_train[:max_train],
                                                                     X_valid[:max_valid], y_valid[:max_valid])

    # Better distribute the data
    sample_percent = 0.75
    data_len = y_train_onehot.shape[0]

    dist_index = resampling(X_train, y_train_onehot, int(sample_percent * data_len))

    X_train = X_train[dist_index]
    y_train_onehot = y_train_onehot[dist_index]

    # Plot preprocessed data for debugging
    if DEBUG > 20:
        plot_data(X_train, y_train, history_length, history_length + 5, 'Sample Train Data')
        plot_data(X_train, y_train, history_length, history_length + 5, 'Sample Validation Data')

    # train model (you can change the parameters!)
    train_model(X_train, y_train_onehot, X_valid, y_valid_onehot,
                history_length=history_length, n_minibatches=n_batches, batch_size=batch_size, lr=lr,
                arq_file=arq_file, ckpt_file=ckpt_file, net_name=net_name)

    # </JAB>
