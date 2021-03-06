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
def resequence(x, y, history_length, batch_indexes):
    # This functions reshapes the data by adding <history_length> - 1  empty images to the front of the sequence
    # and then copy it into small sequence chunks to have a sliding window in time for the data.
    # It will be used in many-to-one LSTMs so that the previous <history_length> - 1 frames are taken into
    # consideration when predicting an action

    batch_len   = batch_indexes.shape[0]
    image_width = x.shape[1]
    image_len   = x.shape[2]
    image_chan  = x.shape[3]

    h = history_length

    #tmp_x = np.empty((batch_len - h + 1, h, image_width, image_len, image_chan))
    tmp_x = []
    for i in batch_indexes:
        tmp_x.append(x[i-h+1:i+1, :, :, :])

    tmp_x = np.stack(tmp_x)

    tmp_y = y[batch_indexes]

    return tmp_x, tmp_y

def resampling(x, y, percentage):

    tmp_y = np.argmax(y, axis=1)

    batch_size = x.shape[0]

    probabilities = np.zeros(batch_size)

    actions = [STRAIGHT,
               LEFT,
               RIGHT,
               ACCELERATE,
               BRAKE]

    for action in actions:
        action_indexes = tmp_y == action
        action_occurrence = np.sum(action_indexes)

        if action_occurrence != 0:
            probabilities[action_indexes] = batch_size / action_occurrence

    # Normalize probabilities so that they sum 1
    samples = int(percentage * y.shape[0])
    probabilities = probabilities / np.sum(probabilities)
    index_list = np.random.choice(np.arange(batch_size), samples, replace=False, p=probabilities)

    return x[index_list], y[index_list]

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

def evaluate_model(x, y, agent, max_batch_size = 500):

    acc = 0
    loss = 0

    batch_size = y.shape[0]
    num_iter = batch_size // max_batch_size

    history_length = agent.history_length

    if batch_size % max_batch_size != 0:
        num_iter += 1

    for i in range(num_iter):

        start = i * max_batch_size + history_length - 1
        end   = start + max_batch_size

        if end > batch_size:
            end = batch_size

        count = end - start

        indexes = np.arange(start, end)
        x_in, y_in = resequence(x, y, history_length, indexes)


        tmp_loss, tmp_acc = agent.evaluate(x_in, y_in)

        acc  += tmp_acc  * count
        loss += tmp_loss * count

    return loss / batch_size, acc / batch_size


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr,
                model_dir="./models", tensorboard_dir="./tensorboard", history_length = 1,
                arq_file = '', ckpt_file='', net_name='JABnet'):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your neural network in model.py 
    agent = Model(history_length=history_length, name = net_name, learning_rate=lr, from_file=arq_file)

    # Get the history length from the actual model, whether it comes from a file or a parameter
    history_length = agent.history_length


    if ckpt_file != '':
        agent.load(ckpt_file)

    model_name = agent.name
    tensorboard_dir = tensorboard_dir + '/' + model_name

    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    
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

        minibatch_indexes = np.random.randint(history_length - 1, X_train.shape[0], batch_size)

        X_minibatch, y_minibatch = resequence(X_train, y_train, history_length, minibatch_indexes)

        agent.train(X_minibatch, y_minibatch)

        if i % 1000 == 0:
            loss_train, acc_train = evaluate_model(X_train, y_train, agent)
            loss_valid, acc_valid = evaluate_model(X_valid, y_valid, agent)

            print("Minibatch: ", i ,
                  " Train accuracy: ", '{:.4f}%'.format(acc_train * 100),
                  " Train Loss: ", '{:.6f}'.format(loss_train),
                  "  |   Test accuracy: ", '{:.4f}%'.format(acc_valid * 100),
                  " Test Loss: ", '{:.6f}'.format(loss_valid))

            # Save intermediate checkpoints in case training crashes or for Early Stop
            agent.save(suffix='_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_i' + str(i) + '_TrAcc_' + "{:.4f}".format(acc_train * 100),
                       dump_architecture=False)

            eval_dict = {
                'loss'  : loss_train,
                'acc'   : acc_train,
                'vloss' : loss_valid,
                'vacc'  : acc_valid
            }
            tensorboard_eval.write_episode_data(i, eval_dict)

    agent.save(dump_architecture=False)
    # </JAB>


if __name__ == "__main__":

    # <JAB>
    parser = argparse.ArgumentParser()

    parser.add_argument('--arq_file' , action='store', default='',                 help='Load Architecture from file.')
    parser.add_argument('--ckpt_file', action='store', default='',                 help='Load Parameters from file.')
    parser.add_argument('--data_file', action='store', default='data_ln.pkl.gzip', help='Training data file.')
    parser.add_argument('--net_name' , action='store', default='JABnet',           help='Model Name.')
    parser.add_argument('--lr'       , action='store', default=0.0001,             help='Learning Rate.'                , type=float)
    parser.add_argument('--bs'       , action='store', default=64,                 help='Batch Size.'                   , type=int)
    parser.add_argument('--n_batch'  , action='store', default=100000,             help='Number of training batches.'   , type=int)
    parser.add_argument('--his_len'  , action='store', default=5,                  help='History Length for RNN.'       , type=int)
    parser.add_argument('--debug'    , action='store', default=10,                 help='Debug verbosity level [0-100].', type=int)
    parser.add_argument('--resample' , action='store', default=0,                  help='"Uniformly" resample data.'    , type=float)

    args = parser.parse_args()

    history_length = args.his_len
    batch_size = args.bs
    n_batches = args.n_batch
    lr = args.lr
    DEBUG = args.debug

    resample = args.resample

    arq_file  = args.arq_file
    ckpt_file = args.ckpt_file
    data_file = args.data_file
    net_name  = args.net_name

    # Read Data
    X_train, y_train, X_valid, y_valid = read_data("./data", data_file=data_file)

    # Pre-process Data
    X_train, y_train_onehot, X_valid, y_valid_onehot = preprocessing(X_train, y_train,
                                                                     X_valid, y_valid)

    # Better distribute the data
    if resample > 0:

        if DEBUG > 3:
            print('Train      shapes: x: ', X_train.shape, ' y: ', y_train_onehot.shape)
            print('Validation shapes: x: ', X_valid.shape, ' y: ', y_valid_onehot.shape)

        if DEBUG > 5:
            tmp_y = np.argmax(y_train_onehot, axis=1)
            train_hist = np.histogram(tmp_y, bins=[0, 1, 2, 3, 4, 5])
            train_hist = list(zip(train_hist[0], train_hist[1][:-1]))

            tmp_y = np.argmax(y_valid_onehot, axis=1)
            valid_hist = np.histogram(tmp_y, bins=[0, 1, 2, 3, 4, 5])
            valid_hist = list((zip(valid_hist[0], valid_hist[1][:-1])))

            print('Original   Training Action Distribution: ', train_hist)
            print('Original Validation Action Distribution: ', valid_hist)

        X_train, y_train_onehot = resampling(X_train, y_train_onehot, resample)
        X_valid, y_valid_onehot = resampling(X_valid, y_valid_onehot, resample)

        if DEBUG > 3:
            print('Data resampled to ', 100 * resample, '%')
            print('New      Train shapes: x: ', X_train.shape, ' y: ', y_train_onehot.shape)
            print('New Validation shapes: x: ', X_valid.shape, ' y: ', y_valid_onehot.shape)

        if DEBUG > 5:
            tmp_y = np.argmax(y_train_onehot, axis=1)
            train_hist = np.histogram(tmp_y, bins=[0, 1, 2, 3, 4, 5])
            train_hist = list(zip(train_hist[0], train_hist[1][:-1]))

            tmp_y = np.argmax(y_valid_onehot, axis=1)
            valid_hist = np.histogram(tmp_y, bins=[0, 1, 2, 3, 4, 5])
            valid_hist = list((zip(valid_hist[0], valid_hist[1][:-1])))

            print('Original   Training Action Distribution: ', train_hist)
            print('Original Validation Action Distribution: ', valid_hist)



    # Plot preprocessed data for debugging
    if DEBUG > 20:
        plot_data(X_train, y_train, history_length, history_length + 5, 'Sample Train Data')
        plot_data(X_train, y_train, history_length, history_length + 5, 'Sample Validation Data')

    # train model (you can change the parameters!)
    train_model(X_train, y_train_onehot, X_valid, y_valid_onehot,
                history_length=history_length, n_minibatches=n_batches, batch_size=batch_size, lr=lr,
                arq_file=arq_file, ckpt_file=ckpt_file, net_name=net_name)

    # </JAB>
