import tensorflow as tf

# <JAB>
import json
import numpy as np
from utils import DEBUG
# </JAB>

class Model:
    
    def __init__(self, from_file='',  name = 'JABnet', path = './models/',
                 conv_layers_conf = None, lstm_layers_conf = None, fc_layers_conf = None,
                 learning_rate = 0.1, history_length = 1, dropout_rate = 0.8,
                 l2_penalty = 0.01,
                 in_image_width = 96, in_image_height = 96, in_channels = 1, out_classes = 5, ):

        # <JAB>
        self.name = name
        self.savepath = path

        if from_file != '':
            # Get architecture from file
            # ================================================================================
            arq_file = open(from_file, "r")
            net_arq = json.load(arq_file)

            self.name     = net_arq['name']
            self.savepath = net_arq['path']

            self.history_length = net_arq['history_length']
            self.learning_rate  = net_arq['learning_rate']
            self.dropout_rate   = net_arq['dropout_rate']
            self.l2_penalty     = net_arq['l2_penalty']

            self.in_image_width  = net_arq['in_image_width']
            self.in_image_height = net_arq['in_image_height']
            self.in_channels     = net_arq['in_channels']
            self.out_classes     = net_arq['out_classes']

            self.conv_layers_conf = net_arq['conv_layers']
            self.lstm_layers_conf = net_arq['lstm_layers']
            self.fc_layers_conf   = net_arq['fc_layers']


        else:
            # Get architecture from parameters
            self.history_length = history_length
            self.learning_rate  = learning_rate
            self.dropout_rate   = dropout_rate
            self.l2_penalty     = l2_penalty

            self.in_image_width  = in_image_width
            self.in_image_height = in_image_height
            self.in_channels     = in_channels
            self.out_classes     = out_classes

            # DEFAULT Architecture if parameters are empty
            # ================================================================================
            if conv_layers_conf is None:
                self.conv_layers_conf = [
                    {
                        'name'        : 'conv1',
                        'filters'     : 16     ,
                        'kernel size' : 5      ,
                        'padding'     : 'SAME' ,
                        'stride'      : [1, 1, 1, 1],
                        'activation'  : 'relu',

                        'pooling'     : 'max',
                        'pool ksize'  : [1, 2, 2, 1],
                        'pool stride' : [1, 2, 2, 1],
                        'pool padding': 'VALID'
                    },

                    {
                        'name'       : 'conv2',
                        'filters'    : 24,
                        'kernel size': 3,
                        'padding'    : 'SAME',
                        'stride'     : [1, 1, 1, 1],
                        'activation' : 'relu',

                        'pooling'     : 'max',
                        'pool ksize'  : [1, 2, 2, 1],
                        'pool stride' : [1, 2, 2, 1],
                        'pool padding': 'VALID'
                    }
                ]
            else :
                self.conv_layers_conf = conv_layers_conf

            if lstm_layers_conf is None:
                self.lstm_layers_conf = [
                    {
                        'name'        : 'lstm1',
                        'units'       : 16     ,
                    }
                ]
            else:
                self.lstm_layers_conf = lstm_layers_conf

            if fc_layers_conf is None:
                self.fc_layers_conf = [
                    {
                        'units'      : 100,
                        'activation' : 'relu',
                        'name'      : "fc1",
                    },
                    {
                        'units'     : 30,
                        'activation': 'relu',
                        'name'      : "fc2",
                    }
                ]
            else:
                self.fc_layers_conf = fc_layers_conf


        # MODEL
        # ================================================================================

        # Input Layer
        with tf.name_scope("inputs"):
            '''
                5D Input Tensor X
                with shape [B T W H C]
            '''
            self.X = tf.placeholder(tf.float32,
                               shape = [None, self.history_length, self.in_image_width, self.in_image_height, self.in_channels],
                               name  = "x"
                              )
            self.y = tf.placeholder(tf.float32,
                               shape = [None, self.out_classes],
                               name  = "y"
                              )

        batch_size = tf.shape(self.X)[0]

        last_output = self.X

        # Convolutional Layers
        if self.conv_layers_conf != []:

            # Create and initialize the weights and biases to be used by all layers.
            # The weights and biases are to be shared by all sequences in history_length.
            # Basically, slice the input into each state frame, convolve every frame with the same filters
            # and then reunite them again so as to not loose the temporal dimension, while at the same time
            # allowing to use this architecture with multi-channel images.

            conv_filters = []
            conv_bias    = []
            l = 1
            for layer in self.conv_layers_conf:

                if conv_filters == []:
                    last_shape = last_output.get_shape().as_list()
                else :
                    last_shape = conv_filters[-1].get_shape().as_list()

                # Select the best initialization method for each activation function
                initializer = self.init_from_activation_str(layer['activation'])

                with tf.name_scope(layer['name']):
                    new_filter = tf.get_variable('kernel_' + str(l),
                                                 [layer['kernel size'], layer['kernel size'], last_shape[-1], layer['filters']],
                                                 initializer=initializer)

                    new_bias = tf.get_variable('bias_' + str(l),
                                                 [layer['filters']],
                                                 initializer=initializer)

                conv_filters.append(new_filter)
                conv_bias.append(new_bias)
                l += 1

            conv_layer_count = len(self.conv_layers_conf)
            for i in range(conv_layer_count):

                layer = self.conv_layers_conf[i]

                tmp_conv_4Doutputs = []

                last_outputs = tf.unstack(last_output, axis=1)

                with tf.name_scope(layer['name']):

                    for h in range(self.history_length):

                        # Slice across temporal dimension
                        tmp_conv_4Dinput = last_outputs[h]

                        # Select the best initialization method for each activation function
                        activation = self.activation_from_str(layer['activation'])
                        tmp_conv_4Dout = tf.nn.conv2d(tmp_conv_4Dinput,
                                                   conv_filters[i],
                                                   layer['stride'],
                                                   layer['padding'],
                                                   name = layer['name'] + '_seq' + str(h))
                        tmp_conv_4Dout = tf.nn.bias_add(tmp_conv_4Dout, conv_bias[i])
                        tmp_conv_4Dout = activation(tmp_conv_4Dout)

                        if 'pooling' in layer:
                            with tf.name_scope(layer['name'] + '_pool'):
                                if layer['pooling'] == 'max':
                                    tmp_conv_4Dout = tf.nn.max_pool(tmp_conv_4Dout,
                                                                ksize   = layer['pool ksize'],
                                                                strides = layer['pool stride'],
                                                                padding = layer['pool padding'],
                                                                name    = layer['name'] + '_pool' + '_seq' + str(h)
                                                               )

                        tmp_conv_4Doutputs.append(tmp_conv_4Dout)

                # Restack the temporal dimension after convolving every image individually
                last_output = tf.stack(tmp_conv_4Doutputs)
                last_output = tf.transpose(last_output, perm=[1,0,2,3,4])

            '''
            Flatten the output so as to make the tensor shape 3D, ready for LSTM or Fully Connected Layer
            Shape: [B, T, F]
            
            Where:
                   B - Batch Size
                   T - Time (history_length)
                   F - Feature Map Size ( W x H x C )
                                            W - Feature Map Width
                                            H - Feature Map Height
                                            C - Feature Map Channels
            '''
            last_shape = last_output.get_shape().as_list()
            flat_size = last_shape[2] * last_shape[3] * last_shape[4]
            last_output = tf.reshape(last_output, shape=[-1, self.history_length,  flat_size])

        # Recurrent Layers
        if self.lstm_layers_conf != []:
            with tf.name_scope('RNN'):

                lstm_cells = []
                for layer in self.lstm_layers_conf:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(layer['units'])
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_rate)
                    lstm_cells.append(lstm_cell)

                rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cells)

                # Transpose again and unstack just for the RNN to see it as a sequence
                last_output = tf.transpose(last_output, [1, 0, 2])
                last_output = tf.unstack(last_output)

                outputs, final_state = tf.nn.static_rnn(rnn_cells, last_output, dtype='float32')

                #last_output = rnn_outputs[-1]
                last_output = final_state[0].h
        else:
            last_shape = last_output.get_shape().as_list()
            flat_size = last_shape[1] * last_shape[2]
            last_output = tf.reshape(last_output, shape=[-1, flat_size])


        # Dense Layers
        if self.fc_layers_conf != []:
            with tf.name_scope("FCL"):
                for layer in self.fc_layers_conf:

                    activation = self.activation_from_str(layer['activation'])

                    last_output = tf.layers.dense(last_output,
                                      layer['units'],
                                      activation=activation,
                                      name=layer['name'])

                    last_output = tf.nn.dropout(last_output, self.dropout_rate, name=layer['name'] + '_dropout')


        # Output Layer
        with tf.name_scope("output"):
            self.logits = tf.layers.dense(last_output,
                                     self.out_classes,
                                     name="output")
            self.Y_proba = tf.nn.softmax(self.logits,
                                    name="Y_proba")

        # LOSS AND OPTIMIZER
        # ================================================================================
        with tf.name_scope("train"):
            self.x_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='x_entropy')

            # L2 Regularization
            variables = tf.trainable_variables()
            weights = []
            self.l2_loss = tf.zeros([batch_size], name='l2_loss')
            for variable in variables:
                if 'bias' not in variable.name:
                    weights.append(variable)
                    self.l2_loss += tf.nn.l2_loss(variable)

            self.loss = tf.reduce_mean(self.x_entropy + (self.l2_penalty * self.l2_loss))


            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.trainer = self.optimizer.minimize(self.loss)


        with tf.name_scope("eval"):
            self.correct = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.Y_proba, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        # TENSORFLOW SESSION
        # ================================================================================
        self.init = tf.global_variables_initializer()

        self.session = tf.Session()

        self.saver = tf.train.Saver()

    def train(self, x, y):
        self.session.run(self.trainer, feed_dict={self.X: x, self.y: y})

    def evaluate(self, x, y):
        acc  = self.accuracy.eval(feed_dict={self.X: x, self.y: y},
                session = self.session)

        loss = self.loss.eval(feed_dict={self.X: x, self.y: y},
                session=self.session)

        return loss, acc

    def predict(self, x):
        return self.Y_proba.eval(feed_dict={self.X: x}, session=self.session)

    def load(self, file_name=''):
        # <JAB>
        if file_name == '':
            file_name = self.savepath + self.name + '.ckpt'
        # </JAB>

        self.saver.restore(self.session, file_name)

    def save(self, file_name='', suffix='', dump_architecture = True):
        # <JAB>
        if file_name == '':
            file_name = self.savepath + self.name + suffix + '.ckpt'


        # Dump the Architecture
        if dump_architecture:
            self.save_arq()
        # </JAB>

        self.saver.save(self.session, file_name)

        if DEBUG > 15:
            print("Model saved in file: %s" % file_name)


    def save_arq(self, file_name='', suffix=''):

        if file_name == '':
            file_name = self.savepath + self.name + suffix + '.narq.json'

        # Build a Dictionary with the network's configuration
        net_arq = {}
        net_arq['name'] = self.name
        net_arq['path'] = self.savepath

        net_arq['history_length'] = self.history_length
        net_arq['learning_rate']  = self.learning_rate
        net_arq['dropout_rate']   = self.dropout_rate
        net_arq['l2_penalty']     = self.l2_penalty

        net_arq['in_image_width']  = self.in_image_width
        net_arq['in_image_height'] = self.in_image_height
        net_arq['in_channels']     = self.in_channels
        net_arq['out_classes']     = self.out_classes

        net_arq['conv_layers'] = self.conv_layers_conf
        net_arq['lstm_layers'] = self.lstm_layers_conf
        net_arq['fc_layers']   = self.fc_layers_conf

        f = open(file_name, 'w')
        json.dump(net_arq, f)
        f.close()



    def activation_from_str(self, activation_string):

        def nn_linear(x):
            return x

        if   activation_string == 'relu':
            return tf.nn.relu
        elif activation_string == 'sigmoid':
            return tf.nn.sigmoid
        elif activation_string == 'tanh':
            return tf.nn.tanh
        else:
            return nn_linear

    def init_from_activation_str(self, activation_string):
        if activation_string == 'relu':
            return tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        elif activation_string == 'sigmoid':
            return tf.contrib.layers.xavier_initializer()
        elif activation_string == 'tanh':
            return tf.contrib.layers.xavier_initializer()
        else:
            return tf.random_normal_initializer
