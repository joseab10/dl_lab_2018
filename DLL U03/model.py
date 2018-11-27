import tensorflow as tf

class Model:
    
    def __init__(self, history_length = 1, conv_layers = None, lstm_layers = None, name = '', learning_rate = 0.1, path = './models/'):

        # <JAB>
        self.name = name
        self.savepath = path


        # DEFAULT VALUES
        # ================================================================================
        image_width  = 96
        image_height = 96
        classes      = 5

        if conv_layers is None:
            conv_layers = [
                {
                    'name'        : 'conv1',
                    'filters'     : 16     ,
                    'kernel size' : 5      ,
                    'padding'     : 'SAME' ,
                    'stride'      : (1,1)  ,
                    'activation'  : tf.nn.relu,

                    'pooling'     : 'max',
                    'pool ksize'  : [1, 2, 2, 1],
                    'pool stride' : [1, 2, 2, 1],
                    'pool padding': 'VALID'
                }
            ]


        if lstm_layers is None:
            lstm_layers = [
                {
                    'name'        : 'lstm1',
                    'units'       : 16     ,
                    'kernel size' : 5      ,
                    'padding'     : 'SAME' ,
                    'stride'      : (1,1)  ,
                    'activation'  : tf.nn.relu
                }
            ]









        # MODEL
        # ================================================================================

        # Input Layer
        with tf.name_scope("inputs"):
            self.X = tf.placeholder(tf.float32,
                               shape = [None, image_width, image_height, history_length],
                               name  = "x"
                              )
            self.y = tf.placeholder(tf.float32,
                               shape = [None, classes],
                               name  = "y"
                              )


        # Convolutional Layers
        conv_input = self.X
        for layer in conv_layers:
            with tf.name_scope(layer['name']):
                conv_input = tf.layers.conv2d(conv_input,
                                              filters     = layer['filters'],
                                              kernel_size = layer['kernel size'],
                                              strides     = layer['stride'],
                                              padding     = layer['padding'],
                                              activation  = layer['activation'],
                                              name        = layer['name']
                                             )
            if 'pooling' in layer:
                with tf.name_scope(layer['name'] + '_pool'):
                    if layer['pooling'] == 'max':
                        conv_input = tf.nn.max_pool(conv_input,
                                                    ksize   = layer['pool ksize'],
                                                    strides = layer['pool stride'],
                                                    padding = layer['pool padding'],
                                                   )

        if conv_layers != []:
            # Flatten the layer
            last_shape = conv_input.get_shape().as_list()
            flat_size = last_shape[1] * last_shape[2] * last_shape[3]
            conv_input = tf.reshape(conv_input, shape=[-1,  flat_size])

        # Recurrent Layers
        lstm_input = conv_input
        for layer in lstm_layers:
            with tf.name_scope(layer['name']):

                if tf.test.is_gpu_available(cuda_only = True):
                    print('GPU Available, using CUDA optimized LSTM')
                    lstm_cell = tf.contrib.cudnn_rnn.CudnnLSTM(layer['num layers'],
                                                               layer['num units']
                                                              )

                else:
                    lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(layer['num units'])

                    lstm_input, new_states = tf.nn.dynamic_rnn(lstm_cell, lstm_input)


        # Dense Layers
        dense_input = lstm_input

        dense_input = tf.layers.dense(dense_input,
                              30,
                              activation=tf.nn.relu,
                              name="fc1")


        # Output Layer
        with tf.name_scope("output"):
            self.logits = tf.layers.dense(dense_input,
                                     classes,
                                     name="output")
            self.Y_proba = tf.nn.softmax(self.logits,
                                    name="Y_proba")

        # LOSS AND OPTIMIZER
        # ================================================================================
        with tf.name_scope("train"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            self.loss = tf.reduce_mean(xentropy)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.trainer = self.optimizer.minimize(self.loss)


        with tf.name_scope("eval"):
            correct = tf.equal(tf.argmax(self.y,1), tf.argmax(self.logits,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # TENSORFLOW SESSION
        # ================================================================================
        self.init = tf.global_variables_initializer()

        self.session = tf.Session()

        self.saver = tf.train.Saver()

    def load(self, file_name=''):
        # <JAB>
        if file_name == '':
            file_name = self.savepath + self.name + '.ckpt'
        # </JAB>

        self.saver.restore(self.sess, file_name)

    def save(self, file_name=''):
        # <JAB>
        if file_name == '':
            file_name = self.savepath + self.name + '.ckpt'
        # </JAB>

        self.saver.save(self.sess, file_name)
        print("Model saved in file: %s" % file_name)
