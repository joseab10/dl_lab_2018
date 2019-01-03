import tensorflow as tf
import numpy as np

# TODO: add your Convolutional Neural Network for the CarRacing environment.

class NeuralNetwork():
    """
    Neural Network class based on TensorFlow.
    """
    def __init__(self, state_dim, num_actions, hidden=20, lr=1e-4, net_name='DQN'):
        self._build_model(state_dim, num_actions, hidden, lr, net_name=net_name)
        
    def _build_model(self, state_dim, num_actions, hidden, lr, net_name):
        """
        This method creates a neural network with two hidden fully connected layers and 20 neurons each. The output layer
        has #a neurons, where #a is the number of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with a learning rate of 1e-4).
        """

        with tf.variable_scope(net_name):
            self.states_ = tf.placeholder(tf.float32, shape=[None, state_dim])
            self.actions_ = tf.placeholder(tf.int32, shape=[None])                  # Integer id of which action was selected
            self.targets_ = tf.placeholder(tf.float32,  shape=[None])               # The TD target value

            # network
            fc1 = tf.layers.dense(self.states_, hidden, tf.nn.relu)
            fc2 = tf.layers.dense(fc1, hidden, tf.nn.relu)
            self.predictions = tf.layers.dense(fc2, num_actions)

            # Get the predictions for the chosen actions only
            batch_size = tf.shape(self.states_)[0]
            gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_
            self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

            # Calculate the loss
            self.losses = tf.squared_difference(self.targets_, self.action_predictions)
            self.loss = tf.reduce_mean(self.losses)

            # Optimizer Parameters from original paper
            self.optimizer = tf.train.AdamOptimizer(lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        prediction = sess.run(self.predictions, { self.states_: states })
        return prediction


    def update(self, sess, states, actions, targets):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.
        
        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """
        feed_dict = { self.states_: states, self.targets_: targets, self.actions_: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class TargetNetwork(NeuralNetwork):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """
    def __init__(self, state_dim, num_actions, hidden=20, lr=1e-4, tau=0.01):
        NeuralNetwork.__init__(self, state_dim, num_actions, hidden, lr, net_name='TargetDQN')
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx,var in enumerate(tf_vars[0:total_vars//2]):
            op_holder.append(tf_vars[idx+total_vars//2].assign(
              (var.value()*self.tau) + ((1-self.tau)*tf_vars[idx+total_vars//2].value())))
        return op_holder

        #from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DQN')
        #to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TargetDQN')
#
        #var_ids = {}
#
        #for id, var in enumerate(to_vars):
        #    var_name = var.name.split('/', 1)[1]
        #    var_ids[var_name] = id
#
        #op_holder = []
        #for from_var in from_vars:
        #    var_name = from_var.name.split('/', 1)[1]
#
        #    to_var = to_vars[var_ids[var_name]]
#
        #    op_holder.append(to_var.assign((self.tau * from_var.value()) + ((1 - self.tau) * to_var.value())))
#
        #return op_holder
      
    def update(self, sess):
        for op in self._associate:
          sess.run(op)

    def predict_Q(self, sess, states, actions):

        Q_prediction = sess.run(self.action_predictions, {self.states_: states, self.actions_: actions})

        return Q_prediction



class CNN():
    def __init__(self, img_width, img_height, hist_len, num_actions, lr=1e-4, net_name='DQN',
                 conv_layers=2, fc_layers = 1):

        self._build_model(img_width, img_height, hist_len, num_actions, lr, net_name=net_name,
                          conv_layers=conv_layers, fc_layers=fc_layers)

    def _build_model(self, img_width, img_height, hist_len, num_actions, lr, net_name,
                     conv_layers=2, fc_layers=1):
        """
        This method creates a neural network with two hidden fully connected layers and 20 neurons each. The output layer
        has #a neurons, where #a is the number of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with a learning rate of 1e-4).
        """

        with tf.variable_scope(net_name):

            with tf.name_scope("INPUT"):
                self.states_ = tf.placeholder(tf.float32, shape=[None, img_width, img_height, hist_len], name='states')
                self.actions_ = tf.placeholder(tf.int32, shape=[None], name='actions')  # Integer id of which action was selected
                self.targets_ = tf.placeholder(tf.float32, shape=[None], name='targets')  # The TD target value

            # network

            with tf.name_scope("CNN"):
                with tf.name_scope("cl1"):
                    cv1 = tf.layers.conv2d(self.states_, 3, 7,
                                          strides=(1, 1),
                                          padding='VALID',
                                          activation=tf.nn.relu,
                                          name='conv1'
                                          )

                    last_layer = tf.nn.max_pool(cv1,
                                         [1, 2, 2, 1],
                                         [1, 2, 2, 1],
                                         'VALID',
                                         name='maxp1'
                                        )
                if conv_layers >= 2:
                    with tf.name_scope("cl2"):
                        cv2 = tf.layers.conv2d(last_layer, 5, 5,
                                               strides=(1, 1),
                                               padding='VALID',
                                               activation=tf.nn.relu,
                                               name='conv2'
                                               )

                        last_layer = tf.nn.max_pool(cv2,
                                             [1, 2, 2, 1],
                                             [1, 2, 2, 1],
                                             'VALID',
                                             name='maxp2'
                                             )
                if conv_layers >= 3:
                    with tf.name_scope("cl3"):
                        cv3 = tf.layers.conv2d(last_layer, 10, 3,
                                               strides=(1, 1),
                                               padding='VALID',
                                               activation=tf.nn.relu,
                                               name='conv3'
                                               )

                        last_layer = tf.nn.max_pool(cv3,
                                             [1, 2, 2, 1],
                                             [1, 2, 2, 1],
                                             'VALID',
                                             name='maxp3'
                                             )

                if conv_layers >= 4:
                    with tf.name_scope("cl4"):
                        cv4 = tf.layers.conv2d(last_layer, 10, 3,
                                               strides=(1, 1),
                                               padding='VALID',
                                               activation=tf.nn.relu,
                                               name='conv4'
                                               )

                        last_layer = tf.nn.max_pool(cv4,
                                             [1, 2, 2, 1],
                                             [1, 2, 2, 1],
                                             'VALID',
                                             name='maxp4'
                                             )

            last_shape = last_layer.get_shape().as_list()
            flat_size = last_shape[1] * last_shape[2] * last_shape[3]
            flattened_output = tf.reshape(last_layer, shape=[-1, flat_size])

            with tf.name_scope("MLP"):
                last_layer = tf.layers.dense(flattened_output, 400, tf.nn.relu, name='fcl1')
                if fc_layers >= 2:
                    last_layer = tf.layers.dense(last_layer, 100, tf.nn.relu, name='fcl2')
                if fc_layers >= 3:
                    last_layer = tf.layers.dense(last_layer, 30, tf.nn.relu, name='fcl3')

            with tf.name_scope("OUTPUT"):
                self.predictions = tf.layers.dense(last_layer, num_actions, name='out1')

            # Get the predictions for the chosen actions only
            batch_size = tf.shape(self.states_)[0]
            gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_
            self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

            # Calculate the loss
            self.losses = tf.squared_difference(self.targets_, self.action_predictions)
            self.loss = tf.reduce_mean(self.losses)

            # Optimizer Parameters from original paper
            self.optimizer = tf.train.AdamOptimizer(lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        prediction = sess.run(self.predictions, {self.states_: states})
        return prediction

    def update(self, sess, states, actions, targets):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """
        feed_dict = {self.states_: states, self.targets_: targets, self.actions_: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class CNNTargetNetwork(CNN):

    def __init__(self, img_width, img_height, hist_len, num_actions, lr=1e-4, tau=0.01):
        CNN.__init__(self, img_width, img_height, hist_len, num_actions, lr=lr, net_name='TargetDQN')
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx, var in enumerate(tf_vars[0:total_vars // 2]):
            op_holder.append(tf_vars[idx + total_vars // 2].assign(
                (var.value() * self.tau) + ((1 - self.tau) * tf_vars[idx + total_vars // 2].value())))
        return op_holder

        #from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DQN')
        #to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TargetDQN')
#
        #var_ids = {}
#
        #for id, var in enumerate(to_vars):
        #    var_name = var.name.split('/', 1)[1]
        #    var_ids[var_name] = id
#
        #op_holder = []
        #for from_var in from_vars:
        #    var_name = from_var.name.split('/', 1)[1]
#
        #    to_var = to_vars[var_ids[var_name]]
#
        #    op_holder.append(to_var.assign((from_var.value() * self.tau) + ((1 - self.tau) * from_var.value())))
#
        #return op_holder


    def update(self, sess):
        for op in self._associate:
            sess.run(op)

    def predict_Q(self, sess, states, actions):

        Q_prediction = sess.run(self.action_predictions, {self.states_: states, self.actions_: actions})

        return Q_prediction