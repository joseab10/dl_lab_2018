import tensorflow as tf
import numpy as np
from dqn.replay_buffer import ReplayBuffer

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, discount_factor=0.99, batch_size=64, epsilon=0.05, act_probabilities = None):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            discount_factor: gamma, discount factor of future rewards.
            batch_size: Number of samples per batch.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
        """
        self.Q = Q      
        self.Q_target = Q_target
        
        self.epsilon = epsilon

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount_factor = discount_factor

        # define replay buffer
        self.replay_buffer = ReplayBuffer()

        # Start tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        # <JAB>
        if act_probabilities is None:
            self.act_probabilities = np.ones(num_actions) / num_actions
        else:
            self.act_probabilities = act_probabilities


    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update: 
        #       2.1 compute td targets: 
        #              td_target =  reward + discount * argmax_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #              self.Q.update(...)
        #       2.3 call soft update for target network
        #              self.Q_target.update(...)

        # <JAB>
        self.replay_buffer.add_transition(state,action,next_state,reward, terminal)

        buffer = self.replay_buffer.next_batch(self.batch_size)
        batch_states      = buffer[0]
        batch_actions     = buffer[1]
        batch_next_states = buffer[2]
        batch_rewards     = buffer[3]
        batch_dones       = buffer[4]

        td_target = batch_rewards
        td_target[np.logical_not(batch_dones)] += self.discount_factor * np.max(self.Q_target.predict(self.sess, batch_next_states), axis=1)[np.logical_not(batch_dones)]
        self.Q.update(self.sess, batch_states, batch_actions, td_target)
        self.Q_target.update(self.sess)
   

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # TODO: take greedy action (argmax)
            # action_id = ...
            # <JAB>
            action_id = np.argmax(self.Q.predict(self.sess, state.reshape(1,-1)))
            # </JAB>

        else:

            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            # action_id = ...
            # <JAB>
            action_id = np.random.choice(np.arange(self.num_actions), p=self.act_probabilities)
            # </JAB>
          
        return action_id


    def load(self, file_name):
        self.saver.restore(self.sess, file_name)
