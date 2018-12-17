# export DISPLAY=:0 

import sys
sys.path.append("../") 

import numpy as np
import gym
from gym import wrappers

from dqn.dqn_agent import DQNAgent
from dqn.networks import CNN, CNNTargetNetwork
from tensorboard_evaluation import *
import itertools as it
from utils import *



def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0

    state = env.reset()
    #env._max_episode_steps = max_timesteps

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    
    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)
        action_id = agent.act([state], deterministic)
        action = id_to_action(action_id)

        #print('\tStep ', '{:7d}'.format(step), ' Action: ', ACTIONS[action_id]['log'])

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        #if terminal or (step * (skip_frames + 1)) > max_timesteps :
        #    break

        if terminal:
            break

        if step % 100 == 0:
            print('\t\tStep ', '{:4d}'.format(step), ' Reward: ', '{:4.4f}'.format(reward))

        step += 1

    return stats


def train_online(env, agent, num_episodes,
                 history_length=0,
                 model_dir="./models_carracing", tensorboard_dir="./tensorboard", rendering=False,
                 min_epsilon=0.05, epsilon_decay=0.9):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward", "straight", "left", "right", "accel", "brake"])

    max_timesteps = 1000

    for i in range(num_episodes):
        #print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)

       
        stats = run_episode(env, agent, max_timesteps=max_timesteps, deterministic=False, do_training=True, rendering=rendering)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })

        # TODO: evaluate agent with deterministic actions from time to time
        # ...

        if i % 100 == 0 or (i >= num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))

            max_timesteps += 100

        print('\tEpisode ', '{:7d}'.format(i), ' Reward: ', '{:4.4f}'.format(stats.episode_reward))

        if agent.epsilon > min_epsilon:
            agent.epsilon = epsilon_decay * agent.epsilon

    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    env = gym.make('CarRacing-v0').unwrapped


    
    # TODO: Define Q network, target network and DQN agent
    # ...
    # <JAB>
    num_actions = 5
    hidden = 20
    lr = 1e-4
    tau = 0.01
    hist_len = 0

    rendering = False

    if not rendering:
        env = wrappers.Monitor(env, './video', video_callable=False, force=True)

    Q = CNN(96, 96, hist_len + 1, 5, lr)
    Q_Target = CNNTargetNetwork(96, 96, hist_len + 1, 5, lr, tau)

    discount_factor = 0.99
    batch_size = 64
    epsilon = 0.75
    act_probabilities = np.ones(num_actions)
    act_probabilities[STRAIGHT] = 5
    act_probabilities[ACCELERATE] = 40
    act_probabilities[LEFT] = 30
    act_probabilities[RIGHT] = 30
    act_probabilities[BRAKE] = 1
    act_probabilities /= np.sum(act_probabilities)

    agent = DQNAgent(Q, Q_Target, num_actions, discount_factor=discount_factor, batch_size=batch_size, epsilon=epsilon, act_probabilities=act_probabilities)
    # </JAB>

    min_epsilon = 0.05
    epsilon_decay = 0.9

    num_episodes = 10000
    
    train_online(env, agent, num_episodes=num_episodes, history_length=hist_len, model_dir="./models/carracing",
                 tensorboard_dir='./tensorboard/carracing', rendering=rendering,
                 min_epsilon=min_epsilon, epsilon_decay=epsilon_decay)

