from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from model import Model
from utils import *

# <JAB>
import argparse
import matplotlib.pyplot as plt
# </JAB>


def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()

    # <JAB>
    history_length = agent.history_length
    width = agent.in_image_width
    height = agent.in_image_height
    chans = agent.in_channels
    h = history_length

    # First action while the agent collects the first <history_length> samples in the queue (ACCELERATE!!!)
    # After <history_length> samples have been gathered, then the agent takes control
    first_action = ACCELERATE

    # FIFO Queue for recurrent models
    historical_states = []

    # </JAB>


    while True:

        # <JAB>
        state = rgb2gray(state.astype('float32'))
        state = state.astype('uint8')
        # TOOK ME 3 DAYS TO   ^
        # FIND THAT NOTHING   |
        # WORKED BECAUSE OF   |
        # A MISSING "U" ------+
        state = state.reshape((width, height, chans))
        state = state.astype('float32')

        # Add current step to the bottom of the queue
        historical_states.append(state)

        if h > 0:
            a = ACTIONS[first_action]['value']
            a_text = ACTIONS[first_action]['log']
            h -= 1
        else:
            # Pop the first sample from the queue and discard it
            historical_states = historical_states[1:]
            np_his_states = np.stack(historical_states)
            np_his_states = np.reshape(np_his_states, (1, history_length, width, height, chans))

            a = agent.predict(np_his_states)
            a = np.argmax(a)
            a_text = ACTIONS[a]['log']
            a = ACTIONS[a]['value']

        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1

        if DEBUG > 1:
            print('Step: ', step, ' Action: ', a_text)
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    #<JAB>
    parser = argparse.ArgumentParser()

    parser.add_argument('--arq_file', action="store" , default='models/JABnet.narq.json'                                           , help='Load Architecture from file.')
    parser.add_argument('--ckpt_file', action="store", default='models/early stop/JABnet_20181202-155353_i26000_TrAcc_82.5796.ckpt', help='Load Parameters from file.')
    parser.add_argument('--agent', action="store"    , default=''                                                                  , help='Automatically Load an Agent from its narq and ckpt files')
    parser.add_argument('--debug', action='store'    , default=0                                                                   , help='Debug verbosity level [0-100].', type=int)

    args = parser.parse_args()

    DEBUG = args.debug

    arq_file = args.arq_file
    ckpt_file = args.ckpt_file

    # Type less in the command line
    agents = {
        'net1'  : {'narq': './models/net1_CNN.narq.json' , 'ckpt': './models/net1.ckpt'},
        'net2'  : {'narq': './models/net2_CNN.narq.json' , 'ckpt': './models/net2.ckpt'},
        'net3'  : {'narq': './models/net3_CRNN.narq.json', 'ckpt': './models/net3_CRNN.ckpt'},
        'net4'  : {'narq': './models/net4_CRNN.narq.json', 'ckpt': './models/net4_CRNN.ckpt'},
        'JABnet': {'narq': './models/JABnet.narq.json'   , 'ckpt': './models/JABnet.ckpt'}
    }

    if args.agent != '':
        arq_file  = agents[args.agent]['narq']
        ckpt_file = agents[args.agent]['ckpt']


    agent = Model(from_file=arq_file)
    agent.load(file_name=ckpt_file)
    # </JAB>

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/" + agent.name + "_results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
