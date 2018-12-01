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
# </JAB>


def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()

    # <JAB>
    history_length = agent.history_length
    h = history_length

    # first action while the agent collects the first h samples
    first_action = ACCELERATE

    historical_states = []

    # <JAB>


    while True:
        
        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        #    state = ...
        state = np.reshape(rgb2gray(state), (1, 96, 96, 1))
        
        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        historical_states.append(state)

        if h > 0:
            a = ACTIONS[first_action]['value']
            a_text = ACTIONS[first_action]['log']
            h -= 1
        else:
            historical_states = historical_states[1:]
            np_his_states = np.transpose(np.array(historical_states), axes=(1, 0, 2, 3, 4))

            a = agent.predict(np_his_states)
            a_text = ACTIONS[np.argmax(a)]['log']
            a = ACTIONS[np.argmax(a)]['value']

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

    # TODO: load agent
    parser = argparse.ArgumentParser()

    parser.add_argument('--arq_file', action="store", default='models/net1.narq.json', help='Load Architecture from file.')
    parser.add_argument('--ckpt_file', action="store", default='models/net1.ckpt', help='Load Parameters from file.')
    parser.add_argument('--debug', action='store', default=10, help='Debug verbosity level [0-100].', type=int)

    args = parser.parse_args()

    DEBUG = args.debug

    arq_file = args.arq_file
    ckpt_file = args.ckpt_file


    agent = Model(from_file=arq_file)
    agent.load(file_name=ckpt_file)
    # <JAB>

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
 
    fname = "results/" + agent.name + "results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
