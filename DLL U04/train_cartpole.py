import numpy as np
import gym
import itertools as it
from dqn.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from dqn.networks import NeuralNetwork, TargetNetwork
from utils import EpisodeStats

from schedule import Schedule
from early_stop import EarlyStop


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    
    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:
        
        action_id = agent.act(state=[state], deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:  
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if rendering:
            env.render()

        if terminal or step > max_timesteps: 
            break

        step += 1

    return stats

def train_online(env, agent, num_episodes, epsilon_decay, early_stop,
                 model_dir="./models_cartpole", tensorboard_dir="./tensorboard", rendering = False):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")

    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_duration",
                                                                      "episode_reward", "validation_reward",
                                                                      "episode_reward_100", "validation_reward_10",
                                                                      "a_0", "a_1", "epsilon"])

    # training
    valid_reward = 0

    train_rewards_100 = np.zeros(100)
    valid_rewards_10  = np.zeros(10)
    train_reward_100 = 0
    valid_reward_10  = 0


    for i in range(num_episodes):

        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...

        deterministic = False
        training = True

        # Validation (Deterministic)
        if i % 10 == 0:
            deterministic = True
            #training = False

        if epsilon_decay is not None:
            agent.epsilon = epsilon_decay(i)


        stats = run_episode(env, agent, deterministic=deterministic, do_training=training, rendering=rendering)

        ep_type = '   '
        # Validation (Deterministic)
        if i % 10 == 0:
            valid_reward = stats.episode_reward
            ep_type = '(v)'

            valid_rewards_10 = np.append(valid_rewards_10, valid_reward)
            valid_reward_10 += (valid_reward - valid_rewards_10[0])/10
            valid_rewards_10 = valid_rewards_10[1:]

        train_rewards_100 = np.append(train_rewards_100, stats.episode_reward)
        train_reward_100 += (stats.episode_reward - train_rewards_100[0])/100
        train_rewards_100 = train_rewards_100[1:]

        tensorboard.write_episode_data(i, eval_dict={"episode_reward": stats.episode_reward,
                                                     "validation_reward" : valid_reward,
                                                     "episode_reward_100" : train_reward_100,
                                                     "validation_reward_10": valid_reward_10,
                                                     "episode_duration": stats.episode_steps,
                                                     "a_0": stats.get_action_usage(0),
                                                     "a_1": stats.get_action_usage(1),
                                                     "epsilon": agent.epsilon})


        # store model every 100 episodes and in the end.
        if i % 100 == 0 or i >= (num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))

        print("Episode", ep_type , ": ", '{:7d}'.format(i), ' Reward: ', '{:4.0f}'.format(stats.episode_reward))




        # Early Stopping
        early_stop.step(stats.episode_reward)

        if early_stop.save_flag:
            if early_stop.stop:
                break

            else:
                agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))

   
    tensorboard.close_session()

def prefill_buffer(env, agent, rendering = False, max_timesteps = 1000):

    episode = 0
    while (not agent.replay_buffer.has_min_items()):

        state = env.reset()
        stats = EpisodeStats()

        step = 0
        episode += 1

        while True:
            action_id = agent.act(state=[state], deterministic=False)
            next_state, reward, terminal, info = env.step(action_id)

            agent.replay_buffer.add_transition(state, action_id, next_state, reward, terminal)
            stats.step(reward, action_id)

            state = next_state

            if rendering:
                env.render()

            if terminal or step > max_timesteps:
                break

            step += 1

        print("Prefill Episode: ", '{:7d}'.format(episode), ' Reward: ', '{:4.0f}'.format(stats.episode_reward),
              ' Buffer Filled: ', '{:8d}'.format(agent.replay_buffer.len()))



if __name__ == "__main__":

    # You find information about cartpole in 
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped

    # TODO: 
    # 1. init Q network and target network (see dqn/networks.py)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    # 3. train DQN agent with train_online(...)

    # <JAB>
    state_dim   = 4
    num_actions = 2
    hidden      = 20

    lr          = 1e-4
    tau         = 0.01

    discount_factor = 0.99
    batch_size      = 100

    double_dqn = True
    rendering  = False

    epsilon_0        = 0.9
    min_epsilon      = 0.05
    decay_episodes   = 400
    decay_function   = 'exponential'
    cosine_annealing = True
    annealing_cycles = 10

    num_episodes    = 4000
    buffer_capacity = 10000

    early_stop_patience = 20

    # Q-Function Neural Networks
    Q = NeuralNetwork(state_dim, num_actions, hidden, lr, )
    Q_Target = TargetNetwork(state_dim, num_actions, hidden, lr, tau)

    # DQN Agent
    agent = DQNAgent(Q, Q_Target, num_actions, discount_factor=discount_factor, batch_size=batch_size,
                     epsilon=epsilon_0, double_dqn=double_dqn, buffer_capacity=buffer_capacity)

    # Exploration-vs-Exploitation Parameter (Epsilon) Schedule
    epsilon_schedule = Schedule(epsilon_0, min_epsilon, decay_episodes, decay_function=decay_function,
                                 cosine_annealing=cosine_annealing, annealing_cycles=annealing_cycles)

    # Early Stop
    early_stop = EarlyStop(early_stop_patience, min_steps=decay_episodes)

    # Buffer Filling
    print("*** Prefilling Buffer ***")
    prefill_buffer(env, agent, rendering)

    # Training
    print("\n\n*** Training Agent ***")
    train_online(env, agent, num_episodes, epsilon_schedule, early_stop,
                 model_dir = './models/cartpole', tensorboard_dir='./tensorboard/cartpole',
                 rendering=rendering)


    # Wake me when you need me
    from time import sleep

    def beep(n=3):
        for _ in range(n):
            print('\a')
            sleep(0.75)

    beep()