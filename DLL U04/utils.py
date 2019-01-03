import numpy as np


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []
        self.episode_steps = 0

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)
        self.episode_steps += 1

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))


STRAIGHT = 0
LEFT =1
RIGHT = 2
ACCELERATE =3
BRAKE = 4

ACTIONS = {
    LEFT       : {'log': '[<- ] Left'      , 'value': [-1.0,  0.0, 0.0]},
    RIGHT      : {'log': '[ ->] Right'     , 'value': [+1.0,  0.0, 0.0]},
    STRAIGHT   : {'log': '[   ] Straight'  , 'value': [ 0.0,  0.0, 0.0]},
    ACCELERATE : {'log': '[ ^ ] Accelerate', 'value': [ 0.0, +1.0, 0.0]},
    BRAKE      : {'log': '[ _ ] Break'     , 'value': [ 0.0,  0.0, +0.2]}
}

def id_to_action(id):

    action = ACTIONS[id]['value']

    return action


def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    gray =  2 * gray.astype('float32') - 1
    return gray

"""
                    &                 & \node[f]{};     & \node[f]{};     & \node[f]{};     & \node[f]{};     &                 &                 &             \\
		\node[f]{}; & \node[f]{};     & \node[f]{};     & \node(A)[g]{A}; & \node(B)[g]{B}; & \node[f]{};     & \node[f]{};     & \node[f]{};     & \node[f]{}; \\
		\node[f]{}; & \node(C)[g]{C}; & \node(D)[g]{D}; & \node(E)[g]{E}; & \node(F)[g]{F}; & \node(G)[g]{G}; & \node(H)[g]{H}; & \node(I)[g]{I}; & \node[f]{}; \\
		\node[f]{}; & \node(J)[g]{J}; & \node[f]{};     & \node(K)[g]{K}; & \node(L)[g]{L}; & \node[f]{};     & \node(M)[g]{M}; & \node(N)[g]{N}; & \node[f]{}; \\
		\node[f]{}; & \node(O)[g]{O}; & \node(P)[g]{P}; & \node(Q)[g]{Q}; & \node(R)[g]{R}; & \node[f]{};     & \node(S)[g]{S}; & \node(T)[g]{T}; & \node[f]{}; \\
		\node[f]{}; & \node[f]{};     & \node[f]{};     & \node[f]{};     & \node[f]{};     & \node[f]{};     & \node[f]{};     & \node[f]{};     & \node[f]{}; \\
		

"""
