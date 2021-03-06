import numpy as np

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4

# <JAB>

DEBUG = 0

ACTIONS = {
    LEFT       : {'log': '[<- ] Left'      , 'value': [-1.0,  0.0, 0.0]},
    RIGHT      : {'log': '[ ->] Right'     , 'value': [+1.0,  0.0, 0.0]},
    STRAIGHT   : {'log': '[   ] Straight'  , 'value': [ 0.0,  0.0, 0.0]},
    ACCELERATE : {'log': '[ ^ ] Accelerate', 'value': [ 0.0, +1.0, 0.0]},
    BRAKE      : {'log': '[ _ ] Break'     , 'value': [ 0.0,  0.0, +0.2]}
}

# </JAB>

def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    # <JAB>
    #classes = np.unique(labels)
    #n_classes = classes.size
    n_classes = 5
    classes = range(n_classes)
    # </JAB>

    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    gray =  2 * gray.astype('float32') - 1 
    return gray 


def action_to_id(a):
    """ 
    this method discretizes actions
    """

    subactions = []
    if a[0] == -1.0:
        subactions.append(LEFT)
    elif a[0] == 1.0:
        subactions.append(RIGHT)

    if a[1] == 1.0:
        subactions.append(ACCELERATE)

    if a[2] != 0.0:
        subactions.append(BRAKE)

    actions_per_step = len(subactions)
    if actions_per_step == 0:
        return STRAIGHT
    elif actions_per_step == 1:
        return subactions[0]
    else:
        # If more than one action took place in this step, choose one randomly.
        rand_index = np.random.randint(0, actions_per_step - 1, dtype='int8')
        return subactions[rand_index]
