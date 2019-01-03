from matplotlib import pyplot as plt
from schedule import Schedule
import numpy as np

e_0 = 0.9
e_min = 0.05
epochs = 100

annealing_cycles = 5

x = np.arange(0,200,1)

decay_functions =[
    {'decay': 'constant'   , 'annealing' : True , 'steps' :  0, 'label': 'Constant'},
    #{'decay': 'linear'     , 'annealing' : False, 'steps' :  0, 'label': 'Linear'},
    #{'decay': 'exponential', 'annealing' : False, 'steps' :  0, 'label': 'Exponential'},
    # Stepped
    #{'decay': 'linear'     , 'annealing' : False, 'steps' : 10, 'label': 'Linear Step'},
    {'decay': 'exponential', 'annealing' : False, 'steps' : 10, 'label': 'Exponential Step'},
    # Annealed
    #{'decay': 'linear'     , 'annealing': True  , 'steps' :  0, 'label': 'Linear Annealing'},
    #{'decay': 'exponential', 'annealing': True  , 'steps' :  0, 'label': 'Exponential Annealing'},
    # Stepped and Annealed
    #{'decay': 'linear'     , 'annealing': True  , 'steps' : 10, 'label': 'Linear Step Annealing'},
    {'decay': 'exponential', 'annealing': True  , 'steps' : 10, 'label': 'Exponential Step Annealing'},
]



fig = plt.figure(1)
plot = fig.add_subplot(1,1,1)

for function in decay_functions:
    epsilon = Schedule(e_0, e_min, epochs, decay_function=function['decay'], cosine_annealing=function['annealing'],
                           steps = function['steps'], annealing_cycles=annealing_cycles)

    y = epsilon(x)

    plot.plot(x, y, label=function['label'])

fig.legend()
fig.show()

