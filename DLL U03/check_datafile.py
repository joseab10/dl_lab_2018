# <JAB>
# Script used to preprocess allready stored data
import os
import pickle
import gzip

import matplotlib.pyplot as plt
import numpy as np


def plot_data(x, n_random_images = 10, filename = 'sample_data_'):

    image_path = './report/img'

    data_len = len(x)
    random_indexes = np.random.randint(0,data_len,n_random_images)



    for r in random_indexes:

        fig = plt.figure(1)

        subplot = fig.add_subplot(1, 1, 1)

        subplot.imshow(x[r])
        subplot.set_xticks([])
        subplot.set_yticks([])

        fig.show()

        fig.savefig(image_path + filename + str(r) + '.png', dpi=300)

def unhot_actions(y):

    #tmp_y = np.zeros(len(y))
    #tmp_y[y == [0, 1, 0, 0, 0]] = 1
    #tmp_y[y == [0, 0, 1, 0, 0]] = 2
    #tmp_y[y == [0, 0, 0, 1, 0]] = 3
    #tmp_y[y == [0, 0, 0, 0, 1]] = 4

    tmp_y = np.matmul(y,np.array([0, 1, 2, 3, 4])).astype('int8')

    return tmp_y





file = './data/data[5k]_pp.pkl.gzip'
file2 = './data/data[40k]_pp.pkl.gzip'
file3 = './data/data[50k]_pp.pkl.gzip'

f = gzip.open(file, 'rb')
data = pickle.load(f)

f2 = gzip.open(file2, 'rb')
data2 = pickle.load(f2)

f3 = gzip.open(file3, 'rb')
data3 = pickle.load(f3)

y1 = unhot_actions(data['action'])
breaks = np.where(y1 > 3)
y2 = unhot_actions(data2['action'])
y3 = unhot_actions(data3['action'])

y = np.concatenate((y1, y2, y3))

bins = [-.5, .5, 1.5, 2.5, 3.5, 4.5]
bins = [0, 1, 2, 3, 4, 5]
plt.hist(y, bins = bins)
plt.savefig('./report/img/act_hist.png', dpi=300)

#plot_data(data2['state'], filename='sample_data_rgb_')
#plot_data(data3['state'], filename='sample_data_pp_')

print('States')
print(len(data['state'])        , ', ', type(data['state'])         , ', ',
      len(data['state'][0])     , ', ', type(data['state'][0])      , ', ',
      len(data['state'][0][0])  , ', ', type(data['state'][0][0])   , ', ',
          data['state'][0][0][0], ', ', type(data['state'][0][0][0])
    )
print(len(data2['state'])        , ', ', type(data2['state'])         , ', ',
      len(data2['state'][0])     , ', ', type(data2['state'][0])      , ', ',
      len(data2['state'][0][0])  , ', ', type(data2['state'][0][0])   , ', ',
          data2['state'][0][0][0], ', ', type(data2['state'][0][0][0])
    )

print('\nNext States')
print(len(data['next_state'])        , ', ', type(data['next_state'])         , ', ',
      len(data['next_state'][0])     , ', ', type(data['next_state'][0])      , ', ',
      len(data['next_state'][0][0])  , ', ', type(data['next_state'][0][0])   , ', ',
          data['next_state'][0][0][0], ', ', type(data['next_state'][0][0][0])
    )
print(len(data2['next_state'])        , ', ', type(data2['next_state'])         , ', ',
      len(data2['next_state'][0])     , ', ', type(data2['next_state'][0])      , ', ',
      len(data2['next_state'][0][0])  , ', ', type(data2['next_state'][0][0])   , ', ',
          data2['next_state'][0][0][0], ', ', type(data2['next_state'][0][0][0])
    )


print('\nActions')
print(len(data['action'])     , ', ', type(data['action'])   , ', ',
      len(data['action'][0])  , ', ', type(data['action'][0]), ', ',
          data['action'][0][0], ', ', type(data['action'][0][0])
    )
print(len(data2['action'])     , ', ', type(data2['action'])   , ', ',
      len(data2['action'][0])  , ', ', type(data2['action'][0]), ', ',
          data2['action'][0][0], ', ', type(data2['action'][0][0])
    )


#print('\nRewards')
#print(len(data['reward']), ', ', type(data['reward']))
#print(len(data2['reward']), ', ', type(data2['reward']))
#
#
#print('\nTerminals')
#print(len(data['terminal']), ', ', type(data['terminal']))
#print(len(data2['terminal']), ', ', type(data2['terminal']))