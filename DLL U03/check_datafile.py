# <JAB>
# Script used to preprocess allready stored data
import os
import pickle
import gzip


file = './data/data.pkl.gzip'
file2 = './data/data[5k].pkl.gzip'

f = gzip.open(file, 'rb')
data = pickle.load(f)

f2 = gzip.open(file2, 'rb')
data2 = pickle.load(f2)

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