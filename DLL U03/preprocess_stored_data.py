
# <JAB>
# Script used to preprocess allready stored data
import os
import pickle
import gzip
import numpy as np
from utils import rgb2gray, one_hot, action_to_id

path = './data/'
ext = '.pkl.gzip'
suffix = '_pp'
files = [ 'data[5k]',
         'data[40k]']

max_samples = 5000


for file in files:

    data_path = os.path.join(path, file + ext)

    print('* Preprocessing File ', data_path)

    f = gzip.open(data_path, 'rb')
    unprocessed_data = pickle.load(f)

    total_samples = len(unprocessed_data['state'])
    samples = total_samples // max_samples

    if total_samples % max_samples != 0:
        samples += 1


    preprocessed_data = {
        'state'      : [],
        'action'     : [],
        'next_state' : [],
     #   'reward'     : [],
     #   'terminal'   : []
    }

    # Not used in the exercise, so not stored to save space and loading time
    #preprocessed_data['reward']   = unprocessed_data['reward']
    #preprocessed_data['terminal'] = unprocessed_data['terminal']

    for i in range(samples):

        start_sample = i * max_samples
        end_sample = start_sample + max_samples

        if end_sample > total_samples:
            end_sample = total_samples

        print('    Preprocessing Batch ', i, ' ranging from sample ', start_sample, ' to ', end_sample)

        print('        Preprocessing States')
        tmp_state = rgb2gray(np.array(unprocessed_data['state'][start_sample:end_sample]).astype('float32'))
        preprocessed_data['state'][start_sample:end_sample] = tmp_state.astype('uint8')

        print('        Preprocessing Next States')
        tmp_state = rgb2gray(np.array(unprocessed_data['next_state'][start_sample:end_sample]).astype('float32'))
        preprocessed_data['next_state'][start_sample:end_sample] = tmp_state.astype('uint8')

        print('        Preprocessing Actions')
        tmp_actions = []
        for i in range(start_sample, end_sample):
            tmp_actions.append(action_to_id(unprocessed_data['action'][i]))

        preprocessed_data['action'][start_sample:end_sample] = one_hot(np.array(tmp_actions).astype('uint8')).astype('uint8')

    preprocessed_data['state']      = np.array(preprocessed_data['state']).astype('uint8')
    preprocessed_data['next_state'] = np.array(preprocessed_data['next_state']).astype('uint8')
    preprocessed_data['action'] = np.array(preprocessed_data['action']).astype('uint8')


    output_path = os.path.join(path, file + suffix + ext)
    output_file = gzip.open(output_path, 'wb')
    pickle.dump(preprocessed_data, output_file)
    print('* Saved Preprocessed File ', output_file)
    output_file.close()


# </JAB>