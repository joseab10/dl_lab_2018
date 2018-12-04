from model import Model
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

import gzip
import pickle

file3 = './data/data[40k]_pp.pkl.gzip'
f3 = gzip.open(file3, 'rb')
data3 = pickle.load(f3)


model_path = './models/'
image_path = './report/img/'
models = [
    { 'arq':  'JABnet.narq.json'   , 'ckpt' : 'JABnet.ckpt'},
    #{ 'arq':  'net1_CNN.narq.json' , 'ckpt' : 'net1.ckpt'},
    #{ 'arq':  'net2_CNN.narq.json' , 'ckpt' : 'net2_CNN.ckpt'},
    #{ 'arq':  'net3_CRNN.narq.json', 'ckpt' : 'net3_CRNN.ckpt'},
    #{ 'arq':  'net4_CRNN.narq.json', 'ckpt' : 'net4_CRNN.ckpt'},
]

tensorboard_dir="./tensorboard"

sample_state = data3['state'][2202]
sample_state = np.copy(np.reshape(sample_state, (96, 96, 1)))

fig = plt.figure(15)

subplot = fig.add_subplot(1, 1, 1)

subplot.imshow(np.reshape(sample_state, (96,96)), cmap='gray')
subplot.set_xticks([])
subplot.set_yticks([])

#fig.show()
fig.savefig(image_path + 'Sample_state.png', dpi=300)
plt.gcf().clear()
plt.clf()
plt.cla()
plt.close()


data3 = None


for m in models:

    agent = Model(from_file=model_path + m['arq'])
    agent.load(file_name=model_path + m['ckpt'])

    tmp_state = []
    for i in range(agent.history_length):
        tmp_state.append(sample_state)
    sample_state_agent = np.stack(tmp_state)
    sample_state_agent = np.reshape(sample_state_agent, (1, agent.history_length, 96, 96, 1))
    sample_x = {agent.X: sample_state_agent}

    variables = tf.trainable_variables()

    tf_writer = tf.summary.FileWriter(tensorboard_dir + agent.name)

    filter_summaries = []

    i = 1
    for variable in variables:
        if 'bias' not in variable.name  and \
            not ('rnn' in variable.name or 'fc' in variable.name or 'output' in variable.name):

            # Add to TensorBoard Summaries

            x_min = tf.reduce_min(variable)
            x_max = tf.reduce_max(variable)
            kernels = (variable - x_min) / (x_max - x_min)
            kernels = tf.image.convert_image_dtype(kernels, dtype=tf.uint8)

            kernels = tf.transpose(kernels, [3, 0, 1, 2])

            tf.summary.image('conv' + str(i) + '/' + variable.name, kernels, max_outputs=kernels.shape[0])

            # Show using matplotlib
            variable_val = agent.session.run(variable)
            variable_val = np.transpose(variable_val, (3, 2, 0, 1))
            num_filters = variable_val.shape[0]

            row_cols = np.ceil(np.sqrt(num_filters))

            kern_fig = plt.figure(1)

            title = agent.name + ' conv' + str(i) + ' Kernels'
            #kern_fig.suptitle(title)
            for j in range(num_filters) :
                kern_subplot = kern_fig.add_subplot(row_cols, row_cols, j + 1)
                kern_subplot.imshow(variable_val[j][0], cmap='gray')
                kern_subplot.set_xticks([])
                kern_subplot.set_yticks([])

            #kern_fig.show()
            kern_fig.savefig(image_path + title + '.png', dpi=300)
            plt.gcf().clear()

            i += 1



    for conv_layer in agent.conv_layers_conf:

        conv_op_name = conv_layer['name'] + '_1/' + conv_layer['name'] + '_seq0'
        conv_op = tf.get_default_graph().get_operation_by_name(conv_op_name)

        pool_op_name = conv_layer['name'] + '_1/' + conv_layer['name'] + '_pool/' + conv_layer['name'] +'_pool_seq0'
        pool_op = tf.get_default_graph().get_operation_by_name(pool_op_name)

        fmap_conv = conv_op.outputs[0].eval(feed_dict=sample_x, session=agent.session)
        fmap_conv = np.transpose(fmap_conv, (0, 3, 1, 2 ))[0]

        fmap_pool = pool_op.outputs[0].eval(feed_dict=sample_x, session=agent.session)
        fmap_pool = np.transpose(fmap_pool, (0, 3, 1, 2 ))[0]

        fmap_conv_fig = plt.figure(8)

        row_cols = np.ceil(np.sqrt(fmap_conv.shape[0]))

        for f in range(fmap_conv.shape[0]):
            fmap_min = np.min(fmap_conv[f])
            fmap_max = np.max(fmap_conv[f])

            fmap_range = fmap_max - fmap_min
            if fmap_range == 0:
                fmap_range = 1

            fmap_norm = (fmap_conv[f] - fmap_min) / fmap_range

            fmap_subplot = fmap_conv_fig.add_subplot(row_cols, row_cols, f + 1)
            fmap_subplot.imshow(fmap_norm, cmap='gray')
            fmap_subplot.set_xticks([])
            fmap_subplot.set_yticks([])

        #fmap_conv_fig.show()
        fmap_conv_fig.savefig(image_path + agent.name + '_' + conv_layer['name'] + '_fmaps_conv.png', dpi=300)
        plt.gcf().clear()
        plt.clf()
        plt.cla()
        plt.close()

        fmap_pool_fig = plt.figure(6)

        for f in range(fmap_conv.shape[0]):

            fmap_min = np.min(fmap_pool[f])
            fmap_max = np.max(fmap_pool[f])

            fmap_range = fmap_max - fmap_min
            if fmap_range == 0:
                fmap_range = 1

            fmap_norm = (fmap_pool[f] - fmap_min) / fmap_range


            pool_subplot = fmap_pool_fig.add_subplot(row_cols, row_cols, f + 1)
            pool_subplot.imshow(fmap_norm, cmap='gray')
            pool_subplot.set_xticks([])
            pool_subplot.set_yticks([])

        #fmap_pool_fig.show()
        fmap_pool_fig.savefig(image_path + agent.name + '_' + conv_layer['name'] + '_fmaps_pool.png', dpi=300)
        plt.gcf().clear()
        plt.clf()
        plt.cla()
        plt.close()


    for summary in filter_summaries:
        tf_writer.add_summary(summary)

    tf.reset_default_graph()