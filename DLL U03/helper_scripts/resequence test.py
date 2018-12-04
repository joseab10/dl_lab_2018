from train_agent import resequence
import numpy as np



y = np.arange(0,20)

print(y)

x = np.zeros((y.shape[0], 2, 2, 1))

for i in range(x.shape[0]):
    x[i,:,:,:] = i

print(x.shape)
print(x)

his_len = 3
batch_size = 5
indexes = np.random.randint(his_len - 1, x.shape[0], batch_size)
x_seq, y_seq = resequence(x, y, his_len, indexes)

print('\n\nX sequence: shape:', x_seq.shape,' data: ', x_seq)
print('\nY sequence: shape: ', y_seq.shape, ' data: ', y_seq)