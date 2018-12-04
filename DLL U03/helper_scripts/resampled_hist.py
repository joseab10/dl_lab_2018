import numpy as np
import matplotlib.pyplot as plt


hist = [(1682, 0), (1125, 1), (595, 2), (1088, 3), (10, 4)]

y = []
for i in hist:
    tmp_data = np.zeros(i[0]) + i[1]
    y.append(tmp_data)

y = np.concatenate(y)


bins = [0, 1, 2, 3, 4, 5]
plt.hist(y, bins = bins)
#plt.show()
plt.savefig('./report/img/resampled_0.75_hist.png', dpi=300)