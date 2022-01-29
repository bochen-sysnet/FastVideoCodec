import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rcParams



config = {

    "mathtext.fontset": 'stix',

    "font.family": 'serif',

    "font.serif": ['Times New Roman'],

    "font.size": 20,

}

rcParams.update(config)



data = np.random.rand(2, 3, 100) * 15 + 15


data_mean = np.mean(data, axis=2)
data_std = np.std(data, axis=2)


fig = plt.figure()
ax = fig.add_subplot(111)
num_methods = data_mean.shape[1]
num_env = data_mean.shape[0]
center_index = np.arange(1, num_env + 1)
colors = ['lightcoral', 'orange', 'yellow'] #light_colors

# colors = ['coral', 'orange', 'green']

methods = ['LSVC', 'DVC', 'RLVC']
hatches = ['/', '-', 'O']
machines = ['CPU', 'GPU']

for i in range(num_methods):
    x_index = center_index - (i - (num_methods - 1) / 2) * 0.3
    plt.barh(x_index, data_mean[:, i], xerr=data_std[:, i], height=0.25,
             linewidth=2, color=colors[i], label=methods[i],
             hatch=hatches[i], edgecolor='k')
ax.grid()
ax.spines['bottom'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
plt.yticks(np.linspace(1, 2, 2)+0.15, machines, size=32,rotation=90)
plt.xticks(np.linspace(0, 30, 7), size=32)
ax.set_xlabel('FPS', size=40)
# ax.set_xlabel('latency', size=40)
plt.legend(bbox_to_anchor=(1.4, 0.5), fancybox=True,
           loc='right', ncol=1, fontsize=20)
fig.savefig('fps.pdf', bbox_inches='tight', format='pdf')
plt.close()

