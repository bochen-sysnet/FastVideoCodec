import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


data = np.random.rand(2, 5, 100) * 0.5 + 0.5

data_mean = np.mean(data, axis=2)
data_std = np.std(data, axis=2)

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "font.size": 20,
}
rcParams.update(config)

fig = plt.figure()
ax = fig.add_subplot(111)
num_methods = data_mean.shape[1]
num_env = data_mean.shape[0]
center_index = np.arange(1, num_env + 1)
colors = ['lightcoral', 'orange', 'yellow', 'palegreen', 'lightskyblue']
# colors = ['coral', 'orange', 'green', 'cyan', 'blue']
methods = ['A', 'B', 'C', 'D', 'E']
hatches = ['/', '-', 'O', '|', '\\']
envs = ['WiFi', 'Lossy WiFi']

for i in range(num_methods):
    x_index = center_index + (i - (num_methods - 1) / 2) * 0.16
    plt.bar(x_index, data_mean[:, i], width=0.15, linewidth=2,
            color=colors[i], label=methods[i], hatch=hatches[i], edgecolor='k')
    plt.errorbar(x=x_index, y=data_mean[:, i],
                 yerr=data_std[:, i], fmt='k.', elinewidth=3)

ax.grid()
ax.spines['bottom'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
plt.xticks(np.linspace(1, 2, 2), envs, size=32)
plt.yticks(np.linspace(0, 1, 6), size=32)
ax.set_ylabel('Latency', size=40)
# ax.set_xlabel('latency', size=40)

plt.legend(bbox_to_anchor=(0.46, 1.28), fancybox=True,
           loc='upper center', ncol=3, fontsize=20)
fig.savefig('latency.pdf', bbox_inches='tight', format='pdf')
plt.close()
