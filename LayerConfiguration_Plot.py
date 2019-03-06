from __future__ import print_function
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as pyplot
from matplotlib.ticker import MaxNLocator
import itertools
from numpy import linalg as la
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

'''Reading the configuration file'''
import json
with open('Config.json') as g:
    config = json.load(g)
with open('Plot.json') as f:
    file = json.load(f)

npzfile = np.load(file["File name"])
config_set = npzfile['arr_0']
precision_log = npzfile['arr_1']

tickfont = 13
labelfont = 14
bins = 10.**np.arange(-16.,2.,1.)

config_set = np.array(config_set)
config_set = config_set.T
N_set = np.arange(config["Test"]["L_min"][1],config["Test"]["L_max"][1],config["Test"]["Steps"])
M_set = np.arange(config["Test"]["L_min"][0],config["Test"]["L_max"][0],config["Test"]["Steps"])


log_abs = np.absolute(np.reshape(precision_log,(len(N_set),len(M_set))))
log_abs += 1e-16

x_steps = 2
y_steps = 1
tick_labely = N_set[0::y_steps]
tick_labelx = M_set[0::x_steps]
x_ticks = np.arange(0.5, len(M_set), x_steps)
y_ticks = np.arange(0.5, len(N_set), y_steps)

tickfont = 14
labelfont = 15

fig, ax = plt.subplots()
ax.set_ylabel("Neurons in first layer ($n^{(1)}$)", fontsize=labelfont)
ax.set_xlabel("Neurons in second layer ($n^{(2)}$)", fontsize=labelfont)
ax.set_xticklabels(tick_labelx, fontsize=tickfont)
ax.set_yticklabels(tick_labely, fontsize=tickfont)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
pcm = ax.pcolor(log_abs, norm=colors.LogNorm(vmin=1e-16, vmax=1.),
                cmap=cm.Spectral)
cbar = fig.colorbar(pcm, ax=ax, ticks=[10 ** (-i) for i in np.arange(1., 17., 2.)])
cbar.ax.tick_params(labelsize=tickfont)
cbar.ax.set_ylabel('Relative Error ($\\Delta E$)', rotation=90, labelpad=4, fontsize=labelfont)
fig.tight_layout()
fig.savefig(file["File name"][0:-12]+'LayerConfiguration.pdf')
