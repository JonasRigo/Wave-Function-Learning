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
with open('Plot.json') as f:
    file = json.load(f)

npzfile = np.load(file["File name"])
config_set = npzfile['arr_0']
learning_log = npzfile['arr_4']

tickfont = 13
labelfont = 14
color = '#1f77b4'

with PdfPages(file["File name"][0:-12]+'LearningCurve.pdf') as pdf:
    for i in range(len(config_set)):
        fig, ax = plt.subplots()
        fig.suptitle('Layer-configuration: '+str(config_set[i]), fontsize=labelfont)
        ax.plot(np.absolute(learning_log[i,1]), label='Learning Curve', color=color)
        pyplot.xscale('log')
        pyplot.yticks(fontsize = tickfont)
        pyplot.xticks(fontsize = tickfont)
        ax.set_xlabel("Epoch", fontsize = labelfont)
        ax.set_ylabel("Relative error ($\\Delta E$)", fontsize = labelfont)
        pdf.savefig()
        plt.close()
