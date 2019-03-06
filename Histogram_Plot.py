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
histogram_log = npzfile['arr_2']

tickfont = 13
labelfont = 14
bins = 10.**np.arange(-16.,2.,1.)

with PdfPages(file["File name"][0:-12]+'Histogram.pdf') as pdf:
    for i in range(len(histogram_log)):
        fig, ax = plt.subplots()
        fig.suptitle('Layer-configuration: '+str(config_set[i]), fontsize=labelfont)
        pyplot.hist(histogram_log[i], bins,color = '#1f77b4',rwidth=0.9 , log = False)#, label='bias = False')
        pyplot.xscale('log')
        #pyplot.legend(loc='upper left', fontsize = tickfont)
        pyplot.yticks(fontsize = tickfont)
        pyplot.xticks(fontsize = tickfont)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Counts", fontsize = labelfont)
        ax.set_xlabel("Relative error ($\\Delta E$)", fontsize = labelfont)
        pdf.savefig()
        plt.close()
