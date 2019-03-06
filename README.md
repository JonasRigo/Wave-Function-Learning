# ANN-WF

The ANN-WF repository gives the opportunity to investigate the ability of Artificial Neural Networks to approximate ground-state quantum wave functions. The ANN-multi-test.py script allows to test many different networks on the example of the 1D antiferromagnetic Heisenberg Hamiltonian. The file ANN-multi-test.py allows three types of batch gradient descent variants, but this has to be changed in the script variants. As standard method AdaGrad (see http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants for further information) is used. For now only batch GD is implemented, perhaps it might be extended to mini batch GD at a later point. The functionality of ANN-multi-test.py is best explained by walking thorught he input file, which we will do now.

## Config.json

ANN-multi-test.py needs an input file (Config.json) where one can specifiy the Fock space, the Hamiltonian, the way of testing and the network itself. Hereunder we can see an example for Config.json:
```Java
{
	"System":{
		"N": 8,
		"TotalSz": "0",
		"SignTransform": true
	},
	"Test":{
		"L_max": [11,11],
		"L_min": [9,9],
		"Epochs": 1e3,
		"Steps" : 1,
		"Precision": 8e-16,
    		"Repetitions": 5,
    		"Pre-Training": true
	},
	"Network":{
		"Name": "TDT",
		"Architecture":["Linear",0,"Tanh","Linear",1,"Triangle","Linear",0,"Tanh"],
		"Loss": "Energy"
	}
}
```
We begin with the settings of the system. First of all we set the system size N to any integer larger 0, but for the sake of your system memory you should not exceed "N": 16. Next one can restrict the Fock space to the TotalSz = 0 subspace by setting "TotalSz": "0", anything else yields the whole Fock space. Finally the a Marshalls Sign Transformation can be applied to the Hamiltonian by setting "SignTransform": true, if not desired just set "SignTransform": false.

The "Test" settings specify which layer sizes are tested. For a L layer network "L_max" and "L_min" have to be lists of legnth L-1, the Lth layer is preset to 1. The "Steps" variable defines how many steps are taken between L_max and L_min.To have just one configuration "L_min": [l_1,...l_(L-1)] and "L_max": [l_1+1,...l_(L-1)+1]. The "Epochs" setting specifies how many GD steps are iterated, and one can set a precision ("Precision") to, once reached, abort the optimization process. Every configuration can be tested "Repetitions"-many times. Eventually one can make use of unsupervised pre-training to initialize the network parameters by setting "Pre-Training": true. For a gaussain random distributed initialization set "Pre-Training": false.

"Network" is the part where we design the network. First of all one can give it a name "Name", which is then part of the output file name. "Architecture" is a list that has the following structure ["linear_operation",bias,"activation_function",...] which is repeated for every network layer. The linear_operation can be chosen to be:
* "Linear" for the standard affine operation,
* "Convolution" for a weighted cross sum,
* "lrf_Linear" for the standard affine operation, but with a sparse receptive field.
The bias variable is an integer and can be set to 1 in order to use bias and 0 in order to not.
The activation_function can be chosen to be:
* "ReLu" for the rectified linear unit,
* "Tanh",
* "Triangle" for the triangle signal function,
* "Sigmoid",
* "Sinc",
* "Cos",
* "Softmax",

see https://en.wikipedia.org/wiki/Activation_function for details on the activation functions. Finally one has to choose a cost-function, for which we have only one option the variational energy "Loss": "Energy". 

## Output

The output file contains in order the following lists:
* the configuration set as calculated from "Test",
* the lowest relative errors (calculated as (E_ann - E_ed)/E_ed, where E_ed the energy obtained form exact diagonalization and E_ann the result from the network) for every configuration,
* all the relative errors for every configuration as obtained from every repetitions,
* the weight matrices corresponding to the lowest relative erros for every configuration,
* the learning curve contains the relative error and variational energy of every optimization step for every configurations lowest final relative error.

The name of the ouput file is composed form the system settings
* AFH-pm no sign transformation
* AFH-p sign transformation
* AFH-Sz0-... restricted Fock space
* PreTrain-True/False denotes whether pre-training was used or not.

## Plot.josn

The Plot.josn file contains the file name one wants to plot:
```Java
{
	"File name": "TDT_AFH-Sz0-p_N8_PreTrain-True_PlotFile.npz"
}
```
There are three plot options so far. One can plot all the relative errors for every configuration as histogram with Histogram_Plot.py. In a similar fashion one can plot the learning curves for every configuration with LearningCurve_Plot.py. Only for three layer networks we have the LayerConfiguration_Plot.py option, which plots the lowest obtained relative error for every configuration in a color scheme on a 2D grid.
