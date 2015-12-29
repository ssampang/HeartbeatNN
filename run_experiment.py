import cPickle, lasagne, theano, theano.tensor as T
from HBNet import HBNet
import numpy as np

trainingFileName = 'BinaryTrain80.pkl'
testFileName = 'BinaryTest20.pkl'
cross_val_fold_size = 4
num_epochs = 500
learning_rate = 0.01
lr_decay = (learning_rate - 0.001)/num_epochs
momentum = 0.9
reg_lambda = 0.005

def loadData(filename):
	f = open(filename)
	return cPickle.load(f)

def addLayer(network,layer,name):
	network[name] = layer
	print lasagne.layers.get_output_shape(layer)

def build_cnn(input_var=None):
	network = {}
	addLayer(network, lasagne.layers.InputLayer(shape=(None,1,361,2), input_var = input_var), 'Input')
	addLayer(network, lasagne.layers.Conv2DLayer(network['Input'],num_filters=50,filter_size=(30,1)), 'Conv1')
	addLayer(network, lasagne.layers.MaxPool2DLayer(network['Conv1'], pool_size=(5,1)), 'MaxPool1')
	addLayer(network, lasagne.layers.DenseLayer(lasagne.layers.dropout(network['MaxPool1'], p=0.5), num_units=100), 'FC1')
	addLayer(network, lasagne.layers.DenseLayer(network['FC1'],num_units=1,nonlinearity = lasagne.nonlinearities.sigmoid),'Sigmoid')
	return (network, network['Sigmoid'], [network['Conv1'], network['FC1']])

def modify_cnn(network):
	addLayer(network, lasagne.layers.DenseLayer(lasagne.layers.dropout(network['FC1'], p=0.2), num_units=300), 'FC2')
	addLayer(network, lasagne.layers.DenseLayer(network['FC2'],num_units=1,nonlinearity = lasagne.nonlinearities.sigmoid),'Sigmoid')
	return (network, network['Sigmoid'], [network['Conv1'], network['FC1'], network['FC2']])



print('Loading data...')
X_train, y_train = loadData(trainingFileName)
X_test, y_test = loadData(testFileName)
print('Done loading data')

X_train = np.swapaxes(X_train,2,3)
X_test = np.swapaxes(X_test,2,3)

training_sets = []
num_examples = len( X_train )
indices = [ num_examples*i/4 for i in range(cross_val_fold_size)]
for i in range( len(indices) ):
	if i < len(indices)-1:
		training_sets.append( (X_train[ indices[i] : indices[i+1] ], y_train[ indices[i] : indices[i+1] ]) )
	else:
		training_sets.append( (X_train[ indices[i] : ], y_train[ indices[i] : ]) )

for i in range(10):
	reg_lambda = 0.001*(1+i)
	input_var = T.tensor4('inputs')
	network, output_layer, reg_layers = build_cnn( input_var)

	net = HBNet( 'CNN', network, output_layer, input_var, reg_layers, learning_rate, lr_decay, momentum, reg_lambda, training_sets)
	net.train(num_epochs, True) 
