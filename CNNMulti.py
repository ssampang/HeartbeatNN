import cPickle, time, random, numpy as np
import lasagne, theano, theano.tensor as T

batch_size=256
learning_rate = 0.01
momentum = 0.9
trainingFileName = 'AllLabelsTrain5050.pkl'
valFileName = 'AllLabelsVal5050.pkl'


def loadData(filename):
	f = open(filename)
	return cPickle.load(f)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]

def save_network(filename,param_values):
	f = file(filename, 'wb')
	cPickle.dump(param_values,f,protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()

def load_network(filename):
	f = file(filename, 'rb')
	param_values = cPickle.load(f)
	f.close()
	return param_values

def addLayer(network,layer,name):
	network[name] = layer
	print lasagne.layers.get_output_shape(layer)

def build_cnn(input_var=None):
	network = {}
	addLayer(network, lasagne.layers.InputLayer(shape=(None,1,2,361), input_var = input_var), 'Input')
	addLayer(network, lasagne.layers.Conv2DLayer(network['Input'],num_filters=50,filter_size=(1,30)), 'Conv1')
	addLayer(network, lasagne.layers.MaxPool2DLayer(network['Conv1'], pool_size=(1,5)), 'MaxPool1')
	addLayer(network, lasagne.layers.DenseLayer(lasagne.layers.dropout(network['MaxPool1'], p=0.5), num_units=50), 'FC1')
	addLayer(network, lasagne.layers.DenseLayer(lasagne.layers.dropout(network['FC1'], p=0.5), num_units=50), 'FC2')
	addLayer(network, lasagne.layers.DenseLayer(network['FC2'],num_units=14,nonlinearity = lasagne.nonlinearities.softmax),'Softmax')
	return (network, network['Softmax'], [network['Conv1'], network['FC1'], network['FC2']])

def modify_cnn(network):
	addLayer(network, lasagne.layers.DenseLayer(lasagne.layers.dropout(network['FC1'], p=0.5), num_units=100), 'FC2')
	addLayer(network, lasagne.layers.DenseLayer(network['FC2'],num_units=14,nonlinearity = lasagne.nonlinearities.softmax),'Softmax')
	return (network, network['Softmax'], [network['Conv1'], network['FC1'], network['FC2']])
	
def setupFunctions(outputLayer, regLayers, input_var, target_var):
	prediction = lasagne.layers.get_output(outputLayer)
	loss = lasagne.objectives.categorical_crossentropy(prediction,target_var)
	loss = loss.mean()
	#reg = lasagne.regularization.regularize_layer_params(regLayers, lasagne.regularization.l2)
	#loss = loss + reg

	params = lasagne.layers.get_all_params(outputLayer, trainable=True)
	updates = lasagne.updates.nesterov_momentum(loss,params,learning_rate=learning_rate, momentum=momentum)

	#only useful if I eventually use dropout
	test_prediction = lasagne.layers.get_output(outputLayer, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
	test_loss = test_loss.mean()
	test_acc = T.mean(T.eq(T.argmax(test_prediction,axis=1),target_var),dtype=theano.config.floatX)

	train_fn = theano.function([input_var,target_var],loss, updates=updates)
	val_fn = theano.function([input_var,target_var],[test_loss,test_acc])
	return train_fn, val_fn

def train(numEpochs, outputLayer):
#	try:	
	bestParams = []
	best_val_acc = 0.0
	total_val_batches = 0
	best_epoch = 0
	epoch = 1
	runForever = False
	if numEpochs == -1:
		runForever = True
	while(epoch < numEpochs or runForever):
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train,y_train,batch_size):
			inputs,targets = batch
			train_err += train_fn(inputs,targets)
			train_batches += 1

		val_err = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val,y_val,min(batch_size,len(X_val))):
			inputs,targets = batch
			err,acc = val_fn(inputs,targets)
			val_err += err
			val_acc += acc
			val_batches += 1

		print("Epoch {} of {} took {:.3f}s training loss: {:.6f}\t validation loss: {:.6f}\t validation accuracy: {:.2f} %".format(epoch + 1, numEpochs, time.time() - start_time, train_err / train_batches, val_err / val_batches, val_acc / val_batches * 100))
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			bestParams = lasagne.layers.get_all_param_values(outputLayer)
			total_val_batches = val_batches
			best_epoch = epoch
		epoch += 1
		if epoch % 1000 ==0:
			save_network('network-'+str(epoch)+'.pkl',lasagne.layers.get_all_param_values(outputLayer))
	#finally:
#		print 'Saving network from epoch ' + str(best_epoch) + ' with validation accuracy '+str( best_val_acc / total_val_batches * 100)
#		save_network('network-'+str(best_epoch)+'.pkl',bestParams)

print('Loading data...')
X_train, y_train = loadData(trainingFileName)
X_val, y_val = loadData(valFileName)
train_fn,val_fn = [],[]
print('Done loading data')

def main(numEpochs=-1):
	global train_fn,val_fn
	input_var = T.tensor4('inputs')
	target_var = T.imatrix('targets')
	
	network, outputLayer, regLayers = build_cnn(input_var)
	#params = load_network('networks/networkFC1-1934.pkl')
	#lasagne.layers.set_all_param_values(outputLayer,params)
	#network, outputLayer, regLayers = modify_cnn(network)
	train_fn,val_fn = setupFunctions(outputLayer, regLayers, input_var, target_var)
	#for epoch in range(num_epochs):
	train(numEpochs, outputLayer)

if __name__ == '__main__':
	main()
