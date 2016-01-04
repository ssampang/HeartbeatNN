import cPickle, time, random, operator, os, math, numpy as np
import lasagne, theano, theano.tensor as T

#learning_rate = 0.01
#momentum = 0.9

class HBNet:

	batch_size = 256

	def __init__(self, name, network, output_layer, input_var, reg_layers, learning_rate, lr_decay, momentum, reg_lambda, training_sets):
		self.name = name
		self.network = network
		self.output_layer = output_layer
		self.original_weights = lasagne.layers.get_all_param_values(self.output_layer)
		self.reg_layers = reg_layers
		self.original_learning_rate = np.array(learning_rate, dtype = theano.config.floatX)
		self.learning_rate = theano.shared( self.original_learning_rate )
		self.lr_decay = np.array(lr_decay, dtype = theano.config.floatX)
		self.momentum = momentum
		self.reg_lambda = reg_lambda
		self.training_sets = training_sets

		self.input_var = input_var
		self.target_var = T.imatrix('targets')
		self.dir_name = self.name + '-' + str(self.reg_lambda).replace('.','_')
		self.train_fn, self.val_fn = self.setup_functions()

	def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
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

	def save_network(self, filename,param_values):
		f = file(self.dir_name + '/' + filename, 'wb')
		cPickle.dump(param_values,f,protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()

	def setup_functions(self):
		prediction = lasagne.layers.get_output(self.output_layer)
		loss = lasagne.objectives.binary_crossentropy(prediction,self.target_var)
		loss = loss.mean()
		regularization = lasagne.regularization.regularize_layer_params(self.reg_layers, lasagne.regularization.l2)
		loss = loss + self.reg_lambda*regularization

		params = lasagne.layers.get_all_params(self.output_layer, trainable=True)
		updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate = self.learning_rate, momentum=self.momentum)

		test_prediction = lasagne.layers.get_output(self.output_layer, deterministic=True)
		test_loss = lasagne.objectives.binary_crossentropy(test_prediction,self.target_var)
		test_loss = test_loss.mean()
		test_acc = T.mean(T.eq(T.gt(test_prediction,0.5),self.target_var),dtype=theano.config.floatX)

		train_fn = theano.function([self.input_var,self.target_var],loss, updates=updates)
		val_fn = theano.function([self.input_var,self.target_var],[test_loss,test_acc])
		return train_fn, val_fn

	def train(self, num_epochs, save_best_network):

		os.system('mkdir ' + self.dir_name)
		print 'Training ' + self.name + ' with regularization lambda ' + str(self.reg_lambda)
		stats = [ [] for i in range( len(self.training_sets) )]
		for i in range( len(self.training_sets) ):
			print 'Cross validation fold ' + str(i+1)
			X_val, y_val = self.training_sets[i]
			X_train = [ self.training_sets[j][0] for j in range( len(self.training_sets) ) if j!=i]
			X_train = reduce(lambda x,y: np.concatenate( (x,y) ) , X_train, [])
			y_train = [ self.training_sets[j][1] for j in range( len(self.training_sets) ) if j!=i]
			y_train = reduce(lambda x,y: np.concatenate( (x,y) ) , y_train, [])
			
			self.learning_rate.set_value( self.original_learning_rate )
			lasagne.layers.set_all_param_values(self.output_layer, self.original_weights)

			try:	
				best_params = []
				best_val_acc = 0.0
				total_val_batches = math.ceil(len(X_val)/self.batch_size)
				best_epoch = 0
				epoch = 0
				training_start_time = time.time()
				run_forever = False
				if num_epochs == -1:
					run_forever = True
				while(epoch < num_epochs or run_forever):
					train_err = 0
					train_batches = 0
					start_time = time.time()
					for batch in self.iterate_minibatches(X_train,y_train,min(self.batch_size,len(X_train)), True):
						inputs,targets = batch
						train_err += self.train_fn(inputs,targets)
						train_batches += 1

					val_err = 0
					val_acc = 0
					val_batches = 0
					for batch in self.iterate_minibatches(X_val,y_val,min(self.batch_size,len(X_val))):
						inputs,targets = batch
						err,acc = self.val_fn(inputs,targets)
						val_err += err
						val_acc += acc
						val_batches += 1
					
					if epoch % 50 == 0 or epoch < 40:
						print("Epoch {} of {} took {:.3f}s training loss: {:.6f}\t validation loss: {:.6f}\t validation accuracy:{:.2f} %".format(epoch + 1, num_epochs, time.time() - start_time, train_err / train_batches, val_err / val_batches, val_acc / val_batches * 100))
					stats[i].append( (time.time() - training_start_time, train_err / train_batches, val_err / val_batches, val_acc / val_batches))
					if val_acc > best_val_acc:
						best_val_acc = val_acc
						best_params = lasagne.layers.get_all_param_values(self.output_layer)
						total_val_batches = val_batches
						best_epoch = epoch
					epoch += 1

					self.learning_rate.set_value( max( np.array(0.001, dtype = theano.config.floatX) , self.learning_rate.get_value() - self.lr_decay) )
					#if epoch % 1000 ==0:
					#	self.save_network('network' + self.name + '-' + str(epoch)+'.pkl',lasagne.layers.get_all_param_values(output_layer))
			finally:
				print 'Best network from epoch '+str(best_epoch) + ' with validation accuracy '+str( best_val_acc / total_val_batches * 100)
				print 'Total elapsed time for this fold is ' + str( time.time() - training_start_time)
				if save_best_network:
					print 'Saving best network...'
					self.save_network('network-' + self.name + '-' + 'cross_val-' + str(i) + '-' +str(best_epoch)+'.pkl',best_params)
		saveStats = open(self.dir_name + '/stats.pkl', 'wb')
		cPickle.dump(stats, saveStats)
		saveStats.close()
		#res = open(self.dir_name + '/res.txt', 'wb')
		#res.write('Best network from epoch '+str(best_epoch) + ' with validation accuracy '+str( best_val_acc / total_val_batches * 100))
		#res.close()
