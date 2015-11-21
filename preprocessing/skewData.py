import cPickle
import random
import numpy as np

f = open('../data.pkl')
data = cPickle.load(f)
f.close()
data = filter(lambda x: len(x[0])==361, data)
normals = filter( lambda x: x[1][1]=='N', data)
abnormals = filter( lambda x: x[1][1]!='N', data)
del data
remove = ['!','"','+','Q','[',']','x','|','~']
abnormals = [x for x in abnormals if x[1][1] not in remove]
random.shuffle(normals)
random.shuffle(abnormals)
X_val = []
y_val = []
X_val.extend(normals[:5000])
y_val = [1 for x in range(5000)]
X_val.extend(abnormals[:5000])
y_val.extend([0 for x in range(5000)])
del normals[:5000]
del abnormals[:5000]
X_val = np.array([ np.array([y[1:] for y in x[0]]).T for x in X_val],dtype='float32')
X_val = np.expand_dims(X_val,1)
y_val = np.array(y_val,dtype='int32')
y_val = np.expand_dims(y_val,1)

moreCopies = ['S', 'e', 'J', 'E', 'a', 'j', 'F']
training = []
for el in abnormals:
	if el[1][1] in moreCopies:
		training.extend([el for i in range(7)])
	else:
		training.extend([el for i in range(2)])
training.extend(normals)
random.shuffle(training)
X_train = np.array([ np.array([y[1:] for y in x[0]]).T for x in training], dtype='float32')
X_train = np.expand_dims(X_train,1)
y_train = np.array([ int(x[1][1]=='N') for x in training], dtype='int32')
y_train = np.expand_dims(y_train,1)

f = open('BinaryTrain.pkl', 'wb')
cPickle.dump((X_train,y_train),f)
f.close()

f = open('BinaryVal.pkl','wb')
cPickle.dump((X_val,y_val),f)
f.close()
