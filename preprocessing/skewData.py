import cPickle
import random
import numpy as np
from math import floor

f = open('data.pkl')
data = cPickle.load(f)
f.close()
beatsMissing1Sample = filter(lambda x: len(x[0])==360, data)
for beat in beatsMissing1Sample:
	beat[0].append(beat[0][-1])
data = filter(lambda x: len(x[0])==361, data)
data.extend(beatsMissing1Sample)
normals = filter( lambda x: x[1][1]=='N', data)
abnormals = filter( lambda x: x[1][1]!='N', data)
del data

labels = ['A', 'E', 'J', 'L', 'N', 'R', 'a', 'e', 'j', 'S']
labelNum = 0
labelDict = {}
for i in labels:
	labelDict[i] = labelNum
	labelNum += 1

abnormals = [x for x in abnormals if x[1][1] in labels]
random.shuffle(normals)
random.shuffle(abnormals)
X_test = []
y_test = []
test_length = int(floor( len(abnormals)*0.2 ))
X_test.extend(normals[-test_length:])
X_test.extend(abnormals[-test_length:])
y_test = np.array([ int(x[1][1]=='N') for x in X_test], dtype='int32')
del normals[-test_length:]
del abnormals[-test_length:]
X_test = np.array([ np.array([y[1:] for y in x[0]]).T for x in X_test],dtype='float32')
X_test = np.expand_dims(X_test,1)
y_test = np.array(y_test,dtype='int32')
y_test = np.expand_dims(y_test,1)

normals = normals[:len(abnormals)]
training = normals
training.extend(abnormals)

X_train = np.array([ np.array([y[1:] for y in x[0]]).T for x in training], dtype='float32')
X_train = np.expand_dims(X_train,1)
y_train = np.array([ int(x[1][1]=='N') for x in training], dtype='int32')
y_train = np.expand_dims(y_train,1)

X_train = np.swapaxes(X_train,2,3)
X_test = np.swapaxes(X_test,2,3)

f = open('BinaryTrain80.pkl', 'wb')
cPickle.dump((X_train,y_train),f)
f.close()

f = open('BinaryTest20.pkl','wb')
cPickle.dump((X_test,y_test),f)
f.close()
