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
labels = {'!': 472, '"': 437, '+': 1244, '/': 7028, 'A': 2546, 'E': 106, 'F': 803, 'J': 83, 'L': 8075, 'N': 75052, 'Q': 33, 'S': 2, 'R': 7259, 'V': 7130, '[': 6, ']': 6, 'a': 150, 'e': 16, 'f': 982, 'j': 229, 'x': 193, '|': 132, '~': 615}
labels = labels.keys()
labels = [x for x in labels if x not in remove]
label = 0
labelDict = {}
for i in labels:
	labelDict[i] = label
	label += 1

abnormals = [x for x in abnormals if x[1][1] not in remove]
random.shuffle(normals)
random.shuffle(abnormals)
X_val = []
y_val = []
X_val.extend(normals[:len(normals)/2])
X_val.extend(abnormals[:len(abnormals)/2])
y_val = np.array([ labelDict[x[1][1]] for x in X_val], dtype='int32')
del normals[:len(normals)/2]
del abnormals[:len(abnormals)/2]
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
y_train = np.array([ labelDict[x[1][1]] for x in training], dtype='int32')
y_train = np.expand_dims(y_train,1)

f = open('AllLabelsTrain5050.pkl', 'wb')
cPickle.dump((X_train,y_train),f)
f.close()

f = open('AllLabelsVal5050.pkl','wb')
cPickle.dump((X_val,y_val),f)
f.close()
