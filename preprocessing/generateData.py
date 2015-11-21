import os
import matplotlib.pyplot as plt
import cPickle

records = [100,101,102,103,104,105,106,107,108,109,111,112,113,114,115,116,117,118,119,121,122,123,124,200,201,202,203,205,207,208,209,210,212,213,214,215,217,219,220,221,222,223,228,230,231,232,233,234]

def generateHist(data):
	global intervals
	for datum in data:
		waves,annotations = datum
		for i in range(len(annotations)-3):
			intervals.append( annotations[i+2][0] - annotations[i+1][0] )
	plt.hist(intervals,bins=100, range=(0,3) )
	plt.title('Intervals between beat annotations in seconds')		
	plt.xlabel('Seconds')
	plt.ylabel('Counts')
	plt.show()

def textToData():
 	data = []
	for record in records:
		if not os.path.exists('annotations/'+str(record)+'.txt'):
			os.system('/home/sid/Projects/HeartbeatNN/wfdb-10.5.24/build/bin/rdann -r mitdb/'+str(record)+' -f 0 -t 1805.556 -a atr -x >annotations/'+str(record)+'.txt')
		
		f = file('annotations/'+str(record)+'.txt')
		text = f.read().split('\n')
		annotations = [[y for y in x.split() if y!=''] for x in text[:-1]]
		annotations = [ [float(x[0]),x[3]] for x in annotations]

		if not os.path.exists('waveforms/'+str(record)+'.csv'):
			os.system('/home/sid/Projects/HeartbeatNN/wfdb-10.5.24/build/bin/rdsamp -r mitdb/'+str(record)+' -c -H -f 0 -t 1805.556 -v -p >waveforms/'+str(record)+'.csv')
		
		f.close()
		f = file('waveforms/'+str(record)+'.csv')
		text = f.read().split('\n')
		waves = [ [float(y) for y in x.split(',')] for x in text[2:-1]]	
		data.append((waves,annotations))
		f.close()
	
	savefile = open('intermediateData.pkl','wb')
	cPickle.dump(data,savefile)
	savefile.close()

def dataToSamples(intData, windowSize):
	data = []
	ind = 0
	for datum in intData:
		waves,annotations = datum	
		waveIndex = 0
		
		for annotation in annotations:
			temp = []
			checkTime = lambda x: abs(x[0]-annotation[0]) <= windowSize/2
			#samples = filter( lambda x: abs(x[0]-annotation[0]) < windowSize, waves)
			#data.append( (samples,annotation) )	
			while waveIndex>0 and waves[waveIndex-1][0] >= annotation[0]-windowSize/2:
				waveIndex = waveIndex-1

			while not waveIndex == len(waves) and waves[waveIndex][0] < annotation[0]-windowSize/2:
				waveIndex = waveIndex+1
			
			while not waveIndex == len(waves) and checkTime(waves[waveIndex]):
				temp.append(waves[waveIndex])
				waveIndex += 1

			data.append(( temp, annotation, records[ind] ))
		ind += 1
				
	savefile = open('data.pkl','wb')
	cPickle.dump(data,savefile)
	savefile.close()
	return data

if not os.path.exists('intermediateData.pkl'):
	textToData()

f = file('intermediateData.pkl')
intermediateData = cPickle.load(f)
f.close()
#intervals = []
#generateHist(data)

data = dataToSamples(intermediateData,1.0)
