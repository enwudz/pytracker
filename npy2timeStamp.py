#!/usr/bin/python

import numpy as np
import glob
from datetime import datetime, timedelta
from matplotlib import dates
import sys

## this takes a LONG time for multi-day datasets. Whew!
#data = analysisTools.loadData('xy16')

def concatenateData(filenames): 
	print('Concatenating Data')
	firstFile = filenames.pop(0)
	data = np.load(firstFile)
    
	if len(filenames) > 0:
		for f in filenames:
			d = np.load(f)
			data = np.vstack((data,d))
	return data
	
def findDataFiles(searchType):
	print('\nLooking for .npy files ... ') 
	filenames = glob.glob(searchType+'*.npy')
	if len(filenames) == 0: # checks to see if there are any files
		sys.exit('No .npy files to analyze')
	return sorted(filenames)

def main(searchType):
	# find the .npy files for this experiment		
	filenames = findDataFiles('xy16')

	# convert the filenames to the times at which they were saved
	saveTimes = [datetime(int('20'+x[2:4]),int(x[4:6]),int(x[6:8]),int(x[9:11]),int(x[11:13])) for x in filenames]

	# find the intervals between save times (needed to estimate experiment start time if no frame1)
	intervals = [(saveTimes[x+1]-saveTimes[x]).seconds for x in range(len(saveTimes)-1)]
	avgSaveInterval = (np.round(np.mean(intervals)))

	#find experiment start time
	try: 
		bf = glob.glob('frame1-16*.pnffffg')[-1]
		y = int('20'+bf[7:9])
		m = int(bf[9:11])
		d = int(bf[11:13])
		H = int(bf[14:16])
		M = int(bf[16:18])
		S = int(bf[18:20])
		startTime = datetime(y,m,d,H,M,S)
	except:
		startTime = saveTimes[0] - timedelta(seconds=avgSaveInterval)
	
	# find the average fps
	# how many frames in first .npy file?
	d = np.load(filenames[0])
	numFrames = np.shape(d)[0]
	fpsInIntervals = [numFrames/float(x) for x in intervals]
	avgfps = np.mean(fpsInIntervals)

	# when did files start saving? take time of last save and add average secs / frame
	fileStarts = [x + timedelta(seconds=1/avgfps) for x in saveTimes]

	# insert startTime into beginning of fileStarts, and remove the last start
	fileStarts.insert(0,startTime)
	fileStarts.pop(-1)

	# make the timeStamps for this experiment
	# for each fileStart, start at that time, and space the time increments evenly over
	# numFrames in that file to the saveTime of that file
	timeStamps = []
	for f in range(len(filenames)):
		d = np.load(filenames[f])
		numFrames = np.shape(d)[0]
		startTimeDatenum = dates.date2num(fileStarts[f])
		endTimeDatenum = dates.date2num(saveTimes[f])
		if startTimeDatenum > endTimeDatenum: # debugging
			exit('trouble with timeTravel!') 
		timeRange = np.linspace(startTimeDatenum,endTimeDatenum,numFrames)
		timeStamps.extend(timeRange)

	print('\nExperiment Start:')
	print(dates.num2date(timeStamps[0]))
	print('Experiment End:')
	print(dates.num2date(timeStamps[-1]))

	# concatenate the data, add the timestamps, and save as 'xyDataWithTimeStamps.npy'

	# concatenate data ... takes a LONG time if there is lots of data
	print('\nConcatenating the data (may take awhile . . . )')
	data = np.load(filenames.pop(0))
	for f in filenames:
		d = np.load(f)
		data = np.vstack((data,d))

	# insert the timestamps into the data
	print('\nAdding the timestamps')
	data = np.insert(data,0,timeStamps,axis=1)

	# save the data!
	timestampeddata = 'xyDataWithTimeStamps.npy'
	print('Saving %s' % timestampeddata)
	np.save(timestampeddata,data)

if __name__ == '__main__':
	main(searchType)

