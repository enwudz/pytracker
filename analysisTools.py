#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import trackingTools
import colorsys
from matplotlib import gridspec, cm, dates
import math
import glob
from datetime import datetime, timedelta
import csv
import sys
from bisect import bisect_left
from scipy import stats

'''
to do:
'''

# these functions added 20200823 to deal with time gaps before binning
def getSteps(vec):
    return np.array([vec[i+1]-x for i,x in enumerate(vec[:-1])])
def getBigStepsByTime(vec,secs=1):
    steps = getSteps(vec)
    timeThresh = secs * 1/(24.0*60*60)
    indices = list(np.where(steps > timeThresh)[0])
    return [h+1 for h in indices]
def getBigSteps(vec,zthresh=5): # can change stringency here.
    steps = getSteps(vec)
    z = np.abs(stats.zscore(steps))
    indices = list(np.where(z > zthresh)[0])
    return [h+1 for h in indices]
def getVectorSubsets(vec,indices):
    subsets=[]
    if len(indices) > 0:
        indices.insert(0,0)
        for i,ind in enumerate(indices):
            if i < len(indices)-1:
                subsets.append(vec[ind:indices[i+1]])
            else:
                subsets.append(vec[ind:])
    else:
        subsets = [vec]
    return subsets
def getChunks(vec):
    indices = getBigSteps(vec)
    return getSubsets(vec,indices)
def checkDataForTimeGaps(data):
    timeStamps = data[:,0]
    indices = getBigStepsByTime(timeStamps,3)
    return indices
def getMatrixSubsets(mat,indices):
    subsets=[]
    if len(indices) > 0:
        indices.insert(0,0)
        for i,ind in enumerate(indices):
            if i < len(indices)-1:
                subsets.append(mat[ind:indices[i+1],:])
            else:
                subsets.append(mat[ind:,:])
    else:
        subsets = [mat]
    return subsets

# concatenate all data (in .npy files) into a single array
# dataType is 'xy' or 'dpix' or 'xyData' or path
def loadData(dataType='xy'):
	print('Loading .npy files')
	filenames = sorted(glob.glob(dataType+'*.npy'))

	if len(filenames) > 0: # checks to see if there are any files
		firstFile = filenames.pop(0)
	else:
		sys.exit('No .npy files to analyze')

	data = np.load(firstFile)

	if len(filenames) > 0:
		for f in filenames:
			d = np.load(f)
			data = np.vstack((data,d))
	return data

# load data after a set time point
# added 20210111 to deal with adding new data
def loadDataFromStartTime(startTime, dataType = 'xy'):

    startYMD = int(startTime.split('-')[0])
    startHMS = int(startTime.split('-')[1])

    print('Loading .npy files after ' + startTime)
    filenames = sorted(glob.glob(dataType+'*.npy'))

    if len(filenames) > 0: # checks to see if there are any files
        firstFile = filenames.pop(0)
    else:
        sys.exit('No .npy files to analyze')

    data = []
    for file in filenames:
        fileTimeStamp = file.split(dataType)[1].split('.')[0]
        fileYMD = int(fileTimeStamp.split('-')[0])
        fileHMS = int(fileTimeStamp.split('-')[1])

        if fileYMD >= startYMD and fileHMS > startHMS:
            d = np.load(file)
            if len(data) > 0:
                data = np.vstack((data,d))
            else:
                data = np.load(file)
    if len(data) == 0:
        sys.exit('No .npy files to analyze')
    else:
        return data

def concatenateCsv(searchString):
    import shutil
    import os

    filenames = sorted(glob.glob(searchString+'*.csv'))
    outFile = searchString + '_concatenated.csv'
    with open(outFile,'wb') as wfd:
        for f in filenames:
            print('... adding ' + f + ' to ' + outFile)
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)

# get time stamp of last .npy file
def lastTimeStamp(dataType = 'xy'):
    filenames = sorted(glob.glob(dataType+'*.npy'))
    lastfile = filenames[-1]
    lastFileSave = lastfile.split(dataType)[1].split('.')[0]
    return lastFileSave

# load information about the analyzed video
def loadVidInfo():
	vidInfo = {}
	for key,val in csv.reader(open('vidInfo.csv')):
		vidInfo[key] = val
	return vidInfo

# save data as csv (numbers only!)
def saveDataAsCSV(data,filename):
	np.savetxt(filename, data, delimiter=",", fmt='%10.2f')

def getElapsedTimeInSeconds(timeVec):
	return (timeVec[-1]-timeVec[0]) * (24*60*60)

def getAvgMotion(dpixData,roiNames,roiGroups):
	secs = getElapsedTimeInSeconds(dpixData[:,0])
	boxData = []
	for r in range(len(roiNames)):
		totalMotion = np.sum(dpixData[:,roiGroups[r]],0)
		avgMotion = totalMotion / float(secs)
		boxData.append(avgMotion)
		if len(roiNames) > 1:
			print(roiNames[r])
		for a in avgMotion:
			print(a)
	return boxData

# decide best way to bin the data
def findOptimalBinSize(timeVec):

	# find elapsed time in seconds
	s = getElapsedTimeInSeconds(timeVec)

	if s < 60:
		binSizeInSeconds = 1
	elif s < 60 * 5:
		binSizeInSeconds = 10
	elif s < 60 * 20:
		binSizeInSeconds = 30
	elif s < 60 * 60 * 6:
		binSizeInSeconds = 60
	elif s < 60 * 60 * 8:
		binSizeInSeconds = 60 * 2
	elif s < 60 * 60 * 10:
		binSizeInSeconds = 60 * 5
	elif s < 60 * 60 * 12:
		binSizeInSeconds = 60 * 10
	elif s < 60 * 60 * 24:
		binSizeInSeconds = 60 * 20
	else:
		binSizeInSeconds = 60 * 30

	if binSizeInSeconds <= 120:
		u = 'seconds'
	else:
		u = 'minutes'

	return binSizeInSeconds, u

# bin the data into larger chunks of time
def binData(data,binSizeInSeconds):

	# first column of data is time
	# other columns are either dPix data or distance data for each ROI
	timeVec = data[:,0]
	roiData = data[:,1:]

	# what is total time span in seconds?
	print('Getting elapsed time in seconds . . . ')
	timeSpan = getElapsedTimeInSeconds(timeVec)

	# how many COMPLETE bins of binSizeInSeconds are in the total timespan?
	numBins = int(math.floor(timeSpan / binSizeInSeconds))

	# convert the timeVec to dateTime
	#timeVec = [dates.num2date(x,tz=None) for x in timeVec]

	# make a vector of time intervals, based on binSizeInSeconds
	print('Getting bin starts . . . ')
	binStartTimes = [timeVec[0] + binSizeInSeconds * i/ (24*60*60) for i in range(numBins)]

	# binStarts = the indices of timeVec that match these time intervals
	binStarts = [bisect_left(timeVec, b) for b in binStartTimes]

	# make new timeVec from original timeVec based on these indices
	binnedTimeVec = [timeVec[b] for b in binStarts]
	binnedTimeVec = binnedTimeVec[:-1] # last bin omitted b/c incomplete

	# initialize space for binnedData
	binnedData = np.zeros([len(binStarts)-1,np.shape(roiData)[1]])

	# sum the roi Data in each bin
	for b in range(len(binStarts)-1): # last bin omitted b/c incomplete
		binnedData[b,:] = np.sum(roiData[binStarts[b]:binStarts[b+1],:],axis=0)

	# convert timeVec back to numbers
	#binnedTimeVec = [dates.date2num(x) for x in binnedTimeVec]

	return binnedData, binnedTimeVec

#### Plotting tools

# choose plot format for time, appropriate to length of experiment
def selectTimeFormat(elapsedSeconds):
	if elapsedSeconds < 2 * 60:
		fmt, units, startZero = '%S', 'secs', 1
	elif elapsedSeconds < 120 * 60:
		fmt, units, startZero = '%M', 'minutes', 1
	else:
		fmt, units, startZero = '%H:%M', 'hours', 0
	return fmt, units, startZero

# N colors distributed evenly across a color map
def make_N_colors(cmap_name, N):
	cmap = cm.get_cmap(cmap_name, N)
	colors = [list(cmap(i)) for i in np.linspace(0, 1, N)]
	# cmap = cmap(np.arange(N))[:,0:3]
	# cmap = np.fliplr(cmap)
	#return [[i][0] for i in cmap]
	return colors

# standard error of the mean
def stde(mat,ax):
	s = np.std(mat, ax)
	se = s / np.sqrt(np.shape(mat)[1])
	return se

# format colors of a boxplot object
def formatBoxColors(bp, boxColors):

	baseWidth = 3
	for n,box in enumerate(bp['boxes']):
		box.set( color=boxColors[n], linewidth=baseWidth)

	for n,med in enumerate(bp['medians']):
		med.set( color=boxColors[n], linewidth=baseWidth)

	bdupes=[]
	for i in boxColors:
		bdupes.extend([i,i])

	boxColors = bdupes
	for n,whisk in enumerate(bp['whiskers']):
		#whisk.set( color=(0.1,0.1,0.1), linewidth=2, alpha = 0.5)
		whisk.set( color=boxColors[n], linewidth=baseWidth, alpha = 0.5)

	for n,cap in enumerate(bp['caps']):
		cap.set( color=boxColors[n], linewidth=baseWidth, alpha = 0.5)

	return bp

# find day / night boundaries from a timeStamps vector
def findDayNightInTimeStamps(timeStamps,lightsON,lightsOFF):
	# get day, time of first timeStamp
	d = int(timeStamps[0])
	t = timeStamps[0] - d

	# add this time to a list of startTimes for this experiment
	dnStarts = [timeStamps[0]]
	dayornight = []

	lightsON = float(lightsON)/24
	lightsOFF = float(lightsOFF)/24

	finished = False
	while finished == False:

		if 0 <= t < lightsON: # it is NIGHT
			dayornight.append('n')
			nextEvent = d + lightsON
			if nextEvent < timeStamps[-1]:
				dnStarts.append(nextEvent)
				t = lightsON
			else:
				finished=True

		elif lightsOFF < t < 1: # it is NIGHT
			dayornight.append('n')
			nextEvent = d + lightsON
			if nextEvent < timeStamps[-1]:
				dnStarts.append(nextEvent)
				t = lightsON
				d = d + 1
			else:
				finished=True

		elif t == lightsOFF: # it is NIGHT
			dayornight.append('n')
			nextEvent = d + 1 + lightsON
			if nextEvent < timeStamps[-1]:
				dnStarts.append(nextEvent)
				t = lightsON
				d = d + 1
			else:
				finished=True

		elif t >= lightsON: # it is DAY
			dayornight.append('d')
			nextEvent = d + lightsOFF
			if nextEvent < timeStamps[-1]:
				dnStarts.append(nextEvent)
				t = lightsOFF
			else:
				finished=True

	if len(dnStarts) > 1:
		dnEnds = dnStarts[1:]
		dnEnds.append(timeStamps[-1])
	else:
		dnEnds = [timeStamps[-1]]

	return dnStarts, dnEnds, dayornight

# plot day / night boundaries on little skinny axes as a label for time vs. data plot
def plotdnd(timeForPlot,axeslist):
	# find the day night boundaries and identities
	dnstarts, dnends, dayornight = findDayNightInTimeStamps(timeForPlot,7,23)

	# for each startTime and endTime, find the time window for this part
	b = []
	e = []
	for s in range(len(dnstarts)):
		b.append(bisect_left(timeForPlot, dnstarts[s]))
		e.append(bisect_left(timeForPlot, dnends[s]))

	# show the day / night indicators for the time vs. distance plots
	if len(dayornight) > 1:

		for s in range(len(dayornight)):
			timeRange = timeForPlot[b[s]:e[s]+1]
			if dayornight[s] == 'n':
				fillColor = (0,0,0)
			else:
				fillColor = (1,1,1)

			for ax in axeslist:
				ax.fill_between(timeRange,0,1, facecolor=fillColor, edgecolor='k')
				ax.set_xlim([timeForPlot[0],timeForPlot[-1]])
				ax.axis('off')
	else:
		for ax in axeslist:
			ax.axis('off')

# ribbon plot from axis, time vector, data, color
def timeVdataRibbonPlot(ax,t,d,c): # axis, timeVec, data, color

	ax.plot(t,np.mean(d,axis=1),color=c)
	se = stde(d,1)
	ax.fill_between(t, np.mean(d,1)-se, np.mean(d,1)+se, alpha = 0.3, facecolor=c, edgecolor='')
	ax.set_xlim([t[0],t[-1]])
	ax.set_xticks([])
	#ax.xaxis.set_major_formatter( dates.DateFormatter('%M') )

# day / night activity boxplot from axis, time vector, colors, and roigroups
def makeribbonBoxPlot(ax,t,d,plotColors,roiGroups,roiNames):

	boxData = []

	# find the day night boundaries and identities
	dnstarts, dnends, dayornight = findDayNightInTimeStamps(t,7,23)

	# for each startTime and endTime, find the time window for this part
	b = []
	e = []
	for s in range(len(dnstarts)):
		b.append(bisect_left(t, dnstarts[s]))
		e.append(bisect_left(t, dnends[s]))

		for r in list(range(len(roiGroups))):
			roiName = roiNames[r]
			roiGroup = [ x-1 for x in roiGroups[r] ] # indexing
			dataForBox = d[b[s]:e[s],roiGroup]
			boxData.append(np.mean(dataForBox,axis=0))

	b1 = ax.boxplot(boxData, widths=0.5)
	boxLabels = roiNames * len(dnstarts) # need to have a set of names for each time period

	# format the boxplot colors
	plotColors = plotColors * len(dnstarts)
	b1 = formatBoxColors(b1,plotColors)

	# add a legend and label boxplots, if reasonable number of ROI
	if 1 < len(roiNames) <= 3 and 1 < len(dnstarts) < 3:
		ax.set_xticklabels(boxLabels, fontsize = 12)
	else:
		ax.set_xticklabels('')

	return boxData, dayornight

# label the day/night boxplots with black and white bars, on a little skinny axis
def labelDNDboxplots(ax,boxData,dayornight):
	boxToTimeRatio = len(boxData)/len(dayornight)
	dayornight = np.repeat(dayornight,boxToTimeRatio)
	offset = 0
	startRange = [0.5,1.5]
	for s in range(len(dayornight)):
		xrange = [x + offset for x in startRange]
		if dayornight[s] == 'n':
			fillColor = (0,0,0)
		else:
			fillColor = (1,1,1)
		ax.fill_between(xrange,0,1, facecolor=fillColor, edgecolor='')
		offset += 1
	ax.set_xlim([startRange[0],xrange[1]])
	ax.axis('off')

#### quality control
#plot average fps throughout experiment
def plotFps(timeStamps):
	frameWindow = 100
	starts = range(0,len(timeStamps),frameWindow)
	timeStamps = [dates.num2date(x) for x in timeStamps]
	timeDiffs = [float(frameWindow)/ (timeStamps[s+frameWindow]-timeStamps[s]).seconds for s in starts[:-1] ]
	#  + ((timeStamps[s+frameWindow]-timeStamps[s]).microseconds)/1000000)
	timeStamps = [timeStamps[s] for s in starts][:-1]
	ax1=plt.subplot(111)
	ax1.plot(timeStamps,timeDiffs)
	ax1.xaxis.set_major_formatter( dates.DateFormatter('%H:%M') )
	ax1.set_ylim([0,30])
	plt.show()

#### stats
def statsFromBoxData(boxData,statTest):
	from scipy import stats
	pvals = []
	for i in range(len(boxData)):
		for j in range(i+1,len(boxData)):
			if statTest in ['k','kruskal','kruskalwallis','kw']:
				_,p = stats.kruskal(boxData[i],boxData[j])
				print('%i vs. %i: %1.3f' % (i+1,j+1,p))
				pvals.append(p)
			if statTest in ['t','ttest','t-test']:
				_,p = stats.ttest_ind(boxData[i],boxData[j])
				print('%i vs. %i: %1.3f' % (i+1,j+1,p))
				pvals.append(p)
	print('')
#
#
# 	numComps = len(boxData)/2
# 	pvals = np.zeros(numComps)
# 	j=0
# 	for i in range(0,len(boxData),2):
# 		if statTest in ['k','kruskal','kruskalwallis','kw']:
# 			_,p = stats.kruskal(boxData[i],boxData[i+1])
# 		pvals[j] = p
# 		j += 1
	return pvals

### define or select or load ROI groups: genotypes or treatments, etc.

# from csv file saved from excel of genotypes (wt and mut only),
#    save a roiGroups.csv file for later use by
def csvGenoToGenoFile(csvFile):
	genos = []
	with open(csvFile,'r') as f:
		for line in f:
			genos.extend(line.rstrip().split(','))
	wt = []
	mut = []
	i = 1
	for g in genos:
		if g.lower() in ['wildtype','wt','w','d']:
			wt.append(i)
		elif g.lower() in ['mut','mutant','m','u']:
			mut.append(i)
		i += 1

	with open('roiGroups.csv','w') as f:
		f.write('wt,'+ ','.join([str(i) for i in wt])+'\n')
		f.write('mut,'+','.join([str(i) for i in mut])+'\n')

# load roi groups from existing roiGroups file
def loadROIGroups(roiFile):
	#roiName1,1,2,3,4
	#roiName2,5,6,7,8,
	#...
	names = []
	groups = []
	f = open(roiFile,'r')
	for line in f:
		stuff = line.rstrip().split(',')
		names.append(stuff[0])
		rois = [int(x) for x in stuff[1:]]
		groups.append(rois)
	f.close()
	return names, groups

# save roi names and groups to roiGroups file
def saveROIGroups(names,groups):
	f = open('roiGroups.csv','w')
	for i in range(len(names)):
		f.write(names[i] + ',' + ','.join([str(x) for x in groups[i]]) + '\n')
	f.close()

# input user-defined roi groups	and save to roiGroups file
def makeAndSaveROIGroups():
	names = []
	groups = []
	while(True):
		roiName = raw_input('Enter group name (or hit return to be finished): ')
		if len(roiName) == 0:
			break
		names.append(roiName)
		roiNumbers = raw_input('Enter roi numbers, separated by spaces: ')
		groups.append([int(x) for x in roiNumbers.split()])
	saveROIGroups(names,groups)
	return names, groups

# make random ROI groups for testing, load existing roi groups, or enter roi groups
def getROIGroups(groupType,numROI):
	rows,cols = trackingTools.getRowsColsFromNumWells(numROI)
	nums = [x+1 for x in range(numROI)]
	grid = np.reshape(nums,[rows,cols])

	if groupType in ['topbottom','tb','top','t','ud','updown']:
		names = ['top','bottom']
		groups = [nums[:numROI/2],nums[numROI/2:]]
	elif groupType in ['oe','oddsevens','odds']:
		names = ['odds','evens']
		groups = [nums[0::2],nums[1::2]]
	elif groupType in ['check','checkerboard','c','ch']:
		names = ['UpperLeft','LowerRight']
		ul = []
		lr = []
		for r in range(rows)[0::2]:
			ul.extend(list(grid[r,:][0::2]))
			lr.extend(list(grid[r,:][1::2]))
		for r in range(rows)[1::2]:
			ul.extend(list(grid[r,:][1::2]))
			lr.extend(list(grid[r,:][0::2]))
		groups = [sorted(ul),sorted(lr)]
	elif groupType in ['lr','leftright','left','l']:
		left = []
		right = []
		names = ['left','right']
		for r in range(rows):
			left.extend(list(grid[r,:cols/2]))
			right.extend(list(grid[r,cols/2:]))
		groups = [sorted(left),sorted(right)]
	elif groupType in ['saved','load','geno','enter','e']:
		# load saved data, or enter ROI groups and save
		roiFile = glob.glob('roiGroups.csv')
		if len(roiFile) > 0:
			names, groups = loadROIGroups(roiFile[0])
		else:
			names, groups = makeAndSaveROIGroups()
	elif groupType in ['reversed','rev','revsave']:
		# load saved data, or enter ROI groups and save
		roiFile = glob.glob('roiGroups.csv')
		if len(roiFile) > 0:
			names, groups = loadROIGroups(roiFile[0])
		else:
			exit('No ROI file')
		for r in range(len(groups)):
			revroi = [97-x for x in groups[r]]
			revroi.reverse()
			groups[r] = revroi
	elif groupType in ['all','avg','average']:
		groups = [[x+1 for x in range(numROI)]]
		names = ['all']
	elif groupType in ['individuals','ind','each']:
		groups = [[x+1] for x in range(numROI)]
		names = [str(x+1) for x in range(numROI)]
	else: # random!
		from random import shuffle
		shuffle(nums)
		names = ['grp1','grp2']
		groups = [nums[:numROI/2],nums[numROI/2:]]

	print(names)
	print(groups)
	return names, groups

############ for XY data #######################################################

def subtractMin(vector):
	return vector - np.min(vector)

def subtractFromMax(vector):
	return np.max(vector)-vector

def roiXYVectors(data,roi):
	# roi is the NUMBER of the roi we are interested in looking at
	numROI = int(np.shape(data)[1]/2)
	x = data[:,roi]
	y = data[:,roi+numROI]
	coords = (x,y)
	return coords

def convertCoordstoDistances(coords):
	# coords are the  x and y coordinates for each time point
	x = coords[0]
	y = coords[1]
	distances = [math.hypot(x[i+1]-x[i],y[i+1,]-y[i]) for i in range(len(x)-1)]
	return distances

def omitBeginningOfData(data):
	numToOmit = 100 # 1 second at fps=30
	return data[numToOmit:,:]

def convertXYDataToDistances(xyData):
	#print('Converting xy data into distances . . .')
	roiData = xyData[:,1:]
	numRoi = int(np.shape(roiData)[1] / 2)
	distances = [convertCoordstoDistances(roiXYVectors(roiData,roi)) for roi in range(numRoi)]
	distances = np.transpose(distances)

	# put time vector back in
	distances = np.insert(distances,0,xyData[1:,0],axis=1)
	return distances

def subtractMinsForPlot(coords):
	x = coords[0]
	y = coords[1]
	x = subtractMin(x)
	y = subtractMin(y)
	# on cv2, y counts from top. on matplotlib, y counts from bottom
	y = subtractFromMax(y) # correct for cv2 vs. matplotlib coordinates
	return x, y

# plots traces and velocity vs. time in one ROI during experiment
def showTraceAndVelocityinROI(data, roi, binSizeInSeconds=1):
	# roi is the NUMBER (not the INDEX) of the roi we are interested in looking at
	# so roi #1 is #1 (not zero)
	# note that this takes a LONG time to plot big datasets,
	# b/c each frame is on plot

	print('Plotting data for ROI number: ' + str(roi))
	coords = roiXYVectors(data,roi)
	x,y = subtractMinsForPlot(coords)

	f = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
	gs = gridspec.GridSpec(4, 3)
	ax1 = f.add_subplot(gs[:3,:3])

	elapsedSeconds = (data[-1,0] - data[0,0]) * 86400
	avgfps = len(data[:,0]) / elapsedSeconds
	colors = np.arange(len(x)) / avgfps

	sc = ax1.scatter(x, y, c=colors, cmap='rainbow', edgecolor='')
	# to make colorbar units work, need to divide by 60 if units are minutes
	ax1.axis('equal')
	plt.xticks([]),plt.yticks([])
	plt.colorbar(sc)

	ax2 = f.add_subplot(gs[3,:3])
	distances = convertXYDataToDistances(data)

	# bin the data
	dpt, timeForPlot = binData(distances,binSizeInSeconds)

	# we only need the selected ROI
	dpt=dpt[:,roi-1]

	# start time at zero
	timeForPlot = [x - (timeForPlot[0]-np.fix(timeForPlot[0])) for x in timeForPlot]

	colors = np.arange(len(dpt))
	fmt, units, startZero = selectTimeFormat(elapsedSeconds)

	if startZero == 1:
		# start time at zero
		timeForPlot = [x - (timeForPlot[0]-np.fix(timeForPlot[0])) for x in timeForPlot]

	# do the time vs. distance plot
	ax2.plot(timeForPlot,dpt,'-', c = (0.8,0.8,0.8) )
	ax2.scatter(timeForPlot, dpt, c=colors, cmap='rainbow', edgecolor='')
	ax2.set_xlim([timeForPlot[0],timeForPlot[-1]])
	ax2.set_ylim([0,np.max(dpt)+5])
	ax2.xaxis.set_major_formatter( dates.DateFormatter(fmt) )
	plt.xlabel('time (%s)' % units)
	plt.ylabel('Distance (pixels / %i sec)' % binSizeInSeconds)
	toSave = 'roi' + str(roi) + 'trace.png'
	plt.savefig(toSave)
	print('Saved figure for ROI number: ' + str(roi))
	if np.shape(data)[0] < 50000:
		plt.show()
	else:
		print('See saved figure; too big to plot')
	return dpt

# show traces in ALL rois during experiment
def showAllTraces(data):
	# note that this takes a LONG time to plot big datasets, b/c each frame is on plot
	# saving as image takes less time
	print('Plotting trace data for all ROI')
	figWidth = 12
	figHeight = 8
	dotsPerInch = 80
	f = plt.figure(num=None, figsize=(figWidth,figHeight), dpi=dotsPerInch, facecolor='w', edgecolor='k')
	numROI = int(np.shape(data[:,1:])[1]/2)
	roiColors = make_N_colors('rainbow', numROI)
	for d in range(numROI):
		dotColor = roiColors[d]
		xdata = data[:,d+1]
		ydata = (figHeight*dotsPerInch)-data[:,d+1+numROI]
		plt.scatter(xdata,ydata, facecolor = dotColor, s=2)
	plt.xticks([]),plt.yticks([])
	plt.axis('equal')
	figname = 'allTraces.png'
	plt.savefig(figname)
	print('Saving figure ' + figname)
	if np.shape(data)[0] < 50000:
		plt.show()
	else:
		print('See saved figure; too big to plot')

# plot time vs. distance traveled. Much of this is duplicated (and improved?) in swPlots.py
def timeVdistance(makeribbon,roiNames,roiGroups):

	# activity = xyData converted to distance data, or dpix

	# find the optimal time bin size for an experiment of this duration
	binSizeInSeconds, yunits = findOptimalBinSize(activity[:,0])

	# bin the data
	binnedDistances, timeForPlot = binData(activity,binSizeInSeconds)

	# setup the figure
	f = plt.figure(num=None, figsize=(17,6), dpi=80, facecolor='w', edgecolor='k')
	gs = gridspec.GridSpec(15, 8)
	ax1 = f.add_subplot(gs[1:,:5]) # time vs. distance
	ax2 = f.add_subplot(gs[1:,6:], sharey=ax1) # boxplot
	ax3 = f.add_subplot(gs[:1,:5], sharex=ax1) # day / night indicators for time vs distance
	ax4 = f.add_subplot(gs[:1,6:], sharex=ax2) # day / night indicators for boxplot

	# set up the colors for the figure
	plotColors = make_N_colors('rainbow',len(roiNames))

	# get appropriate units for plot, and decide if start time axis at real time or at zero
	elapsedSeconds = (activity[-1,0] - activity[0,0]) * 86400
	fmt, units, startZero = selectTimeFormat(elapsedSeconds)

	if startZero == 1: # we want to start this plot at zero, b/c short time frame
		timeForPlot = [x - (timeForPlot[0]-np.fix(timeForPlot[0])) for x in timeForPlot]

	# go through groups and plot time vs. distance
	for r in list(range(len(roiGroups))):
		roiName = roiNames[r]
		roiGroup = [ x-1 for x in roiGroups[r] ] # indexing
		toPlot = binnedDistances[:,roiGroup]

		# plot time vs. data
		ax1.plot(timeForPlot,np.mean(toPlot,axis=1),color=plotColors[r],label=roiNames[r], linewidth=2)
		if len(roiGroup) > 1: # ribbon for error if more than 1 ROI averaged
			se = stde(toPlot,1)
			ax1.fill_between(timeForPlot, np.mean(toPlot,axis=1)-se, np.mean(toPlot,axis=1)+se, alpha = 0.3, facecolor=plotColors[r], edgecolor='')

	# adjust x axis to encompass whole time
	ax1.set_xlim([timeForPlot[0],timeForPlot[-1]])

	# Add labels with appropriate units to axes, based on time frame of plot
	if yunits == 'seconds':
		yUnitLabel = 'seconds'
		yUnitNumber = str(binSizeInSeconds)
	else:
		yUnitLabel = 'minutes'
		yUnitNumber = str(binSizeInSeconds/60)
	ax1.set_ylabel('Distance traveled\n(in pixels) per %s %s' % (yUnitNumber, yUnitLabel),fontsize=18)
	ax1.xaxis.set_major_formatter( dates.DateFormatter(fmt) )
	ax1.set_xlabel('Time (%s)' % units, fontsize=18)
	#ax1.locator_params(axis='x',nbins=10)

	# now the boxplots!
	boxData = []
	# find the day night boundaries and identities
	dnstarts, dnends, dayornight = findDayNightInTimeStamps(timeForPlot,9,23)

	# for each startTime and endTime, find the time window for this part
	b = []
	e = []
	for s in range(len(dnstarts)):
		b.append(bisect_left(timeForPlot, dnstarts[s]))
		e.append(bisect_left(timeForPlot, dnends[s]))

		for r in list(range(len(roiGroups))):
			roiName = roiNames[r]
			roiGroup = [ x-1 for x in roiGroups[r] ] # indexing
			dataForBox = binnedDistances[b[s]:e[s],roiGroup]
			boxData.append(np.mean(dataForBox,axis=1))

	b1 = ax2.boxplot(boxData, widths=0.5)
	boxLabels = roiNames * len(dnstarts) # need to have a set of names for each time period

	# format the boxplot colors
	plotColors = plotColors * len(dnstarts)
	b1 = formatBoxColors(b1,plotColors)

	# add a legend and label boxplots, if reasonable number of ROI
	if 1 < len(roiNames) <= 3:
		ax1.legend(loc=0, prop={'size':18}) # 0 is 'best'
		ax2.set_xticklabels(boxLabels, fontsize = 18)
	else:
		ax2.set_xticklabels('')

	# show the day / night indicators for the time vs. distance plots
	# THERE IS NOW A FUNCTION FOR THIS see above
	if len(dayornight) > 1:
		for s in range(len(dayornight)):
			timeRange = timeForPlot[b[s]:e[s]+1]
			if dayornight[s] == 'n':
				fillColor = (0,0,0)
			else:
				fillColor = (1,1,1)
			ax3.fill_between(timeRange,0,1, facecolor=fillColor, edgecolor='k')
	ax3.axis('off')

	# show the day / night indicators boxplots
	# THERE IS NOW A FUNCTION FOR THIS see above
	if len(dayornight) > 1:
		boxToTimeRatio = len(boxData)/len(dayornight)
		dayornight = np.repeat(dayornight,boxToTimeRatio)
		offset = 0
		startRange = [0.5,1.5]
		for s in range(len(dayornight)):
			xrange = [x + offset for x in startRange]
			if dayornight[s] == 'n':
				fillColor = (0,0,0)
			else:
				fillColor = (1,1,1)
			ax4.fill_between(xrange,0,1, facecolor=fillColor, edgecolor='')
			offset += 1
	ax4.axis('off')

	# show the plot!
	plt.show()
	return np.insert(binnedDistances,0,timeForPlot,axis=1)

def characterizeMovements(activityForROI):
	# Take an individual vector of movement magnitude (distances or displaced pixels) and
	# calculate characteristics of movement at framerate resolution
	# how long is each movement? what is the magnitude or distance of each movement?
	# how often are movements initiated (how many movements are there?)
	# how long is each period of rest?

	moveornot = activityForROI > 0
	boutstarts = np.where(moveornot[:-1] != moveornot[1:])[0] + 1
	boutstarts = np.insert(boutstarts,0,0)

	boutlengths = [boutstarts[n+1]-boutstarts[n] for n in range(len(boutstarts)-1)]
	boutlengths.append(len(activityForROI)-boutstarts[-1])
	boutlengths = np.array(boutlengths)

	movetypes = np.empty((len(boutlengths),))
	if moveornot[0] == 1:
		movetypes[::2] = 1
		movetypes[1::2] = 0
	else:
		movetypes[::2] = 0
		movetypes[1::2] = 1

	# movement lengths (in frames)
	moveStarts = boutstarts[np.where(movetypes==1)]
	moveLengths = boutlengths[np.where(movetypes==1)]

	# number of movements (use with time to calculate frequency)
	numMovements = len(moveStarts)

	# distance (or displaced pixels) per movement
	moveDistances = np.array([np.sum(activityForROI[moveStarts[m]:moveStarts[m]+moveLengths[m]]) for m in range(len(moveStarts)) ])

	# rest lengths (in frames)
	restLengths = boutlengths[np.where(movetypes==0)]

	return numMovements, moveLengths, moveDistances, restLengths

# do indMove type analysis for data from an ROI group
def indMove(roiData):
	numROI = np.shape(roiData)[1]
	numMovements = np.zeros(numROI)
	moveLengths = np.zeros(numROI)
	moveDistances = np.zeros(numROI)
	restLengths = np.zeros(numROI)

	for i in range(numROI):
		numMovements[i], ml, md, rl = characterizeMovements(roiData[:,i])
		moveLengths[i] = np.mean(ml)
		moveDistances[i] = np.mean(md)
		restLengths[i] = np.mean(rl)

	return numMovements, moveLengths, moveDistances, restLengths

def plotIndMove(data,roiNames,roiGroups):

	lightsON,lightsOFF = 7,23
	timeStamps = data[:,0]

	# extract day / night timing from timeStamps
	dnStarts, dnEnds, dayornight = findDayNightInTimeStamps(timeStamps,lightsON,lightsOFF)

	# initialize containers for	boxData
	mfBox,mlBox,mdBox,rlBox = [],[],[],[]

	# go through each time period
	for t in range(len(dayornight)):

		# go through roiGroups
		for r in range(len(roiGroups)):

			# get activity data for this roiGroup at this time
			b = bisect_left(timeStamps, dnStarts[t])
			e = bisect_left(timeStamps, dnEnds[t])
			d = data[b:e,roiGroups[r]]

			# get indMove data for this roiGroup
			numMovements, moveLengths, moveDistances, restLengths = indMove(d)

			# convert numMovements to movement Frequency
			elapsedTime = getElapsedTimeInSeconds(data[b:e,0])
			moveFreqs = [x/elapsedTime for x in numMovements]

			# convert moveLengths and restLengths to seconds
			fps = np.shape(data)[0] / float(elapsedTime)
			moveLengths = [x/fps for x in moveLengths]
			restLengths = [x/fps for x in restLengths]

			# add the indMove data to the boxPlotData
			mfBox.append(moveFreqs)
			mlBox.append(moveLengths)
			mdBox.append(moveDistances)
			rlBox.append(restLengths)

	# setup the figure
	plotColors = make_N_colors('rainbow',len(roiNames))
	figWidth = 1.7 * len(dayornight)
	figHeight = 9
	f = plt.figure(num=None, figsize=(figWidth,figHeight), facecolor='w', edgecolor='k')
	gs = gridspec.GridSpec(23, 1)
	axmf = f.add_subplot(gs[4:8,:])
	axml = f.add_subplot(gs[9:13,:])
	axmd = f.add_subplot(gs[14:18,:])
	axrl = f.add_subplot(gs[19:23,:])

	# make the boxplots
	boxAxes = [axmf,axml,axmd,axrl]
	boxData = [mfBox,mlBox,mdBox,rlBox]
	ylabels = ['Movement\nFrequency (Hz)','Movement\nDuration (s)','Movement\nMagnitude (pix)','Rest\nLength (s)']
	pc = plotColors * len(dnStarts)
	for i, ax in enumerate(boxAxes):
		b1 = ax.boxplot(boxData[i], widths=0.5)
		formatBoxColors(b1,pc)
		ax.set_ylabel(ylabels[i])
		ax.set_xticks([])

	# label with DND
	axdnd = f.add_subplot(gs[2,:])
	labelDNDboxplots(axdnd,mfBox,dayornight)

	# label the roi groups
	axroigroups = f.add_subplot(gs[:2,:])
	for i,name in enumerate(roiNames):
		axroigroups.plot([0,2],[-i,-i],color=plotColors[i],linewidth=4)
		axroigroups.text(2.2,-i,name)
	axroigroups.set_xlim([0,3])
	axroigroups.set_ylim([-len(roiNames),0.1])
	axroigroups.axis('off')

	# report some STATS
	if len(roiGroups) == 2:
		compDescriptions = ['frequency','duration','magnitude','rest']
		toComp = [mfBox, mlBox, mdBox, rlBox]
		for i,comp in enumerate(toComp):
			pvals = statsFromBoxData(comp,'k')
			print(compDescriptions[i]+','+','.join([str(p) for p in pvals]))

	# show the plot!
	plt.subplots_adjust(bottom=0.05, top=0.95, left=0.25)
	plt.show()

	return mfBox, mlBox, mdBox, rlBox

############ for social box data #################################################
# uses thigmotaxis stuff, which needs trackingTools, which needs opencv

############ for thigmotaxis data ################################################
# uses thigmotaxis stuff, which needs trackingTools, which needs opencv
def showInnerOuterForROI(data,roi,thresholdFromCenter,shape):
	coordsForROI = roiXYVectors(data,roi)
	x,y = subtractMinsForPlot(coordsForROI)
	f = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
	ax1 = f.add_subplot(111)
	fps = round(len(data[:,0]) / data[-1,0])
	if shape in ['s','square','rect','r','rectangle']:
		thigmo = trackingTools.xyToThigmoSquares(data,thresholdFromCenter)
	else:
		thigmo = trackingTools.xyToThigmoCircles(data,thresholdFromCenter)

	colors = thigmo[:,roi]
	colors = [(1,0,0) if c > 0 else (0,0,1) for c in colors]
	ax1.scatter(x, y, c=colors, edgecolor='')
	ax1.axis('equal')
	plt.xticks([]),plt.yticks([])
	titleString = 'Distance Threshold = ' + str(thresholdFromCenter) + '; red = outer, blue = inner'
	plt.title(titleString)
	plt.show()

def showInnerOuterAllROI(data,thresholdFromCenter):
	# note that this takes a LONG time to plot big datasets, b/c each frame is on plot
	# saving as image takes less time
	print('Plotting thigmotaxis for all ROI')

	figLength = 12
	figHeight = 8
	dotsPerInch = 80
	f = plt.figure(num=None, figsize=(figLength,figHeight), dpi=dotsPerInch, facecolor='w', edgecolor='k')
	ax1 = f.add_subplot(111)
	numROI = int(np.shape(data[:,1:])[1]/2)

	thigmo = trackingTools.xyToThigmo(data,thresholdFromCenter)

	for d in range(numROI):
		roi = d + 1
		colors = thigmo[:,roi]
		colors = [(1,0,0) if c > 0 else (0,0,1) for c in colors]
		xdata = data[:,roi]
		ydata = (figHeight*dotsPerInch)-data[:,roi+numROI]
		plt.scatter(xdata,ydata,c=colors, edgecolor='')

	plt.xticks([]),plt.yticks([])
	plt.axis('equal')
	titleString = 'Distance Threshold = ' + str(thresholdFromCenter) + '; red = outer, blue = inner'
	plt.title(titleString)
	figname = 'allThigmo.png'
	plt.savefig(figname)
	print('Saving figure ' + figname)
	if np.shape(data)[0] < 50000:
		plt.show()
	else:
		print('See saved figure; too big to plot')

def quantifyThigmo(thigmo,distances):

	# thigmo is from trackingTools.xyToThigmo(data,thresholdFromCenter)
	# distances is from convertXYDataToDistances(data)
	# the first column (timeStamps) needs to be removed from these!!!

	# some information about distances
	# print('Distances Summary')
	# print(np.shape(distances))
	#totalDistances = np.sum(distances[:,1:],axis=0)
	#print(totalDistances)

	# some information about thigmo
	# print('Thigmo Summary')
	# print(np.shape(thigmo))

	################### compare time in outer vs. inner: #################################
	# 	add up the frames in thigmo, divide by total frames, and convert via fps
	# 	==> get % of OUTER time in each ROI ... but only if distance is > some threshold

	################### compare distance in outer vs. inner: ###################
	#	for each ROI
	# 		add distances where thigmo is outer
	# 		add total distances
	# 	==> get % of OUTER distance in each ROI ... but only if distance is > some threshold

	# remove first row of thigmo data to equalize size with distances
	thigmo = thigmo[1:,:]

	# for each ROI, add distances where thigmo is outer (i.e. 1)
	numROI = np.shape(thigmo)[1]

	# set minimum distance threshold
	minimumDistanceThreshold = 10

	distanceOuter = []
	timeOuter = []

	for thisROI in range(numROI):
		thisROI = thisROI - 1
		thigmoForThisROI = thigmo[:,thisROI]
		distancesForThisROI = distances[:,thisROI]
		outerDistanceForThisROI = np.sum(distancesForThisROI[np.where(thigmoForThisROI == 1)])
		totalDistanceForThisROI = np.sum(distances[:,thisROI],axis=0)
		if totalDistanceForThisROI > minimumDistanceThreshold:
			distanceOuter.append(outerDistanceForThisROI/totalDistanceForThisROI)
			timeOuter.append(np.sum(thigmo[:,thisROI]) / np.shape(thigmo)[0])

	return timeOuter, distanceOuter

def plotThigmoQuantification(data, roiNames, roiGroups, thresholdFromCenter=0.71):

	timeThigmoData = []
	distanceThigmoData = []
	numROI = (np.shape(data)[1] - 1) / 2
	numGroups = len(roiNames)
	plotColors = make_N_colors('rainbow', numGroups)

	# get thigmo from XY data
	thigmo = trackingTools.xyToThigmoSquares(data,thresholdFromCenter)
	#thigmo = trackingTools.xyToThigmoCircles(data,thresholdFromCenter)

	# get distances from XY data
	distances = convertXYDataToDistances(data)

	for r in range(numGroups):
		# get thigmo only for the ROI's in this group
		thigmoForROI = thigmo[:,roiGroups[r]]

		# get distances data only for the ROI's in this group
		distancesForROI = distances[:,roiGroups[r]]

		# quantify thigmotaxis for this ROI group
		timeOuter, distanceOuter = quantifyThigmo(thigmoForROI,distancesForROI)

		timeThigmoData.append(timeOuter)
		distanceThigmoData.append(distanceOuter)

	f = plt.figure(num=None, figsize=(9,6), dpi=80, facecolor='w', edgecolor='k')
	gs = gridspec.GridSpec(1, 8)
	ax1 = f.add_subplot(gs[0,:3])
	ax2 = f.add_subplot(gs[0,5:],sharey=ax1)

	b1 = ax1.boxplot(timeThigmoData, widths=0.5)
	b1 = formatBoxColors(b1, plotColors)
	ax1.set_xticklabels(roiNames, fontsize = 18)
	ax1.set_ylabel('% Time in outer region', fontsize = 18)
	ax1.set_ylim([-0.1,1.1])

	b2 = ax2.boxplot(distanceThigmoData, widths=0.5)
	b2 = formatBoxColors(b2, plotColors)
	ax2.set_xticklabels(roiNames, fontsize = 18)
	ax2.set_ylabel('% Distance in outer region', fontsize = 18)

	plt.show()
	return distanceThigmoData

def outerPlusInner(activity):
	numROI = (np.shape(activity)[1] - 1) / 2
	for i in range(1,numROI):
		activity[:,i] = activity[:,i] + activity[:,i+numROI]
	return activity[:,:numROI+1]

############ for delta Pix data ################################################

# convert the xy data to dPix style data
def xy2dpix(searchString):
	# load xy data from a single file (no concatenation here!) ... usually 'xyData . . . '
	xyFile = glob.glob(searchString + '*.npy')[0]
	dPixFile = 'dpix' + xyFile[2:]
	print('Converting %s to %s' % (xyFile,dPixFile))

	data = np.load(xyFile)

	# convert to distances
	distances = convertXYDataToDistances(data)

	np.save(dPixFile,distances)
	return distances

# show overview  = heat map of plate, most active, least active, etc
def experimentOverview(searchString):
	data = loadData(searchString)
	print('Preparing overview  . . . ')
	numROI = np.shape(data)[1] - 1
	avgData = np.mean(data[:,1:],axis=0)
	numRows,numCols = trackingTools.getRowsColsFromNumWells(numROI)
	avgData = np.reshape(avgData,(numRows,numCols))
	f = plt.figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

	# heat map
	ax1 = f.add_subplot(2,2,1)
	ax1.imshow(avgData, cmap= 'hot', interpolation = 'nearest')
	ax1.set_xticks(range(numCols))
	ax1.set_xticklabels([str(x+1) for x in range(numCols)])
	ax1.set_yticks(range(numRows))
	ax1.set_yticklabels([str(x+1) for x in range(numRows)])

	# time vs. activity
	ax2 = f.add_subplot(2,2,3)
	binSizeInSeconds,u = findOptimalBinSize(data[:,0])
	binnedData, binnedTimeVec = binData(data,binSizeInSeconds)
	ax2.plot(binnedTimeVec,np.mean(binnedData,axis=1))
	fmt, units, startZero = selectTimeFormat(getElapsedTimeInSeconds(data[:,0]))
	ax2.xaxis.set_major_formatter( dates.DateFormatter(fmt) )
	ax2.set_xlabel('Time (%s)' % units)
	ax2.set_xlim([binnedTimeVec[0],binnedTimeVec[-1]])
	ax2.set_ylabel(('Activity (pixels per %i %s)' % (binSizeInSeconds,u)))
	ax2.set_title('Average Activity')

	# most active ROI
	roiAvg = np.mean(binnedData,axis=0)
	mostActive = int(np.where(roiAvg == np.max(roiAvg))[0])
	ax3 = f.add_subplot(2,2,2)
	ax3.plot(binnedTimeVec,binnedData[:,mostActive])
	ax3.xaxis.set_major_formatter( dates.DateFormatter(fmt) )
	ax3.set_xlabel('Time (%s)' % units)
	ax3.set_xlim([binnedTimeVec[0],binnedTimeVec[-1]])
	ax3.set_ylabel(('Activity (pixels per %i %s)' % (binSizeInSeconds,u)))
	mostActive = mostActive+1
	ax3.set_title(('Most active ROI = %i' % mostActive))

	# least active ROI
	leastActive = int(np.where(roiAvg == np.min(roiAvg))[0])
	ax4 = f.add_subplot(2,2,4)
	ax4.plot(binnedTimeVec,binnedData[:,leastActive])
	ax4.xaxis.set_major_formatter( dates.DateFormatter(fmt) )
	ax4.set_xlabel('Time (%s)' % units)
	ax4.set_xlim([binnedTimeVec[0],binnedTimeVec[-1]])
	ax4.set_ylabel(('Activity (pixels per %i %s)' % (binSizeInSeconds,u)))
	leastActive = leastActive + 1
	ax4.set_title(('Least active ROI = %i' % leastActive))

	plt.tight_layout()
	plt.show()

def plotDpixForROI(distances,roi):
    roi_data = distances[:,roi]
    f,a = plt.subplots(1,1,figsize = (12,3))
    a.plot(roi_data)
    a.set_xlabel('Time (frames)')
    a.set_ylabel('Distance (pixels)')
    a.set_title('Distance for ROI ' + str(roi))
    plt.show()

def showDpixAllROI(data):
	f = plt.figure(num=None, figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
	ftextX = 0.08
	timeVals = data[:,0]
	dpix = data[:,1:]
	maxVal = dpix.max()
	numROI = np.shape(dpix)[1]
	roiColors = lors('rainbow', numROI)
	roiColors.reverse()
	for col in range(numROI):
		plotColor = tuple([(x/255) for x in roiColors[col]])
		ax = plt.subplot(numROI,1,col+1)
		plt.plot(timeVals,dpix[:,col],color = plotColor)
		plt.locator_params(axis='y',nbins=3)
		ax.set_ylim([0,maxVal+5])
		ax.set_xlim([0,timeVals[-1]])
		if numROI > 12:
			plt.yticks([]) # comment in or out if too crowded
			ftextX = 0.1
		if col+1 != numROI:
			plt.xticks([])
	plt.xlabel('Time (sec)')
	f.text(ftextX, 0.5, 'Displaced Pixels', va='center', rotation='vertical')
	plt.show()

# THIS IS OBSOLETE ... use swPlots.py
def timeVsDpix(data,roiNames,roiGroups):

	vidInfo = loadVidInfo()
	fps = int(vidInfo['fps'])

	largerBin = 10 # in seconds
	smallerBin = 1 # in seconds

	largeFrames = largerBin * fps
	smallFrames = smallerBin * fps

	timeVals = data[:,0]
	roiData = data[:,1:]

	# bin the data into largerBin

	binStarts = list(range(0,np.shape(roiData)[0],largeFrames))
	timeForPlot = list(range(0,len(timeVals),largeFrames))

	# throw out last bin b/c incomplete
	binStarts = binStarts[:-1]
	timeForPlot = timeForPlot[:-1]
	timeForPlot = [t/fps for t in timeForPlot]

	largeBinnedData = [roiData[b:b+largeFrames,:] for b in binStarts]
	# data is now binned into chunks of size = largerBin

	# in each of these chunks,
	# bin into smaller bins
	smallBinStarts = list(range(0,largeFrames,smallFrames))

	# data to plot pixels
	movementSums = np.zeros([len(binStarts),np.shape(roiData)[1]])
	pixelSums = np.zeros(np.shape(movementSums))

	i = 0

	for bin in largeBinnedData:
		# for the time vs. pix plot, collect the sums in larger bins
		pixelSums[i,:] = np.sum(bin,0)

		# for the time vs. secs with activity plot ...

		# find the smaller bins
		#print(np.shape(bin))
		smallBins = [bin[b:b+smallFrames,:] for b in smallBinStarts]

		# sum the pixels in each smaller bin
		smallBinSums = np.sum(smallBins,1)
		#print(np.shape(smallBinSums))

		# convert summed pixels to yes/no (1/0) motion
		smallBinSums[np.where(smallBinSums != 0)]=1

		# find sum of the yes/no motion, and collect for plot
		movementSums[i,:] = np.sum(smallBinSums,0)

		i += 1

	# now plot
	f = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
	gs = gridspec.GridSpec(4, 4)
	plotColors = make_N_colors('rainbow',len(roiNames))
	ax1 = f.add_subplot(gs[:2,:3])
	ax2 = f.add_subplot(gs[2:,:3])
	ax3 = f.add_subplot(gs[:2,3], sharey=ax1)
	ax4 = f.add_subplot(gs[2:,3], sharey=ax2)

	pixelsBoxData = []
	movementBoxData = []

	# go through each group and plot
	for r in list(range(len(roiGroups))):
		roiName = roiNames[r]
		roiGroup = [ x-1 for x in roiGroups[r] ] # indexing

		# plot activity by MOVEMENT
		toPlot = movementSums[:,roiGroup]
		se = stde(toPlot,1)
		ax1.plot(timeForPlot, np.mean(toPlot,1),color=plotColors[r], label=roiName, linewidth=2)
		ax1.fill_between(timeForPlot, np.mean(toPlot,1)-se, np.mean(toPlot,1)+se, alpha = 0.3, facecolor=plotColors[r], edgecolor='')
		ax1.set_ylabel('Secs with movement in ' + str(largerBin) + ' secs', fontsize = 18)

		# collect data for boxplot
		movementBoxData.append(np.mean(toPlot,1))

		# plot activity by PIXELS
		toPlot = pixelSums[:,roiGroup]
		se = stde(toPlot,1)
		ax2.plot(timeForPlot, np.mean(toPlot,1),color=plotColors[r], label=roiName, linewidth=2)
		ax2.fill_between(timeForPlot, np.mean(toPlot,1)-se, np.mean(toPlot,1)+se, alpha = 0.3, facecolor=plotColors[r], edgecolor='')
		ax2.set_ylabel('Pixels per ' + str(largerBin) + ' secs', fontsize = 18)
		ax2.set_xlabel('Time (sec)', fontsize = 18)

		# collect data for boxplot
		pixelsBoxData.append(np.mean(toPlot,1))

	# boxplots
	b1 = ax3.boxplot(movementBoxData, widths=0.5)
	#ax3.set_yticks([])
	ax3.set_xticklabels('')

	b2 = ax4.boxplot(pixelsBoxData, widths=0.5)
	#ax4.set_yticks([])
	ax4.set_xticklabels(roiNames, fontsize = 18)

	# format box colors
	b1 = formatBoxColors(b1,plotColors)
	b2 = formatBoxColors(b2,plotColors)

	ax1.legend(loc=0, prop={'size':18}) # 0 is 'best'
	plt.show()

	return

def findDistancesBinAndCsvExport(binSizeInSeconds):
	# load xy data and concatenate files if necessary
	data = loadData('xy')

	#load vidInfo
	vidInfo = loadVidInfo()
	fps = int(vidInfo['fps'])

	# convert to distances
	distances = convertXYDataToDistances(data)

	# bin
	framesToBin = binSizeInSeconds * fps
	binnedDistances, binnedTimes = binData(distances,framesToBin)

	# insert time vector
	binnedDistances = np.insert(binnedDistances,0,binnedTimes,axis=1)

	# save as csv
	saveDataAsCSV(binnedDistances,'distanceData.csv')
