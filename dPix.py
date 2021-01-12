#!/usr/bin/python -tt

"""
TO DO:
update to make analogous to recordXY
	sys to load movie
	timestamps ... etc.

OVERVIEW:
This script collects pixel differences between video frames
and saves these differences to a file or files with the extension '.npy'

NOTES:
Before this script runs, it will ask you if you want to
(a)append new data to the .npy file(s) already in the directory
(n)ew experiment: clear out old .npy files before running the analysis

"""

# IMPORT NECESSARY MODULES
import numpy as np
from datetime import datetime
import sys
import trackingTools
import cv2

# MAIN FUNCTION

def main(videoStream,vidType, pixThreshold):

	# setup video info dictionary
	vidInfo = trackingTools.setupVidInfo()

	vidInfo['vidType'] = vidType
	print(vidType)

	#pixThreshold = trackingTools.getPixThreshold()

	expDuration = 600000 # duration of experiment, in seconds; only relevant for live feed
	saveFreq = 4500 # how often to save data, in frames

	i,m = trackingTools.loadImageAndMask()

	# convert mask to integer values for bincount weights
	m,w = trackingTools.convertMaskToWeights(m)

	# start camera or open video
	cap = cv2.VideoCapture(videoStream)

	# adjust resolution if needed on attached camera
	if videoStream == 0:
		cap = trackingTools.adjustResolution(cap,1) # 1 for logitech camera
		displayDiffs = 1
	else:
		displayDiffs = 0
	#displayDiffs = 1 # override

	# Set Pixel Threshold
	ret,frame = cap.read()
	storedFrame = trackingTools.grayBlur(frame)
	print('PixelThreshold is ' + str(pixThreshold))

	# Acquire data
	if saveFreq  > expDuration: # do shorter of expDuration vs. saveFreq
			saveFreq = expDuration

	pixData = np.zeros([ saveFreq, len(np.unique(w))-1 ])

	i = 0 # a counter for saving chunks of data
	startTime = datetime.now()

	print('Analyzing motion data...')

	while(cap.isOpened()):

		ret,frame = cap.read()

		if ret == False:
			print('End of Video')
			break

		currentFrame = trackingTools.grayBlur(frame)

		# check if i bigger than saveFreq. If yes, save and reset values
		if i >= saveFreq:
			# save data
			trackingTools.saveData(pixData,'dpix')
			# reset pixData and i
			pixData = np.zeros([ saveFreq, len(np.unique(w))-1])
			i = 0

		# stop experiment if user presses 'q' or if experiment duration is up
		if ( cv2.waitKey(1) & 0xFF == ord('q') or
			len(sys.argv) == 1 and datetime.now() > startTime + timedelta(seconds = expDuration)
			):
			break

		# record pixel differences in all of the ROIs
		diff = trackingTools.diffImage(storedFrame,currentFrame,pixThreshold,displayDiffs)

		# calculate and record pixel differences
		counts = np.bincount(w, weights=diff.ravel())
		#print(counts[1:]) # output
		pixData[i,:] = counts[1:]

		storedFrame = currentFrame # comment out if nothing is in first frame
		i += 1

	# done recording. Remove empty rows (those bigger than i) from PixData
	endTime = datetime.now()
	pixData = pixData[:i,:]

	# Write data to file with timestamp:
	trackingTools.saveData(pixData,'dpix')

	# release camera
	cap.release()
	cv2.destroyAllWindows()

	# calculate or acquire elapsed time
	#vidInfo = trackingTools.getElapsedTime(vidType,vidInfo,startTime,endTime)

	# save vidInfo
	#trackingTools.saveVidInfo(vidInfo)

	# calculate fps and add time stamps to data
	#trackingTools.calculateAndSaveFPS('dpix',vidInfo)
	#trackingTools.addTimeStampsToData('dpix')

	return

def cmdLine(videoStream, vidType, pixThreshold):
	clearData = input("(a)ppend this new data, or (n)ew experiment?   >:").rstrip()
	# this is just input() in python3
	trackingTools.keepOrAppend(clearData,'dpix')
	main(videoStream, vidType, pixThreshold)
	return

def saveDataAsCSV():
	import analysisTools
	data = analysisTools.loadData('dpix')
	analysisTools.saveDataAsCSV(data,'saved_dpixData.csv')

if __name__ == '__main__':
	if len(sys.argv) > 2:
		pixThreshold = int(sys.argv[-1])
	else:
		pixThreshold = trackingTools.getPixThreshold(12)
	videoStream, vidType = trackingTools.getVideoStream(sys.argv)
	cmdLine(videoStream, vidType, pixThreshold)
	#saveDataAsCSV()
