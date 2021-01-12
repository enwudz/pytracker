#!/usr/bin/python

import cv2
import numpy as np
import trackingTools
import analysisTools
import sys
from datetime import datetime, timedelta
from matplotlib import dates

'''
OVERVIEW:
This script tracks x,y coordinates of objects

UPDATED 20161117 to add timeStamps in real time
'''

# check if any saved data files here; exit if so
trackingTools.checkForPreviousData('xy*.npy')

# remove any previous xy data in this directory
trackingTools.deleteData('xy*.npy')

# setup video info dictionary
vidInfo = trackingTools.setupVidInfo()

# determine type of video source
vidSource, videoType = trackingTools.getVideoStream(sys.argv)
vidInfo['vidType'] = videoType
	
# acquire the pixel threshold from command line input or set to default
if len(sys.argv) > 2:
	pixThreshold = int(sys.argv[2])
else:
	pixThreshold = trackingTools.getPixThreshold()

print('Pixel Threshold is ' + str(pixThreshold))

# start camera
cap = cv2.VideoCapture(vidSource)

# adjust resolution if needed on attached camera
if vidSource == 0:
	cap = trackingTools.adjustResolution(cap,1) # 1 for logitech camera
else:
	vidInfo['movieFile'] = vidSource

# read the first frame
ret, oldFrame = cap.read()
oldFrame = trackingTools.grayBlur(oldFrame)

# load image mask
mask = cv2.imread('mask.png',0)
numROI = trackingTools.countROIs()

# make a different color for each ROI
roiColors = trackingTools.make_N_colors('rainbow', numROI)

# initialize window of previous positions for each ROI
storedPositions = [ [] for i in range(numROI)]

# get corner coordinates for each ROI (lowerLeft,upperRight)
roiCorners = trackingTools.getROICornersFromMask()
roiOffsets = [(roiCorners[i][0][0],roiCorners[i][1][1]) for i in range(numROI)]

# remove masked areas
maskedFrame = trackingTools.removeNonROI(oldFrame,mask)
# get the ROI slices from the masked image
roiSlices = [trackingTools.getImageSliceFromCorners(maskedFrame,r) for r in roiCorners]
maskSlices = [trackingTools.getImageSliceFromCorners(mask,r) for r in roiCorners]

# initialize center coordinates of the 'blob' in each ROI
#centerCoords = [ (0,0) for i in range(numROI)]
# OR guess the object
centerCoords = [trackingTools.guessTheObject(roiSlices[r],maskSlices[r]) for r in range(numROI)]

# how often to save data, in frames
# usually 9000 or 4500; 4500 is five minutes at 15 fps, about 6.5 minutes at 12 fps
saveFreq = 9000 
i = 0 # a frame counter for saving chunks of data

# preAllocate space for data matrix
# = matrix containing x data for each ROI, followed by y data for each ROI
data = np.empty([ saveFreq, 2 * numROI ])
startTime = datetime.now()
fname = 'frame1-' + startTime.strftime('%y%m%d-%H%M%S') + '.png'
cv2.imwrite(fname,oldFrame)
frameTimes = np.empty(saveFreq)

# go through and get differences and center of difference blobs
while(cap.isOpened()):
	
	# check if i bigger than saveFreq. If yes, save data to disk and reset values
	if i >= saveFreq:    
		print('saving data . . . ')   		
		# here's the big chance to insert timestamps if desired!
		data = np.insert(data,0,frameTimes,axis=1)
		# save data
		trackingTools.saveData(data,'xy')
		# reset data and i (and timeStamps)
		data = np.empty( [ saveFreq, 2 * numROI ] ) 
		i = 0
		frameTimes = np.empty(saveFreq)
	
	ret,currentFrame = cap.read()
	
	if ret == False or ( cv2.waitKey(1) & 0xFF == ord('q') ):
		print('End of Video')
		# save the data
		break
	
	# get time of frame capture
	frameTimes[i] = dates.date2num(datetime.now())
	
	# blur the frame
	blurFrame = trackingTools.grayBlur(currentFrame)
	
	# omit areas of image that are outside the ROI
	blurFrame = trackingTools.removeNonROI(blurFrame,mask)
	
	# find difference image
	diff = trackingTools.diffImage(oldFrame,blurFrame,pixThreshold,0)
	oldFrame = blurFrame
	
	# slice diff image into ROIs
	diffs = [trackingTools.getImageSliceFromCorners(diff,corner) for corner in roiCorners]
	
	# for now, do these in a for loop. Later = try to do list comprehension.
	for r in range(numROI):
		
		# find center coordinates of blob in each diff image
		if np.any(diffs[r]) == True:
			centerCoords[r] = trackingTools.findCenterOfBlob(diffs[r])
				
		# smooth motion by finding average of last X frames
		(centerCoords[r],storedPositions[r]) = trackingTools.smoothByAveraging(centerCoords[r],storedPositions[r])
				
		# update the center coordinate position by adding the offset for this ROI
		centerCoordsOffset = tuple(map(sum, zip(centerCoords[r],roiOffsets[r])))
						
		# show on image with different colors for each ROI
		cv2.circle(currentFrame, centerCoordsOffset, 4, roiColors[r], -1)
		
		# add data to matrix (with offset added)
		data[i][r] = centerCoordsOffset[0] # x data
		data[i][r+numROI] = centerCoordsOffset[1] # y data
		
	# add one to frame counter
	i += 1
	
	# show the image
	cv2.imshow('Press q to quit',currentFrame)

# done recording, save data
endTime = datetime.now()
data = np.insert(data,0,frameTimes,axis=1)
data = data[:i,:]
trackingTools.saveData(data,'xy')

# release camera
cap.release()
cv2.destroyAllWindows()

# concatenate the .npy files into a single file
data = analysisTools.loadData('xy') 

# calculate or acquire elapsed time
vidInfo = trackingTools.getElapsedTime(videoType,vidInfo,startTime,endTime)

# estimate FPS
vidInfo = trackingTools.calculateAndSaveFPS(data,vidInfo)

if videoType != 0:
	
	# recorded movie: use fps to get appropriate values for timestamps
	fps = int(vidInfo['fps'])
	start = datetime.now()
	timeStamps = [start + (i * timedelta(seconds=1./fps)) for i in range(np.shape(data)[0])]
	timeStamps = [dates.date2num(t) for t in timeStamps]

	# replace the timeStamps added during recording with these new ones
	data[:,0] = timeStamps

# save the data!
np.save('xyDataWithTimeStamps.npy',data)



