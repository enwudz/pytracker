## Updated 11 Jan 2021
# usage:
# To get ALL .npy files in the current directory:
# python npy2csv.py
#
# To get .npy AFTER a particular time
# python npy2csv.py 201231-083544
#     you can grab the '201231-083544' from the most recent 'distances' file

import numpy as np
import sys
import analysisTools
from matplotlib import dates

binSize = 60

# choose whether to get all of the data, or just those after a certain time
if len(sys.argv) > 1: # if an start time specified, gather data AFTER that time
        startData = sys.argv[1]
        data = analysisTools.loadDataFromStartTime(startData,'xy')
else: # if not options given, just get all of the .npy files
    # load all xy data
    data = analysisTools.loadData('xy')

# get time stamp data of last .npy file, and add these to the names of the saved files
lastTime = analysisTools.lastTimeStamp('xy')
fileExt = str(lastTime) + '_' + str(binSize) + '.csv'
csvFile = 'distances_' + fileExt
timeStampFile = 'timeStamps_' + fileExt

# split data at time gaps, if any
dataChunks = analysisTools.getMatrixSubsets(data,analysisTools.checkDataForTimeGaps(data))

# initialize empty containers for binnedData and binnedTime
binnedData = []
binnedTime = []

for i,data in enumerate(dataChunks):

    print('--> looking at chunk ' + str(i+1) + ' of ' + str(len(dataChunks)))
    # convert coordinates to distances
    d = analysisTools.convertXYDataToDistances(data)

    '''
    bin data
    analysisTools.binData takes two arguments:
    (1) the data to be binned (including timestamps)
    (2) the binSize in seconds
    for 1 minute bins, binSize = 60
    for 10 minutes bins, binSize = 600

    analysisTools.binData returns two outputs
    (1) binned data, (2) binned time vector
    '''
    
    # first, check to see if this chunk of data is longer than the binsize
    # assumption here is that d is collected in seconds
    timeVec = timeVec = data[:,0]
    timeSpan = analysisTools.getElapsedTimeInSeconds(timeVec)
    
    if timeSpan < binSize:
        print('This chunk (' + str(int(timeSpan)) + ' secs) is shorter than ' + str(binSize) + ' seconds')
        continue
   
    bd,bt = analysisTools.binData(d,binSize)
    print(' data in this chunk',np.shape(bd)) # could comment these out
    print(' times in this chunk',np.shape(bt)) # could comment these out

    if len(binnedData) == 0:
        binnedData = bd
        binnedTime = bt
    else:
        binnedData = np.vstack((binnedData,bd))
        binnedTime = np.hstack((binnedTime,bt))

# save the binned distances data to a .csv file
print('    saving data to %s . . . ' % csvFile)
np.savetxt(csvFile, binnedData, delimiter=',', fmt = '%1.2f' ,newline='\n')

# convert time to yyyy-mm-dd hh:mm:ss format
binnedTime = [dates.num2date(x).strftime('%Y-%m-%d %H:%M:%S') for x in binnedTime]

# save the binned time vector to a .csv file
print('    saving timestamps to %s . . . ' % timeStampFile)
with open(timeStampFile,'w') as f:
    for t in binnedTime:
        f.write('%s\n' % t)

# concatenate files if necessary
if len(sys.argv) > 1:
    print('\n')

    # removing any existing 'concatenated' files
    import os
    import glob
    existingFiles = glob.glob('*concatenated*')
    if len(existingFiles) > 0:
        for file in existingFiles:
            print('\n... removing existing ' + file + '\n')
            os.remove(file)

    analysisTools.concatenateCsv('distances')
    analysisTools.concatenateCsv('timeStamps')
    print('\nDone!\n')
