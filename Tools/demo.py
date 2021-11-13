import numpy as np
import os
import copy
import cv2

def general_read(filename, channel):
    """ Read general data from file, return as numpy array. """
    f = open(filename,'rb')
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height*channel
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    general = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width,channel))
    return general

def general_write(filename, general):
    """ Write general to file. """
    height,width = general.shape[:2]
    f = open(filename,'wb')
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    
    general.astype(np.float32).tofile(f)
    f.close()


def single_read(filename):
    f = open(filename,'rb')
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    general = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return general

def single_write(filename, general):
    height,width = general.shape[:2]
    f = open(filename,'wb')
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    
    general.astype(np.float32).tofile(f)
    f.close()


def NormalToColor(normal):
    return (normal + 1.0) / 2.0

def ReadDarksetList(path):
    with open(path, 'r') as fd:
        lines = fd.readlines()

    LongDisparityDict = {}
    DisparityShortDict = {}
    LongExposureDict = {}
    ShortExposureDict = {}
    for iid in range(len(lines)):
        items = lines[iid][:-1].split(' ')
        longName, dispName, shortName, longExposure, shortExposure = items

        if longName not in LongDisparityDict.keys():
            LongDisparityDict[longName] = [dispName]
        else:
            if dispName not in LongDisparityDict[longName]:
                LongDisparityDict[longName].append(dispName)

        if dispName not in DisparityShortDict.keys():
            DisparityShortDict[dispName] = [shortName]
        else:
            if shortName not in DisparityShortDict[dispName]:
                DisparityShortDict[dispName].append(shortName)

        LongExposureDict[longName] = longExposure
        ShortExposureDict[shortName] = shortExposure

    return LongDisparityDict, DisparityShortDict, LongExposureDict, ShortExposureDict


def ShowAll(filePath, LongDisparityDict, DisparityShortDict):

    longNameList = list(LongDisparityDict.keys())
    for iid in range(0, len(longNameList)):
        longName = longNameList[iid]
        # print("%d/%d"%(index, longNum))

        disparityName = LongDisparityDict[longName][0]
        shortNames = DisparityShortDict[disparityName]
        shortGT = shortNames[0]

        depthName = disparityName[:-4] + '.bin'
        normalName = disparityName[:-4] + '.bin'
        leftpathDepthPath = filePath + '/left/depth/' + depthName
        leftpathNormalsPath = filePath + '/left/normal/' + normalName

        # left depth
        LeftDepth = single_read(leftpathDepthPath) 

        # left normal
        LeftNormal = general_read(leftpathNormalsPath, 3) 
        LeftNormal = NormalToColor(LeftNormal)

        # left long
        leftLongPath = filePath + '/left/long_12bit/' + longName
        leftLongImage = cv2.imread(leftLongPath,-1)

        # right long
        rightLongPath = filePath + '/right/long_12bit/' + longName
        rightLongImage = cv2.imread(rightLongPath,-1)

        # left short gt
        leftShortPath = filePath + '/left/short_12bit_gt/' + shortGT
        leftShortImageGt = cv2.imread(leftShortPath,-1)

        # right short gt
        rightShortPath = filePath + '/right/short_12bit_gt/' + shortGT
        rightShortImageGt = cv2.imread(rightShortPath,-1)

        # Show
        LeftDepth = LeftDepth / np.max(LeftDepth)
        cv2.imshow('LeftDepth', LeftDepth)
        LeftNormal = np.stack((LeftNormal[:,:,2],LeftNormal[:,:,1],LeftNormal[:,:,0]), axis=-1)
        cv2.imshow('LeftNormal', LeftNormal)

        row0 = cv2.hconcat((leftLongImage, rightLongImage)) / 4095.0
        row1 = cv2.hconcat((leftShortImageGt, rightShortImageGt)) / 4095.0

        row01 = cv2.vconcat((row0,row1))

        for jjd in range(0,len(shortNames),5):
            shortName = shortNames[jjd]

            print('%d:%d/%d %s %s'%(iid,jjd,len(shortNames), longName, shortName))

            # Load !!!
            leftShortPath = filePath + '/left/short_12bit/' + shortName
            rightShortPath = filePath + '/right/short_12bit/' + shortName
                
            # left right short gt
            leftShortImage = cv2.imread(leftShortPath,-1)
            rightShortImage = cv2.imread(rightShortPath,-1)

            row2 = cv2.hconcat((leftShortImage, rightShortImage)) / 4095.0

            comb = cv2.vconcat((row01,row2))

            comb = cv2.resize(comb, None, None, 0.75, 0.75)
            cv2.imshow('comb', comb)
            cv2.waitKey(0)

    return 0


def main():
    filePath = './Tools/List.txt'
    data_path = './'

    LongDisparityDict, DisparityShortDict, LongExposureDict, ShortExposureDict = ReadDarksetList(filePath)
    ShowAll(data_path, LongDisparityDict, DisparityShortDict)



if __name__ == '__main__':
    main()