import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob


def corners_unwarp(img, nx, ny, mtx, dist):
	# 1) Undistort the image
    undist = cv2.undistort(img, mtx, dist)
    
    # 2) Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny))

    mg = 100 # offset for dst points
    w, h = gray.shape[1], gray.shape[0]
    if ret:
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        dst = np.float32([[mg,mg],[w-mg,mg],[w-mg,h-mg],[mg,h-mg]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist, M, (w,h))
        return ret, warped, M
    else:
        return ret, None, None
        
# prepare object points
nx = 9 #the number of inside corners in x
ny = 6 #the number of inside corners in y
grid_size = 10
cal_dir = 'camera_cal'

imgpoints = []
objpoints = []
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Make a list of calibration images
fnames = glob.glob(cal_dir+'/*.jpg')
for f in fnames:
    print("reading: {}".format(f))
    img = cv2.imread(f)
	
	# Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if 'calibration3.jpg' in f: # this image has some reflection, fix it
        gray[gray>200] = 255

	# Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

	# If found, draw corners
    if ret == True:
        # Draw and display the corners
        #cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
		
        # Append to imgpoints and objpoints
        imgpoints.append(corners)
        objpoints.append(objp) # pre-computed

# Perform calibration
ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)			
if not ret:
	print("Calibration failed!!")
	quit()
    
np.savez("calib_params", mtx, dist)

'''
for f in fnames:
	print("reading: {}".format(f))
	img = cv2.imread(f)

	ret, top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
	if ret:
		cv2.imwrite(f+".png",top_down)
'''

