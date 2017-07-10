import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

npzfile = np.load("calib_params.npz")
mtx = npzfile['arr_0']
dist = npzfile['arr_1']


def apply_gray_thresh(img, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    binary_output = np.copy(binary)
    return binary_output

def apply_hls(img, hmax=255, smin=0):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    binary = np.zeros_like(S)
    binary[(S > smin) & (H <= hmax)] = 1
    binary_output = np.copy(binary)
    return binary_output

def apply_sobel(img, sobel_kernel=3):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 3) Take the absolute value of the derivative or gradient
    sobelx = np.absolute(sobelx)
    sobely = np.absolute(sobely)
    
    return sobelx, sobely
    
def abs_sobel_thresh(sobelx, sobely, orient='x', thresh=(0, 255)):
    # 1) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobelx = np.uint8(255*sobelx/np.amax(sobelx))
    sobely = np.uint8(255*sobely/np.amax(sobely))
        
    # 2) Create a mask of 1's where the response is between [thresh_min, thresh_max]
    mask = np.zeros_like(sobelx)
    if orient=='x':
        mask[(sobelx>=thresh[0]) & (sobelx<=thresh[1])]=1
    else:
        mask[(sobely>=thresh[0]) & (sobely<=thresh[1])]=1
    
    # 3) Return this mask as your binary_output image
    binary_output = np.copy(mask)
    return binary_output

def mag_thresh(sobelx, sobely, thresh=(0, 255)):
    # 1) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_mag = np.sqrt(sobelx**2+sobely**2)
    sobel_mag = np.uint8(255*sobel_mag/np.amax(sobel_mag))
        
    # 2) Create a mask of 1's where the response is between [thresh_min, thresh_max]
    mask = np.zeros_like(sobelx)
    mask[(sobel_mag>=thresh[0]) & (sobel_mag<=thresh[1])]=1

    # 3) Return this mask as your binary_output image
    binary_output = np.copy(mask)
    return binary_output

def dir_threshold(sobelx, sobely, thresh=(0, np.pi/2)):
    # 1) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_ori = np.arctan2(sobely, sobelx)
        
    # 2) Create a mask of 1's where the response is between [thresh_min, thresh_max]
    mask = np.zeros_like(sobelx)
    mask[(sobel_ori>=thresh[0]) & (sobel_ori<=thresh[1])]=1

    # 3) Return this mask as your binary_output image
    binary_output = np.copy(mask)
    return binary_output
    

    
def binarize(img, ksize = 3):    
    # Undistort image
    undist = cv2.undistort(img, mtx, dist)

    # Get sobelx, sobely
    sobelx, sobely = apply_sobel(undist, ksize)

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(sobelx, sobely, orient='x', thresh=(20, 255))
    grady = abs_sobel_thresh(sobelx, sobely, orient='y', thresh=(20, 255))
    mag_binary = mag_thresh(sobelx, sobely, thresh=(40, 255))
    dir_binary = dir_threshold(sobelx, sobely, thresh=(0, np.pi/2))
    gray_binary = apply_gray_thresh(undist, thresh=(50,255))
    hls_binary = apply_hls(undist, hmax=100, smin=170) #50, 80

    # Try different combinations of thresholding
    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    #combined[(mag_binary == 1) & (dir_binary == 1) & (gray_binary == 1)] = 1
    #combined[(gray_binary == 1) & (hls_binary == 1)] = 1
    #combined[(hls_binary == 1)] = 1
    combined[(mag_binary==1) & (gray_binary==1) | (hls_binary==1)] = 1
    mask1 = (mag_binary==1) & (gray_binary==1)
    mask2 = (hls_binary==1)
    color_binary = np.dstack(( mask2, mask1, np.zeros_like(mask1)))    
    color_binary = np.uint8(255*color_binary)
    combined = np.uint8(255*combined)
    
    return combined
    
