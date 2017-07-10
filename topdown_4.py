import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

w, h = 1280, 720

# Four points of a trapezoid
src = np.float32([[575,450],[705,450],[w-1,h-1],[0,h-1]])
# For destination points, I'm arbitrarily choosing some points to be
# a nice fit for displaying our warped result 
# again, not exact, but close enough for our purposes
dst = np.float32([[200,0],[1000,0],[1000,h-1],[200,h-1]])
# Given src and dst points, calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

        
npzfile = np.load("calib_params.npz")
mtx = npzfile['arr_0']
dist = npzfile['arr_1']
fnames = glob.glob('test_images/*.jpg')
fnames = ['test_images/test5.jpg']
fnames = ['test5_binary.png']
fnames = glob.glob('./*binary.png')

def topdown(undist): 
    
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, (w, h))
    
    #cv2.imwrite(f.split('/')[-1][:-4]+'_warped.png', warped)
    
    return warped
    
