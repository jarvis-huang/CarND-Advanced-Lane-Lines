import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

npzfile = np.load("calib_params.npz")

mtx = npzfile['arr_0']
dist = npzfile['arr_1']

img = cv2.imread('test_images/test4.jpg')
#img = cv2.imread('test_images/straight_lines1.jpg')
#img = cv2.imread('camera_cal/calibration1.jpg')

undist = cv2.undistort(img, mtx, dist)
cv2.imwrite('test4_undist.png', undist)

fig, axes = plt.subplots(1,2, figsize=(12,6))
axes[0].set_title('before undistortion')
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[1].set_title('after undistortion')
axes[1].imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
plt.tight_layout()

plt.savefig('compare_test4.png')
#cv2.imwrite('calibration1_undist.png', undist)
