import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from moviepy.editor import VideoFileClip # needed to edit/save/watch video clips
from binary_3 import binarize
from topdown_4 import topdown, Minv
from convolve_search_5 import line_fit
from line import Line

npzfile = np.load("calib_params.npz")
mtx = npzfile['arr_0']
dist = npzfile['arr_1']
line_left, line_right = Line(), Line()
first_line_left, first_line_right = Line(), Line()
i = 0

# warped = 3-ch
def drawLaneOnImage(undist, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

# Order: RGB
def process_image(img):
    global i, line_left, line_right
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_undist = cv2.undistort(img, mtx, dist)
    img_binary = binarize(img)
    img_topdown = topdown(img_binary)

    line_fit(img_topdown, line_left, line_right)
    res = drawLaneOnImage(img_undist, line_left.ally, line_left.allx, line_right.allx)
    
    cv2.imwrite(str(i)+'.png', res)
    quit()
    i += 1
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    
    display_str = 'left curvrad: {:5d} m'.format(int(line_left.radius_of_curvature))
    cv2.putText(res, display_str, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    display_str = 'right curvrad: {:5d} m'.format(int(line_right.radius_of_curvature))
    cv2.putText(res, display_str, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    display_str = 'center pos: {:1.2f} m'.format((line_right.line_base_pos+line_left.line_base_pos)/2)
    cv2.putText(res, display_str, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    
    return res
    #return np.array(cv2.merge((img_topdown,img_topdown,img_topdown)), np.uint8)
    

video_in = 'project_video.mp4'
#video_in = 'challenge_video.mp4'
#video_in = 'harder_challenge_video.mp4'
video_out = video_in.replace('.mp4', '_out.mp4')

#clip = VideoFileClip(video_in).subclip(18,20)
clip = VideoFileClip(video_in)
new_clip = clip.fl_image(process_image)
new_clip.write_videofile(video_out, audio=False)
