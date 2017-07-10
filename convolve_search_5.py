import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from line import Line


# Algorithm settings
window_width = 50 # 50
window_height = 80 # Break image into 9 vertical layers since image height is 720
window_area = window_width*window_height
offset = window_width/2
min_resp = window_area * 0.1 # minimum convolution response to be used
margin = 30 # How much to slide left and right for searching
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/800 # meters per pixel in x dimension

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def window_match(window, signal, min_resp):
    window_width = window.shape[-1]    
    # Convolve signal with window kernel
    conv_signal = np.convolve(window, signal)
    # Check minimum response
    max_resp_left = np.amax(conv_signal)
    if max_resp_left>=min_resp:
        center = np.argmax(conv_signal)-window_width/2
    else:
        center = None
    return center
    
def find_window_centroids(warped, nonzerox, nonzeroy, window_width, window_height, min_resp, margin):
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = window_match(window, l_sum, min_resp)
    
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = window_match(window, r_sum, min_resp)
    if r_center:
        r_center += int(warped.shape[1]/2)
    
    # handle cases of no detection
    assert (l_center and r_center), "Initial window search failed!"
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Identify the nonzero pixels in x and y within the window
    win_y_low = int(warped.shape[0]-window_height)
    win_y_high = int(warped.shape[0])
    win_xleft_low, win_xleft_high = l_center-offset, l_center+offset
    win_xright_low, win_xright_high = r_center-offset, r_center+offset
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)        
    
    # Go through each layer looking for max pixel locations
    n_levels = (int)(warped.shape[0]/window_height)
    for level in range(1,n_levels):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        l_min_index = int(max(l_center-margin,0))
        l_max_index = int(min(l_center+margin,warped.shape[1]))
        l_c = window_match(window, image_layer[l_min_index:l_max_index], min_resp)
        if l_c:
            l_center = l_c-offset+l_min_index
            
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center-margin,0))
        r_max_index = int(min(r_center+margin,warped.shape[1]))
        r_c = window_match(window, image_layer[l_min_index:l_max_index], min_resp)
        if r_c:
            r_center = r_c-offset+r_min_index
            
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
        #print("l:{}, r:{}".format(l_center,r_center))
        
        # Identify the nonzero pixels in x and y within the window
        win_y_low = int(warped.shape[0]-(level+1)*window_height)
        win_y_high = int(warped.shape[0]-level*window_height)
        win_xleft_low, win_xleft_high = l_center-offset, l_center+offset
        win_xright_low, win_xright_high = r_center-offset, r_center+offset
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)        

    return window_centroids, left_lane_inds, right_lane_inds

def get_curvature(ploty, y_eval, left_fitx, right_fitx):
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature (in meters)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return (left_curverad, right_curverad)
    

# warped == binary top-down image
def line_fit(warped, prev_left, prev_right, viz=False):
    # BGR -> Gray
    if warped.ndim==3:
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
        
    # If one line is first time detection, do a complete window search
    if not prev_left.current_fit.size or not prev_right.current_fit.size:
        window_centroids, left_lane_inds, right_lane_inds = \
                find_window_centroids(warped, nonzerox, nonzeroy, window_width, window_height, min_resp, margin)
        assert len(window_centroids) > 0, "Window search failed!"
    if prev_left.current_fit.size:
        fit = prev_left.current_fit # fit from previous frame
        left_lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] - margin)) & (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + margin)))
    if prev_right.current_fit.size:
        fit = prev_right.current_fit # fit from previous frame
        right_lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] - margin)) & (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + margin)))

    # Concatenate the arrays of indices
    if type(left_lane_inds[0]) is list or type(left_lane_inds[0]) is np.ndarray:
        left_lane_inds = np.concatenate(left_lane_inds)
    else:
        left_lane_inds = np.array(left_lane_inds, np.bool_)
    if type(right_lane_inds[0]) is list or type(right_lane_inds[0]) is np.ndarray:
        right_lane_inds = np.concatenate(right_lane_inds)
    else:
        right_lane_inds = np.array(right_lane_inds, np.bool_)
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    y_eval = np.max(ploty) # point at which to compute curvature (image bottom)

    
    if left_lane_inds.size:
        #print('LEFT')
        # Extract line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        # Generate dense x and y values
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        # Update current_fit
        prev_left.current_fit = left_fit
        prev_left.line_base_pos = (left_fitx[-1]-warped.shape[1]/2)*xm_per_pix
        prev_left.allx = left_fitx
        prev_left.ally = ploty
        prev_left.detected = True
    else:
        prev_left.detected = False


    if right_lane_inds.size:
        #print('RIGHT')
        # Extract line pixel positions
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        # Fit a second order polynomial to each
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate dense x and y values
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Update Line class
        prev_right.current_fit = right_fit
        prev_right.line_base_pos = (right_fitx[-1]-warped.shape[1]/2)*xm_per_pix
        prev_right.allx = right_fitx
        prev_right.ally = ploty
        prev_right.detected = True
    else:
        prev_right.detected = False
        
                

    # Compute curvatures
    left_curverad, right_curverad = get_curvature(ploty, y_eval, left_fitx, right_fitx)
    if left_lane_inds.size:
        prev_left.radius_of_curvature = left_curverad
    if right_lane_inds.size:
        prev_right.radius_of_curvature = right_curverad
        

    if viz:
        # Color points used for linefit
        img_linefit = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        img_linefit[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        img_linefit[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]                
        
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows     
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
        image_winfit = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results            



    # Display the final results
    if viz:
        fig, axes = plt.subplots(1,3, figsize=(10,4))
        axes[0].imshow(warped, cmap='gray')
        axes[0].set_title('warped binary')
        axes[1].imshow(image_winfit)
        axes[1].set_title('window fitting')     
        axes[2].set_title('line fitting')
        axes[2].imshow(img_linefit)    
        axes[2].plot(left_fitx, ploty, color='yellow')
        axes[2].plot(right_fitx, ploty, color='yellow')
        axes[2].set_xlim(0, 1280)
        axes[2].set_ylim(720, 0)
        #plt.tight_layout()
        #plt.show()
        #plt.savefig(f.split('/')[-1][:-4]+'_fit.png')
    
    return ploty, left_fitx, right_fitx, left_curverad, right_curverad
