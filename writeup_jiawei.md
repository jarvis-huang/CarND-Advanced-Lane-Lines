## Advanced Lane Line Detection
### by Jiawei Huang

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./submit/undist.png "Undistorted"
[image1.1]: ./submit/test4_undist_compare.png "test4_undist_compare"
[image2]: ./submit/test4_undist.png "test4"
[image2.1]: ./submit/test4_binary.png "Binary"
[image2.2]: ./submit/SGR.png "SGR"
[image3]: ./submit/test4_binary_warped.png "Warped"
[image4]: ./submit/test4_binary_warped_fit.png "Fit"
[image5]: ./submit/reprojection.png "Reprojection"
[video1]: ./submit/project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `calib_1.py`.

The key to perform camera calibration is to prepare the imgpoints and `objpoints`. 
The object points will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

I noticed that corner detection was not successful in all images. For those that are failed, I simply skip them.

I also saved the calibration results into a numpy binary file to be loaded easily later on.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction (`undistort_2.py`) to one of the test images like this one:
![alt text][image1.1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color transform and gradient thresholds to generate a binary image (implemented in `binary_3.py`). Through many experiments, the best combination I found is to use a combination of gradient magnitude threshold, grayscale threshold and saturation threshold. The selection criteria is:
`((mag_binary==1) & (gray_binary==1)) | (hls_binary==1)`

Here's an example of my output for this step.

![alt text][image2]
![alt text][image2.1]
![alt text][image2.2]

Green is magnitude threshold response. Red is S threshold response.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform uses OpenCV function `warpPerspective()`, which appears in lines 33 in the file `topdown_4.py`. The `warpPerspective()` function needs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python

src = np.float32(
    [[575,450],
     [705,450],
     [w-1,h-1],
     [0,h-1]])
dst = np.float32(
    [[200,0],
     [1000,0],
     [1000,h-1],
     [200,h-1]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
|  575, 450     |  200,   0     | 
|  705, 450     | 1000,   0     |
| 1279, 719     | 1000, 719     |
|    0, 719     |  200, 719     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image4]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in my code in `convolve_search_5.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in my code in `video_6.py` in the function `drawLaneOnImage()`. I also adopted the tutorial and used a Line class (`line.py`). Here is an example of my result on a test image:

![alt text][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Intially I set the window search margin to 50 pixels, but found it wobbles a lot when the road contrast changes. Then I reduced it to 30 and the result is much improved. The reason is that it is less likely to be influenced by noise far from current search position.

I also used previous frame's search result to speed up line fit process.

I tried my algorithm on the two challenging videos but failed poorly. But I don't have time to fix it. I will do that in the future when I have time.
