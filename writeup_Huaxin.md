## Writeup Of Huaxin

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

[image1]: ./output_images/original_undistorted.png "Undistorted"
[image2]: ./output_images/original_undistorted_example.png "Road Transformed"
[image3]: ./output_images/original_pipeline_example.png "Binary Example"
[image4]: ./output_images/original_perspective_example.png "Warp Example"
[image5]: ./output_images/line_fit_example.png "Fit Visual"
[image6]: ./output_images/line_curverad_example.png "Output"
[video1]: ./project_video_result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how to computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
* 1) Make list for both test images and straight lines images
* 2) Use `cv2.cvColor()` to convert color image into gray, which could increase the effiency.
* 3) Use `cv2.undistort()` with previous the camera calibration and distortion coefficients to undistort the images.

![alt text][image2]

#### d. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

First of all, I define the functions follow the lesson's content, like:
* `hls_select(img, thresh=(0, 255))` to thresholds the S-channel of HLS
* `Color2Gray(img)` to convert color image to gray
* `abs_sobel_thresh(img, orient='x', sobel_kernel=3,abs_thresh=(0, 255))` to applies Sobel x or y, then takes an absolute value and applies a threshold
* `mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255))` to applies Sobel x and y, then computes the magnitude of the gradient and applies a threshold
* `dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2))` to applies Sobel x and y, then computes the direction of the gradient and applies a threshold.
* `pipeline(img, sobel_kernel=3, s_thresh=(170, 255), sx_thresh=(20, 100))` to applies both the S-channel of HS and Sobel x
* `region_of_interest(img, vertices)` to only keeps the region of the image defined by the polygon

Then I run `pipeline()`function and `abs_sobel_thresh()` function with same test image to compare the output performance.

![alt text][image3]
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The most important thing of this perspective transform is to find out the `src` and `dst`matrix, I follow the lesson's guidance to select 4 point for each side to transfer the normal car view into "birds-eye view". 

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

And then apply `cv2.getPerspectiveTransform(src,dst)` function to calculate `M`
(the perspective transform matrix). Then use `cv2.warpPerspective(image, M, imag_size,  flags=cv2.INTER_LINEAR)` function to warp the test image.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Then I combine color transforms, gradients or other methods to create different thresholded binary image and warped image, which is aim to find out the good combination and good parameters process the images.

![alt text][image4]
#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In this task, I just follow the lesson's guidance:
Fistly, I convert a test warped image into binary image. Then take a histogram of the bottom half of the image, find the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines. Choose and set the sliding windows. Step through the windows one by one to extract left and right line pixel positions. Fit a second order polynomial to each road line. Finally, Generate x and y values for plotting.

I use a warped test image as example to run this feature.

![alt text][image5]
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I just follow the equation to calculate the curvature of each detected line:
* Define conversions in x and y from pixels space to meters
`ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
`
* Calculate the new radii of curvature
`left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Given src and dst points, calculate the perspective invert transform matrix,then create an image to draw the lines on. And folow 2nd order fit equation draw the lane onto the warped blank image.Then warp the blank back to original image space using inverse perspective matrix (Minv) and combine the result with the original image. Finally, add curverad value into result image.

![alt text][image6]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* I use a lot of time to tunne the parameter of image process function but result is not very good, still cannot pass the challenge video. And later I want to try to combine HLS and gray Thresholds method, because in the videos there always has a yellow line and write line. I think HLS threshold is good at yellow line detection, and gray threshold is good at write line detection.

* The function which fit positions with a polynomial is not good or efficient enough. Later I need improve it and try to use convolution method. 

* In this submit, I don't realize vehicle position detection feature, I need implement it in next submit. 
