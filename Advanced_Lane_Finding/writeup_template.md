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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the Camera Calibration Section (code cells 2, 3 and 4) of the IPython notebook located in "./advanced_lane_finding.ipynb". I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

[image calibration10 calibration](./output_images/calibration10_calibration.jpg)


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I apply the same distortion correction method to one of the test images (code cell 5 of the IPython notebook located in "./advanced_lane_finding.ipynb") like this one:

[image straight_lines1 calibration](./output_images/straight_lines1_calibration.jpg)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of HLS color, HSV color and x gradient thresholds to generate a binary image (Creating a binary image section or code cell 6 in file "./advanced_lane_finding.ipynb"). The threshold of s color and x gradients are set throught testing the test images. Here's an example of my output for this step. 

[image straight_lines1 binary](./output/straight_lines1_binary.jpg "image straight_lines1 binary")

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in the Perspective Transform Section in the file "./advanced_lane_finding.ipynb"  (or in the code cell 7, 8 of the IPython notebook).  The `warper()` function takes an undistorted image, source points, destination points and image size as inputs. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 120, img_size[1] / 2 + 145],
    [((img_size[0] / 6) - 5), img_size[1]],
    [(img_size[0] * 5 / 6) + 45, img_size[1]],
    [(img_size[0] / 2 + 130), img_size[1] / 2 + 145]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
    
# src and dst points are in the writeup_template.md
src = np.float32(
    [[(img_size[0] / 2) - 120, img_size[1] / 2 + 145],
    [((img_size[0] / 6) + 65), img_size[1]-50],
    [(img_size[0] * 5 / 6) - 30, img_size[1]-50],
    [(img_size[0] / 2 + 130), img_size[1] / 2 + 145]])
    
dst = np.float32(
    [[(img_size[0] / 4), 0.0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0.0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 520, 505      | 320, 0        | 
| 268, 670      | 320, 720      |
| 1037, 670     | 960, 720      |
| 770, 505      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points (red points) onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

[image straight_lines1 undistorted and warped](./output_images/straight_lines1_original_undistorted_and_warped.jpg)

By applying the same procedure to an undistorted binary image (in the code cell 9 of the IPython notebook), it works and the result is shown below.

[image straight_lines1 binary undistorted and warped](./output_images/straight_lines1_binary_undistorted_and_warped.jpg)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I apply the solution code (in the code cell 10 of the IPython notebook) provided in the class to fit lane lines with a 2nd order polynomial kinda like this:

[straight_lines1_fit_polynomial](./output_images/straight_lines1_fit_polynomial.jpg)

First, it searches for the starting point of the left and right lane by using histogram to count the maximum number of nonzero pixels. Then it applies sliding window method to find the location of lanes from their starting points respectively. Finally, their positions are fitted with a polynomial. Hyperparameters are set as follows:

nwindows = 8
margin = 160 
minpix = 50. 

The code for visualization is commented when applying the pipeline.

I also apply the solution code (in the code cell 15 of the Ipython notebook) provided in the class to search lane lines withing the margin around the previous polynomial to make the pipeline faster. The margin is set to 100. The code for visualization is commented when applying the pipeline.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I apply the solution code (in the code cell 11 and 12 of the IPython notebook) provided in the class to calculate the radius of curvature of the lane in pixels and meters respectively. The position of the vehicle with respect to center is measured by the distance of the center of the image in the x direction with respect to the middle position of the left land and right lane in the x direction (in the code cell 13 of the Ipython notebook). 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the 14th code cell of the IPython notebook.  Here is an example of my result on a test image:

[straight_lines1_lane_area_drawn][./output_images/straight_lines1_lane_area_drawn.jpg]


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I create method to check whether detected lanes are parallel or not in the 16th code cell of the Ipython notehobook. In the 17th code cell of the Ipython notebook, I create a method to average the lane coefficients (left_fit and right_fit) over the past count_frames (it is set to 10) number of frames. I create a simple Lane class to track the lane finding process in the 18th code cell of the Ipython notebook and the video process pipeline in the 17th code cell of the Ipython notebook. 

I track the offset and radius detected from each image. If the horizontal lanes' distance is not within the range and the lanes are parallel, the base points of the lane lines are searched through the whole image. Otherwise, the base points of the lane lines are searched within a margin of the previous detected lane lines' positions. 

My pipeline performs reasonably well on the entire project video without catastrophic failures. Here's a [link to my video result of project video](./output_project_video.mp4) 

I have also applied my pipeline to test the challenge_video and the harder_challenge_video respectively. 

Here's a [link to my video result of challenge video](./output_challenge_video.mp4) 

Here's a [link to my video result of harder_challenge video](./output_harder_challenge_video.mp4) 

I provide my thoughts on the discussion section below.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One issue here is to detect lane lines' positions precisely even when the lane lines radius changes fast. My pipeline fails on this point when I try the challenge and harder challenge videos. My pipeline works well when the radius of lane lines does not change fast. 

Another issue is to choose the source and destination point automatically when doing the perspective transform. I choose the source and destination point by hand. I also need to try many times. It costs a lot of time. 

For making it more robust, I will try developing algorithms to solve the two issues that mentioned above.