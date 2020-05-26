# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of the following 5 steps.
1. convert the image to gray scale
2. apply a Gaussian filter to blur the gray scale image
3. apply Canny Edge Detector to the blured gray scale image
4. apply Hough Transform to detect the lines
5. draw the lines on the image

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by drawing the line between the top point of the detected lane and its corresponding point at the bottom of the image respectively. First, based on the slope of detected lines, I calculate the top points of the left and right lanes respectively, and compute the average slope of right lane and left lane. Then according to the values obtained above, I can calculate the intersection of the lanes with the bottom of the image according the formular y=mx+b


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be more than two lanes are detected when there are more than two lanes shown in the image at the region of interest.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to develope a region of interest selection algorithm that can be adaptive to the environment.
