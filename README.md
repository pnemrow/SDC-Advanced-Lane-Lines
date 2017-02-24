##Advanced Lane Line Finding


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

[image1]: ./output_images/chessboard.gif "Undistorted Chess Board"
[image2]: ./output_images/distortion.gif "Undistorted Road"
[image3]: ./output_images/drawn_polygon.png "Drawn Polygon"
[image4]: ./output_images/polygon_new_fit.png "New Fit"
[image5]: ./output_images/warped_image.png "Binary Example"
[image6]: ./output_images/thresholded_image.png "Warp Example"
[image7]: ./output_images/histogram.png "Histogram"
[image8]: ./output_images/detected_line.png "Fit Visual"
[image9]: ./output_images/masked_detected_image.png "Output"
[image10]: ./output_images/highlighted_region.png "Output"
[image11]: ./output_images/original_highlighted.png "Output"
[image12]: ./output_images/final_output.png "Output"
[video1]: https://youtu.be/_QoFJHZtx04 "Video"
[video2]: https://youtu.be/Kyismlnbhn8 "Video"
[video3]: https://youtu.be/FPsG7UG4tfQ "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./pipeline_notebook.ipynb."

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Using the the same camera calibration and distortion coefficients returned from the camera calibration of the chessboard above, I used the same undistort function on a test image of the road below: 

![alt text][image2]


####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Transforming the perspective of an image is performed in my main "pipeline" function in code cell 4 of my pipeline_notebook Ipython notebook. I first define the shape I would like the output image to be, and get a warp matrix, and its inverse (to be used in later unwarping) to pass to the cv2.warpPerspective function. This function takes an image and then warps it by matching the source pixel position with the destination pixel positions defined by the warp matrix.:

```
def pipeline(img):
    ...
    size_for_warp = (int(undistorted_img.shape[1]/2), undistorted_img.shape[0])
    M, M_inv = get_warp_matrix(undistorted_img)
    warped = cv2.warpPerspective(undistorted_img, M, size_for_warp, flags=cv2.INTER_LINEAR)
    ...

```

Here is where I create a warp matrix and an inverse warp matrix.
```
def get_warp_matrix(img):
    src = np.float32([
        [448., 479.],
        [832., 479.],
        [1472., 680.],
        [-192., 680.]
    ])

    dst = np.float32([
        [96., 0.],
        [544., 0.],
        [544., 720.],
        [96., 720.]
    ])

    M = cv2.getPerspectiveTransform(src,dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    return M, M_inv

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 448,  479     | 96,  0        | 
| 832,  479     | 544, 0        |
| 1472, 680     | 544, 720      |
| -192, 680     | 96,  720      |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Here is the original image with the src points drawn on the left image, and the dst points drawn on the right to show how the warp would be made.

![alt text][image3]
![alt text][image4]

Below is the same image after the warp has been made. Notice that the lines are parellel, which will allow us to do some cool linear algebra!

![alt text][image5]


####3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In the threshold_image function in the fourth code cell of pipeline_notebook Ipython notebook, I only used color thresholding to create my thresholded binary image. I did not include gradients, as I felt that gradient thresholding were extremely affected by noise in the images, which caused problems for the harder challenge videos. Rather I made use of the L channels of both HLS and LAB channels to track the bright regions of the image, and the B channel to track the yellow lines on the image. I used top hat morphological operation which helps to focus on areas brighter than their surroundings and ultimately is better suited for lighting changes. With seperate thresholds created for each color channel, we combine these thresholds with a logical OR operator, to give us a total mask. As our thresholds were quite low, to pick up on small changes in color, we also have a lot of noise that we can reduce using an eroding function. 

![alt text][image6]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the locate_line function of the fourth code box in my pipeline_notebook Ipython notebook, I sum the vertical columns of these binary images, as they are essentially 2D arrays, and create a histogram of these images. I only do this on the bottom half of this image to help us identify the starting point of our line on the bottom of the image.

![alt text][image7]

By spliting the image into half its width, we can take the maximum point identified in the histogram, and begin to trace the line from there. I split the image into nine vertical segments, which we iterate through looking between a defined horizontal margin around where our maximum point was found. In each iteration, we take the mean of all non-zero pixels within the segment and margin to identify the the best fit of the line for each segment. By combining the means of all nine segments, we get a good idea of where the line is, and use numpy's polyfit function to get the polynomial coefficients for the line that we have found. 

Below is an image showing the rectangles, made by margins and segments, that we take an average of non-zero pixels to locate the line. 

![alt text][image8]

The line_in_windows function of the fourth code box in the pipeline_notebook is intended for use with videos, where we will have a previously found line. Instead of using the relatively inefficient locate_line function, where we iterate through nine segments of an image, we take the line found in the previous image and just search and get the mean for non-zero pixels within a margin of the pre-existing line. This allows us to quickly search for lines given previous context.

Below is an image showing the margins on either side of the line that non-zero pixels would be averaged within.

![alt text][image9]

We use these lines to color in the portion of this image that we will highlight.

![alt text][image10]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Almost as soon as I find a line in the locate_line or line_in_windows function of my pipeline_notebook, I make a call to my get_radius function, which takes uses the coefficients of the line in the following equation: ​

```
(1 + ( 2Ay + B)^2 )^(3/2) / 2A
```

This equation returns the radius of the curve in pixel space, but not in terms of physical space. In order to calculate this we were able to use the lane lines in pixel space and convert it to real world space using known lane line regulations. With this we found that in the horizontal (x direction) the conversion is about 3.7 meters for every 210 pixels. In the y direction, it is about 3 meters for every 170 pixels. With these converstions, we can then use the values in our calculations of the formula above and find lane curvature by the radius in terms of meters.

In the overlay_image function of the fourth code cell in the pipeline_notebook, I also calculate the distance from center. I am able to do this by averaging the x values of the left and right line, giving me a midpoint of the lane, and then subtracting the midpoint of the image from the lane mid-point. If the calculation is negetive it means the lane mid-point is on the left side. I use the same conversion from pixel-to-real space as I did with the curvature to find the distance from the center in meters.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

My pipeline function in the fourth code cell of my pipeline_notebook will return an image that has the highlighted lane region warped back to the original perspective. 

![alt text][image11]

I also calculated an approximate steering angle for these images which I present below.

![alt text][image12]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project video result][video1]

Here's a [link to my challenge result][video2]

Here's a [link to my harder challenge result][video3]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project I felt that the project video and first challenge were somewhat manageable, but the harder challenge video exploited many of the weaknesses of my current pipeline. Extreme differences in brightness and very wide turns caused my line detections to get of course quickly. I think this could be improved in a couple of different ways, especially if I had more time to work on this challenge. First I think that I should not be hard coding the warp matrix points. These should be dynamically adjusting to a vanishing point, which could be detected with further image preprocessing. Using a dynamic vanishing point would allow my pipeline to adjust to extreme turns, where a majority of lines in the image are pointing elsewhere from center.

I am not sure what other thresholding techniques could be used to adjust for extreme brightness changes, but perhaps this is where we would need more sensory data than just images to make more accurate detection. If I had more time, I could make a smarter "fall back" system, where if one line is not found it will be influenced or guided by the other line. As of now in my pipeline, if one line gets lost it doesn't get updated until the line is founded again, and in turning situations, this means that the line gets farther and farther off track while not being updated.

I live in a snowy area, and I get really concerned about how lane line detection will work in bad/snowy weather.  
