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
[video1]: ./project_video.mp4 "Video"

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

I only used color thresholding to create my thresholded binary image. I did not include gradients, as I felt that gradient thresholding were extremely affected by noise in the images, which caused problems for the harder challenge videos. Rather I made use of the L channels of both HLS and LAB channels to track the bright regions of the image, and the B channel to track the yellow lines on the image. I used top hat morphological operation which helps to focus on areas brighter than their surroundings and ultimately is better suited for lighting changes.

![alt text][image6]

```
def threshold_image(img):
    kernel = np.ones((14,14),np.uint8)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) 
    
    hls_l = hls[:,:,1]
    th_hls_l = cv2.morphologyEx(hls_l, cv2.MORPH_TOPHAT, kernel)
    hls_l_binary = np.zeros_like(th_hls_l)
    hls_l_binary[(th_hls_l > 20) & (th_hls_l <= 255)] = 1

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB) 
    lab_l = lab[:,:,0]
    th_lab_l = cv2.morphologyEx(lab_l, cv2.MORPH_TOPHAT, kernel)
    lab_l_binary = np.zeros_like(th_lab_l)
    lab_l_binary[(th_lab_l > 20) & (th_lab_l <= 255)] = 1

    lab_b = lab[:,:,2]
    th_lab_b = cv2.morphologyEx(lab_b, cv2.MORPH_TOPHAT, kernel)
    lab_b_binary = np.zeros_like(th_lab_b)
    lab_b_binary[(th_lab_b > 5) & (th_lab_b <= 255)] = 1

    full_mask = np.zeros_like(th_hls_l)
    full_mask[(hls_l_binary == 1) | (lab_l_binary == 1) | (lab_b_binary == 1)] = 1

    kernel = np.ones((6,3),np.uint8)
    erosion = cv2.erode(full_mask,kernel,iterations = 1)

    return erosion
```

With seperate thresholds created for each color channel, we combine these thresholds with a logical OR operator, to give us a total mask. As our thresholds were quite low, to pick up on small changes in color, we also have a lot of noise that we can reduce using an eroding function. 


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the locate_line function of the fourth code box in the pipeline_notebook, 
![alt text][image7]
![alt text][image8]

In the line_in_windows function of the fourth code box in the pipeline_notebook, 

![alt text][image9]
![alt text][image10]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image11]
![alt text][image12]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
