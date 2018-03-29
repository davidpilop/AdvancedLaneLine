## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The main goal of the project is to write a  software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. Basically, it is an advinced version of the **Lane Line detection project**.

## Content of this repo

- `ALaneLine.ipynb` this is the main script for this project that ties everything together
- `CameraCalibrator.py` this class calibrates the camera
- `VisionFilters.py` this class encapsulates all code used to create the binary threshold image and calculates the perspective transform
- `LaneFinder.py` this class find lines
- `Line.py`

# Pipeline
The pipeline for this project comprises the following steps:
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
2. Apply a distortion correction to raw images
3. Use color transforms, gradients, etc., to create a thresholded binary image
4. Apply a perspective transform to rectify binary image ("birds-eye view")
5. Detect lane pixels and fit to find the lane boundary
6. Determine the curvature of the lane and vehicle position with respect to center
7. Warp the detected lane boundaries back onto the original image
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Each step is discussed in more detail below, including references to various Python classes and functions implemented for each step.

## Step 1 and 2: Camera calibration and distortion coefficients
All required code is contained in the `CameraCalibrator` class. Function `__calibrate` performs the actual calibration. The OpenCV functions `findChessboardCorners()` and `cv2.calibrateCamera()` are the backbone of the image calibration. Using the checkerboard images provided with the project the calibration matrix and distortion coefficients are calculated. "Object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world were prepared  assuming the chessboard pattern is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` is appended with a copy of it every time `findChessboardCorners()` successfully detect all chessboard corners in a test image. `imgpoints` is appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard pattern detection. The output `objpoints` and `imgpoints` were used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. It returns the camera matrix (`mtx`), distortion coefficients (`dist`), rotation and translation vectors, etc. After calibration completes the calibration matrix and distortion coefficients are written to disc. Subsequent runs of the pipeline read the calibration matrix and distortion coefficients from disc, which is faster than performing the calibration again each time.

One complicating factor was that the checkerboard images provided with this project do not contain the same number of detectable corners. This was resolved by manually determining the number of detectable corners for each image.

To show calibration works one of the checkerboard images is shown below. The first image is the original image, the second image is the image obtained after undistorting it using the calibration matrix and distortion coefficients.

![Undistort_chess](/output_images/Undistort_chess.png)
![Undistort_test](/output_images/Undistort_test.png)

## Step 3: Threshold binary image and Perspective transform
In order to detect lane lines in an image a binary threshold image is used. All code required to create the binary threshold image is contained in function `__binary` in class `VisionFilters`. I only used the L and S channels of the HLS space. I used the s channel for a gradient filter along x and saturation threshold, as well as the l channel for a luminosity threshold filter.

The various thresholds were determined using trial-and-error in combination, the test images provided with this project and the diagnostics view. I implemented a mask to remove all pixels outside the region of interest in function `__region_of_interest`

Here we can see an image test in HLS space
![HLS](/output_images/channels.jpg)
![Binary](/output_images/binary.jpg)

In order to view the lane from above a bird's-eye view transform is applied to the binary threshold image. All code required to create the bird's-eye view transform is contained in the `__init__` function of class `VisionFilters`. The correct source and destination points required by OpenCV's `getPerspectiveTransform` function were determined using the `straight_lines1.jpg` image. The points were manually picked in such a way that the two lane lines appear as two straight and parallel lines in the bird's eye view image.

The perspective transform is only calculated once rather than for each image/video frame because for this project the assumption that the road is always flat is valid.

![warp_roi](/output_images/warp_roi.jpg)


## Step 4: Detect lane lines

The function `__find_peaks` in class `FindLines.py` takes the bottom half of a binarized and warped lane image to compute a histogram of detected pixel values. The result is smoothened using a gaussia filter and peaks are subsequently detected using. The function returns the x values of the peaks larger than `thresh` as well as the smoothened curve.

The function `__get_next_window` takes an binary (3 channel) image `img` and computes the average x value `center` of all detected pixels in a window centered at `center_point` of width `width`. It returns a masked copy of img a well as `center`.

The function `__lane_from_window` slices a binary image horizontally in 6 zones and applies `__get_next_window`  to each of the zones. The `center_point` of each zone is chosen to be the `center` value of the previous zone. Thereby subsequent windows follow the lane line pixels if the road bends. The function returns a masked image of a single lane line seeded at `center_point`.
Given a binary image `left_binary` of a lane line candidate all properties of the line are determined within an instance of a `Line` class.

The `Line.update(img)` method takes a binary input image `img` of a lane line candidate, fits a second order polynomial to the provided data and computes other metrics. Sanity checks are performed and successful detections are pushed into a FIFO que of max length `n`. Each time a new line is detected all metrics are updated. If no line is detected the oldest result is dropped until the queue is empty and peaks need to be searched for from scratch.

![line_fit](/output_images/fitted_lines.jpg)

## Step 5: Extracting the local curvature of the road and vehicle localization

The radius of curvature is computed upon calling the `Line.update()` method of a line. The method that does the computation is called `Line.get_radius_of_curvature()`. The mathematics involved is summarized in [this tutorial here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).  
For a second order polynomial f(y)=A y^2 +B y + C the radius of curvature is given by R = [(1+(2 Ay +B)^2 )^3/2]/|2A|.

The distance from the center of the lane is computed in the `Line.set_line_base_pos()` method, which essentially measures the distance to each lane and computes the position assuming the lane has a given fixed width of 3.7m.

![warp_roi](/output_images/lane.png)

## Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail

If we try to run the code with the `challenge_video` and `harder_challenge_video` videos we can verify that the solution is not good enough for a car to drive autonomously. It is very difficult for the camera to see the image of the road if there is a lot of light or if the road has sharp curves. Also we do not take into account the slope of the road
