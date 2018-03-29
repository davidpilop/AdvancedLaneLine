################################################################################
############################## Camera Calibrator ###############################
################################################################################

import numpy as np
import cv2
import pickle, glob, os
import matplotlib.pyplot as plt

# Camera calibration constants
CAMERA_CAL_PICKLE = "camera_cal/calibration_data.p"
CAMERA_CAL_IMAGES = glob.glob('camera_cal/calibration*.jpg')
TEST_CAL_IMAGE_PATH = 'camera_cal\\calibration1.jpg'
CAMERA_CAL_IMAGES.remove(TEST_CAL_IMAGE_PATH)

class CameraCalibrator:
    '''Calibrate the camera using checkerboard images provided with project P4'''
    def __init__(self):
        '''Initialise class variables'''
        if os.path.isfile(CAMERA_CAL_PICKLE):
            # print('Loading pickled camera calibration information...')
            with open(CAMERA_CAL_PICKLE, mode='rb') as f:
                cal_data = pickle.load(f)
            self.mtx = cal_data['mtx']
            self.dist = cal_data['dist']
        else:
            self.__calibrate()

    def __calibrate(self):
        '''Calibrate the camera, pickle and return calibration data'''
        print('Recalibrating the camera...')

        # Number of detectable corners for each checkerboard image
        # image 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20,3, 4, 5, 6, 7, 8, 9.
        board_dims = [(9, 6), (9, 6), (9, 6), (9, 6), (9, 6), (9, 6), (9, 6),
                      (9, 6), (9, 6), (9, 6), (9, 6), (9, 6), (9, 6), (5, 6),
                      (7, 6), (9, 6), (9, 6), (9, 6), (9, 6)]

        # Arrays holding object points and image points for each image
        objpoints = [] # 3D with z = 0
        imgpoints = [] # 2D corner points

        if (len(CAMERA_CAL_IMAGES) > 0):
            img_shape = cv2.imread(TEST_CAL_IMAGE_PATH).shape

            for index, fname in enumerate(sorted(CAMERA_CAL_IMAGES)):
                # The number of detectable checkerboard corners is different for each image
                (NX, NY) = board_dims[index]

                # Create the chessboard object points grid or the current image (z = 0)
                objp = np.zeros((NX * NY, 3), np.float32)
                objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2) #x, y coordinates

                # Read and convert the image to grayscale and find the corners
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)

                if ret == True:
                    imgpoints.append(corners)
                    objpoints.append(objp)
                    img = cv2.drawChessboardCorners(img, (NX, NY), corners, ret)

                    output_img_file = './output_images/calibration' + str(index) + '.jpg'
                    cv2.imwrite(output_img_file, img)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape[0:2], None, None)

            # Pickle to save time for subsequent runs
            dist_pickle = {}
            dist_pickle["mtx"] = mtx
            dist_pickle["dist"] = dist
            pickle.dump(dist_pickle, open(CAMERA_CAL_PICKLE, "wb"))

            self.mtx = mtx
            self.dist = dist
            self.__calibrated = True
        else:
            print('no images')

    def undistort(self, image):
        '''Return an undistorted version of the image'''
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

def display_undistorted_image(calibration, test_img_dir = TEST_CAL_IMAGE_PATH) :
    # Draw original and undistorted images side by side
    test_img = cv2.cvtColor(cv2.imread(test_img_dir), cv2.COLOR_BGR2RGB)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(test_img)
    ax1.set_title('Original Image', fontsize=50)
    ax1.axis('off')
    ax2.imshow(calibration.undistort(test_img))
    ax2.set_title('Undistorted Image', fontsize=50)
    ax2.axis('off')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

def display_images_with_corners() :
    output_dir = './output_images/calibration*.jpg'
    output_images = glob.glob(output_dir)
    if output_images is None:
        print("Failed to read {}".format(output_dir))
    else:
        test_images = [plt.imread(path) for path in output_images]
        show_images(test_images)

def show_images(images, cmap=None):
    cols = 2
    rows = (len(images)+1)//cols
    plt.figure(figsize=(10, 19))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()
