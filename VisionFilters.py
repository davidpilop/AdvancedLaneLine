################################################################################
################################ Vision Filter #################################
################################################################################

import cv2
import numpy as np

class VisionFilters(object):

    def __init__(self):
        corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])
        new_top_left=np.array([corners[0,0],0])
        new_top_right=np.array([corners[3,0],0])
        offset=[150,0]

        src = np.float32([corners[0],corners[1],corners[2],corners[3]])
        dst = np.float32([corners[0]+offset,new_top_left+offset,new_top_right-offset ,corners[3]-offset])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def __binary(self, s_thresh=(120, 255), sx_thresh=(20, 255),l_thresh=(40,255)):
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS).astype(np.float)
        #h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold saturation channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Threshold lightness
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

        channels = 255*np.dstack(( l_binary, sxbinary, s_binary)).astype('uint8')
        binary = np.zeros_like(sxbinary)
        binary[((l_binary == 1) & (s_binary == 1) | (sxbinary==1))] = 1
        binary = 255*np.dstack((binary,binary,binary)).astype('uint8')
        self.image = binary

    def __warp(self):
        self.image = cv2.warpPerspective(self.image, self.M, (self.shape[1], self.shape[0]), flags=cv2.INTER_LINEAR)

    def __inverse_warp(self):
        self.image = cv2.warpPerspective(self.image, self.Minv, (self.shape[1], self.shape[0]), flags=cv2.INTER_LINEAR)
        return self.image

    def __region_of_interest(self):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        vertices = np.array([[(0,0),(self.shape[1],0),(self.shape[1],0),
                          (6*self.shape[1]/7,self.shape[0]),
                          (self.shape[1]/7,self.shape[0]), (0,0)]],dtype=np.int32)

        mask = np.zeros_like(self.image)

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(self.shape) > 2:
            channel_count = self.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        #returning the image only where mask pixels are nonzero
        return cv2.bitwise_and(self.image, mask)

    def warp_binary_pipeline(self, image):
        self.shape = image.shape
        self.image = image  # Acordarse de calibration.undistort anted de esta función
        self.__binary()
        self.__warp()
        return self.__region_of_interest()

    def unwarp_binary_pipeline(self, image):
        self.shape = image.shape
        self.image = image  # Acordarse de calibration.undistort anted de esta función
        return self.__inverse_warp()
