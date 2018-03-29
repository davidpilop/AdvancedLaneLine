import numpy as np
import cv2, scipy
import matplotlib.pyplot as plt

class FindLines:
    def __find_peaks(self,thresh):
        img_half = self.image[self.img_shape[0]//2:,:,0]
        data = np.sum(img_half, axis=0)
        filtered = scipy.ndimage.filters.gaussian_filter1d(data,20)
        xs = np.arange(len(filtered))
        peak_ind = scipy.signal.find_peaks_cwt(filtered, np.arange(20,300))
        peaks = np.array(peak_ind)
        peaks = peaks[filtered[peak_ind]>thresh]
        return peaks,filtered

    def __get_next_window(self, img, window_center):
        ny,nx,_ = img.shape
        mask  = np.zeros_like(img)
        if (window_center <= self.width/2): window_center = self.width/2
        if (window_center >= nx-self.width/2): window_center = nx-self.width/2

        left  = window_center - self.width/2
        right = window_center + self.width/2

        vertices = np.array([[(left,0),(left,ny), (right,ny),(right,0)]], dtype=np.int32)
        ignore_mask_color=(255,255,255)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked = cv2.bitwise_and(mask,img)

        histogram = np.sum(masked[:,:,0],axis=0)
        if max(histogram > 10000):
            center = np.argmax(histogram)
        else:
            center = window_center
        return masked,center

    def __lane_from_window(self):
        n_zones=6
        ny,nx,nc = self.img_shape
        self.image = self.image.reshape(n_zones,-1,nx,nc)[::-1]
        window,center = self.__get_next_window(self.image[0],self.window_center)

        for zone in self.image[1:]:
            next_window,center = self.__get_next_window(zone,center)
            window = np.vstack((next_window,window))

        return window

    # Define a class to receive the characteristics of each line detection
    def get_binary_lane_image(self, img, line, window_center = 0, width=300):
        self.image = img
        self.img_shape = img.shape
        self.width = width
        if line.detected:
            self.window_center = line.line_pos
        else:
            peaks,filtered = self.__find_peaks(thresh=3000)
            # if len(peaks)!=2:
            #     print('Trouble ahead! '+ str(len(peaks)) +' lanes detected!')
            #     plt.imsave('output_images/troublesome_image.jpg',self.image)

            peak_ind = np.argmin(abs(peaks-window_center))
            peak  = peaks[peak_ind]
            self.window_center = peak

        lane_binary = self.__lane_from_window()
        return lane_binary
