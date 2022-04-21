from re import M
from weakref import ref
import cv2
import numpy as np
from config import Config
from camera import Camera
from pathlib import Path

class Matcher:
    cam = Camera(str(Path(__name__).parent / "mav0/cam0/sensor.yaml"))

    def __init__(self, ref_frame, cur_frame):
        self.ref_frame = ref_frame
        self.cur_frame = cur_frame

        # Calculate fundamental matrix between ref_frame and cur_frame
        self.F, _ = cv2.findFundamentalMat(ref_frame.keypoints_, cur_frame.keypoints_, cv2.FM_RANSAC)

    def searchEpipolarLine(self, ref_feature, depth_estimate, min_depth, max_depth):

        # Use minimum depth to find start point of search on epipolar line
        # Use maximum depth to find end point of search on epipolar line
        #   1. Use feature point (u,v) and projection matrix to find (x,y,1) in world space
        #   2. Scale up point to required depth and project point into second camera
        #   3. Optional - If point not on line find closest point on line
        min_point = Matcher.cam.back_projection(self.ref_frame.T_w_f_, ref_feature, min_depth)
        max_point = Matcher.cam.back_projection(self.ref_frame.T_w_f_, ref_feature, max_depth)

        min_point_in_cur_frame = Matcher.cam.project(self.cur_frame.T_w_f_, min_point)
        max_point_in_cur_frame = Matcher.cam.project(self.cur_frame.T_w_f_, max_point)

        # Search from start to end point to find pixel with best patch correspondence
        # TODO: Optimization - step through by some step length?
        x1, y1 = ref_feature
        lower_bound = min_point_in_cur_frame[1]
        upper_bound = max_point_in_cur_frame[1]
        return self.epipolarCorrespondence(x1, y1, lower_bound, upper_bound)


    def epipolarCorrespondence(self, x1, y1, lower_bound, upper_bound):
        '''
        Input:  im1, the first image
                im2, the second image
                F, the fundamental matrix
                x1, x-coordinates of a pixel on im1
                y1, y-coordinates of a pixel on im1
        Output: x2, x-coordinates of the pixel on im2
                y2, y-coordinates of the pixel on im2
        '''
        im1 = self.ref_frame.image
        im2 = self.cur_frame.image

        l_prime = self.F @ np.array([[x1], [y1], [1]])
        
        get_x_coord = lambda y: (-l_prime[1]/l_prime[0]) * y - l_prime[2]/l_prime[0]

        # TODO: Change the bounds to be constrained by min/max deth of reference frame
        lower_bound = int(Config.Matcher.window_size//2)
        upper_bound = im2.shape[0] - Config.Matcher.window_size//2 - 1

        min_val = float("inf")
        x2, y2 = -1, -1

        for y in range(lower_bound, upper_bound):
            x = get_x_coord(y)
            x = int(np.round(x))

            offset = int(Config.Matcher.window_size // 2)

            # TODO: Change to use different patch comparator
            # Use Sum of Absolute differences
            sad = np.sum(np.abs(im1[y1 - offset: y1 + offset + 1, x1 - offset: x1 + offset + 1].flatten() - im2[y - offset: y + offset + 1, x - offset: x + offset + 1].flatten()))

            if sad < min_val:
                min_val = sad
                x2 = x
                y2 = y

        return x2, y2