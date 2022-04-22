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
        
        self.F, _ = cv2.findFundamentalMat(ref_frame.np_keypoints_, cur_frame.np_keypoints_)


    def searchEpipolarLine(self, ref_feature, min_depth, max_depth):

        # Use minimum depth to find start point of search on epipolar line
        # Use maximum depth to find end point of search on epipolar line
        #   1. Use feature point (u,v) and projection matrix to find (x,y,1) in world space
        #   2. Scale up point to required depth and project point into second camera
        #   3. Optional - If point not on line find closest point on line
        min_point = Matcher.cam.backProjection(ref_feature, min_depth, self.ref_frame.T_w_f_)
        max_point = Matcher.cam.backProjection(ref_feature, max_depth, self.ref_frame.T_w_f_)

        min_point_in_cur_frame = Matcher.cam.project(min_point, self.cur_frame.T_w_f_)
        max_point_in_cur_frame = Matcher.cam.project(max_point, self.cur_frame.T_w_f_)

        
        x1, y1 = ref_feature

        # Search from start to end point to find pixel with best patch correspondence
        return self.epipolarCorrespondence(x1, y1, min_point_in_cur_frame, max_point_in_cur_frame)


    def epipolarCorrespondence(self, x1, y1, lower_bound, upper_bound):
        '''
        Input:  x1, x-coordinates of a pixel on im1
                y1, y-coordinates of a pixel on im1
                lower_bound, start point of search on epipolar line
                upper_bound, end point of search on epipolar line
        Output: x2, x-coordinates of the pixel on im2
                y2, y-coordinates of the pixel on im2
        '''
        im1 = self.ref_frame.image_
        im2 = self.cur_frame.image_
        offset = int(Config.Matcher.WINDOW_SIZE // 2)

        diff = upper_bound - lower_bound

        # Get which axis we will iterate over
        major_axis = 0 if diff[0] > diff[1] else 1
        minor_axis = 1 - major_axis

        # L' is a line with eq: ax + by + c = 0
        l_prime = self.F @ np.array([[x1], [y1], [1]])

        if major_axis == 0:
            # y = (-ax - c) / b
            get_minor_coord = lambda x: (-l_prime[0] * x - l_prime[2]) / l_prime[1]
        else:
            # x = (-by - c) / a
            get_minor_coord = lambda y: (-l_prime[1] * y - l_prime[2]) / l_prime[0]

        min_val = float("inf")
        x2, y2 = -1, -1

        # TODO: Optimization - step through by some step length?
        # TODO: Optimization - calculate where upper/lower bound goes off image rather than skipping
        start = int(lower_bound.flatten()[major_axis])
        stop = int(upper_bound.flatten()[major_axis])
        for major in range(start, stop):
            minor = get_minor_coord(major)
            minor = int(np.round(minor))

            if  major < offset or major > im1.shape[major_axis] - offset:
                continue
            
            if minor < offset or minor > im1.shape[minor_axis] - offset:
                continue

            if major == 0:
                x = major
                y = minor
            else:
                x = minor
                y = major

            # TODO: Change to use different patch comparator
            # Use Sum of Absolute differences
            sad = np.sum(np.abs(im1[int(y1) - offset: int(y1) + offset + 1, int(x1) - offset: int(x1) + offset + 1].flatten() - im2[y - offset: y + offset + 1, x - offset: x + offset + 1].flatten()))

            if sad < min_val:
                min_val = sad
                x2 = x
                y2 = y

        return x2, y2, sad

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from frame import Frame
    from feature_detector import FeatureDetector
    from feature_tracker import FeatureTracker
    from util import loadImage

    # im1, im2 = plt.imread('test_data/im1.png'), plt.imread('test_data/im2.png')
    im1, im2 = loadImage('test_data/im1.png'), loadImage('test_data/im2.png')
    correspondence = np.load('test_data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('test_data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']

    detector = FeatureDetector(im1.shape[1], im1.shape[0])
    tracker = FeatureTracker()
    
    im1 = im1
    im2 = im2

    a = Frame(im1)
    b = Frame(im2)

    detector.detectKeypoints(a)
    detector.detectKeypoints(b)

    b.T_w_f_ = np.array([[0.9994, 0.0333, 0.006, -0.026],
                        [-0.0337, 0.9653, 0.2589, -1.],
                        [0.0028, -0.2589, 0.9659, 0.0796],
                        [0, 0, 0, 1]])

    tracker.trackFeatures(a, b)

    # plt.imshow(im1)
    # plt.plot(a.np_keypoints_[:, 0], a.np_keypoints_[:, 1], "rx")
    # plt.show()

    # plt.imshow(im2)
    # plt.plot(b.np_keypoints_[:, 0], b.np_keypoints_[:, 1], "rx")
    # plt.show()

    Matcher.cam.intrinsics = K1
    matcher = Matcher(a, b)

    matcher.searchEpipolarLine(a.np_keypoints_[9], 2, 4)