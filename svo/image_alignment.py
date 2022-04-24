import numpy as np
from camera import Camera
from scipy.interpolate import RectBivariateSpline
from config import Config


class ImageAlignment:
    cam = Camera()

    def __init__(self):
        self.Transform = None

    def findAlignment(self, previous_frame, current_frame):
        # given previous and current frame features, calculate transform
        # previous frame features : 2 x N
        # current frame features  : 2 x M

        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        threshold = Config.ImageAlignment.THRESHOLD
        maxIters = Config.ImageAlignment.MAX_ITER

        #   find all matching features -> NOT REQUIRED. ASSUMING THAT ALL KEYPOINTS ARE
        #   ORDERED AND THEN TAKING THE TOP N FEATURES (N = min(prev_frame_feat, curr_frame_feat))

        if (previous_frame.np_keypoints_.shape[0] != current_frame.np_keypoints_.shape[0]):
            num_features = np.min(
                previous_frame.np_keypoints_.shape[0], previous_frame.np_keypoints_.shape[0])
            previous_frame.np_keypoints_ = previous_frame[num_features, :]
            current_frame.np_keypoints_ = current_frame[num_features, :]

        ySpline = np.arange(0, previous_frame.image.shape[0])
        xSpline = np.arange(0, previous_frame.image.shape[1])
        current_frame_y_grad, current_frame_x_grad = np.gradient(
            current_frame.image)

        inter_prev_frame = RectBivariateSpline(
            ySpline, xSpline, previous_frame.image)
        inter_curr_frame = RectBivariateSpline(
            ySpline, xSpline, current_frame.image)
        inter_curr_frame_y = RectBivariateSpline(
            ySpline, xSpline, current_frame_y_grad)
        inter_curr_frame_x = RectBivariateSpline(
            ySpline, xSpline, current_frame_x_grad)

        A = np.zeros(
            (previous_frame.np_features_.shape[0], self.windowSize*self.windowSize))
        b = np.zeros(
            (previous_frame.np_features_.shape[0], self.windowSize*self.windowSize))

        for i in range(maxIters):

            for j in range(previous_frame.np_features_.shape[0]):
                # get sub windows from first frame and warp
                x0, y0 = previous_frame.np_keypoints_[j].astype(int)
                x0_temp_range = np.arange(
                    x0 - self.windowSize//2, x0 + self.windowSize//2)
                y0_temp_range = np.arange(
                    y0 - self.windowSize//2, y0 + self.windowSize//2)

                warped_points = M@np.array([x0_temp_range,
                                           y0_temp_range, np.ones((x0.shape[0]))])
                x0_warped, y0_warped = warped_points[0, :], warped_points[1, :]

                x0_temp, y0_temp = np.meshgrid(x0_warped, y0_warped)

                # get sub windows from second frame
                x1, y1 = current_frame.np_keypoints_[j].astype(int)
                x1_temp_range = np.arange(
                    x1 - self.windowSize//2, x1 + self.windowSize//2)
                y1_temp_range = np.arange(
                    y1 - self.windowSize//2, y1 + self.windowSize//2)
                x1_temp, y1_temp = np.meshgrid(x1_temp_range, y1_temp_range)

                # evaluate frames at points
                previous_frame_subimg = inter_prev_frame.ev(
                    y0_temp, x0_temp).reshape((-1, 1))
                current_frame_subimg = inter_curr_frame.ev(
                    y1_temp, x1_temp).reshape((-1, 1))
                dx = inter_curr_frame_x.ev(
                    y0_warped, x0_warped, dy=1).reshape((-1, 1))
                dy = inter_curr_frame_y.ev(
                    y0_warped, x0_warped, dx=1).reshape((-1, 1))

                intensity_res = (previous_frame_subimg -
                                 current_frame_subimg).flatten().reshape(-1, 1)

                jaccobian = np.hstack(
                    (x0_warped*dx, y0_warped*dx, dx, x0_warped*dy, y0_warped*dy, dy))
                A[j] = jaccobian
                b[j] = intensity_res

            del_m = np.linalg.lstsq(A, b, rcond=None)[0].reshape(2, 3)
            M = M + del_m
            self.Transform = M

            if np.linalg.norm(del_m) < threshold:
                break

        self.Transform = np.vstack((self.Transform, np.array([0, 0, 1])))
        return self.Transform

    def getRotTrans(self, H):
        # returns rotation and translation matrix
        '''
        K is the camera calibration matrix
        T is translation
        R is rotation
        '''
        K = ImageAlignment.cam.intrinsics

        H = H.T
        h1 = H[0]
        h2 = H[1]
        h3 = H[2]
        K_inv = np.linalg.inv(K)
        L = 1 / np.linalg.norm(np.dot(K_inv, h1))
        r1 = L * np.dot(K_inv, h1)
        r2 = L * np.dot(K_inv, h2)
        r3 = np.cross(r1, r2)
        T = L * (K_inv @ h3.reshape(3, 1))
        R = np.array([[r1], [r2], [r3]])
        R = np.reshape(R, (3, 3))

        return R, T
