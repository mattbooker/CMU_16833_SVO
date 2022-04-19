import numpy as np

frame_number = 0

class Frame:

    def __init__(self, image):
        self.id = frame_number
        self.image_ = image
        self.keypoints_ = None
        self.kp_scores_ = None
        self.is_keyframe_ = False

        # Transform from world to frame
        self.T_w_f_ = np.eye(4)

        frame_number += 1

    def setKeypoints(self, keypoints, scores):
        self.keypoints_ = keypoints
        self.scores = scores

