import numpy as np

class Frame:
    frame_number = 0

    def __init__(self, image):
        self.id = Frame.frame_number
        self.image_ = image
        self.np_keypoints_ = None
        self.is_keyframe_ = False

        # Transform from world to frame
        self.T_w_f_ = np.eye(4)

        Frame.frame_number += 1

    def setKeyFrame(self):
        self.is_keyframe_ = True