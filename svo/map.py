import numpy as np
from config import Config

class Map:

    def __init__(self):
        self.points = np.empty((0, 3)) # 3D points added to the map
        self.avg_scene_depth = 0 # current scene depth
        self.keyframes = [] # stored keyframes - total 10 (based on paper)
    
    def initial_map(self, world_pts):
        self.points = world_pts

        self.avg_scene_depth = np.mean(self.points[:, -1])

    def checkKeyframe(self, frame):
        total_dist = 0
        max_dist = 0
        for idx, keyframe in enumerate(self.keyframes):
            # Calculate distance between the frame and the keyframe
            T_frame = keyframe.T_w_f_ @ np.linalg.inv(frame.T_w_f_)
            dist = np.linalg.norm(T_frame[:-1, -1])
            print(dist)
            if dist > max_dist:
                max_dist = dist
                max_index = idx
            total_dist += dist
        
        avg_dist = total_dist/len(self.keyframes)
        # print(avg_dist, 0.12 * self.avg_scene_depth)

        if avg_dist > 0.12 * self.avg_scene_depth:
            # frame is a keyframe
            frame.is_keyframe_ = True
            # remove farthest keyframe, and add new keyframe to map list of keyframes
            if len(self.keyframes) >= Config.MAX_KEYFRAMES:
                del self.keyframes[max_index]
            self.addKeyFrame(frame)
    
    def addKeyFrame(self, frame):
        self.keyframes.append(frame)