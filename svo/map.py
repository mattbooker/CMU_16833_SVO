import numpy as np

class Map:

    def __init__(self):
        self.points = np.empty((0, 3)) # 3D points added to the map
        self.avg_scene_depth = 0 # current scene depth
        self.keyframes = [] # stored keyframes - total 10 (based on paper)
    
    def initial_map(self):
        # triangulate initial map from first two views
        pass

    def checkKeyframe(self, frame):
        total_dist = 0
        max_dist = 0
        for idx, keyframe in enumerate(self.keyframes):
            # Calculate distance between the frame and the keyframe
            T_frame = keyframe.T_w_f_ @ np.inv(frame.T_w_f_)
            dist = np.linalg.norm(T_frame[:-1, -1])
            if dist > max_dist:
                max_dist = dist
                max_index = idx
            total_dist += dist
        
        avg_dist = total_dist/len(self.keyframes)

        if avg_dist > 0.12 * self.avg_scene_depth:
            # frame is a keyframe
            frame.is_keyframe_ = True
            # remove farthest keyframe, and add new keyframe to map list of keyframes 
            del self.keyframes[max_index]
            self.keyframes.append(frame)