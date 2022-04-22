import numpy as np
from filter import Filter
from scipy.spatial.distance import cdist
from config import Config

class DepthFilter:

    def __init__(self):
        self.filters = []

    def processFrame(self, frame, map, camera):

        if frame.is_keyframe_:
            self.addKeyFrame(frame, map, camera)

        else: self.updateFilters(frame)

    def addKeyFrame(self, frame, map, camera):
        # For each NEW feature (i.e those not matched to map):
        #   create new filter
        n, _ = map.points.shape
        P = camera.getProjectionMatrix(frame.T_w_f_) # 3x4
        x = np.hstack((map.points, np.ones((n, 1)))) # nx4
        u = (P @ x.T).T # nx3
        u = u/u[:, -1] # normalize image points
        u = u[:, :-1] # drop last column - nx2 - map points projected in current frame

        # Calc distances between all new feature points and projected map points.
        # Check if there is a feature close by - get the closest feature distance. If less than threshold, skip. Else, add a new filter
        min_dists = np.amin(cdist(frame.np_keypoints_, u, metric='euclidean'), axis=1)
        for idx, val in enumerate(min_dists):
            if val > Config.DepthFilter.DIST_THRESH: # feature not in map - no feature close by - create new filter
                f = Filter(map.avg_scene_depth, np.amin(map.points[:, -1]), np.amax(map.points[:, -1]), frame.id, frame.np_keypoints_[idx, :])
                self.filters.append(f)


    def updateFilters(self, frame):
        # For each filter currently stored:
            # check if filter is too old (remove if too old)
            # check if filter is of a point that is visible in this frame (skip if not in frame)
            # 
            # Attempt to match fillter to feature in frame (use epipolar line between this frame and the filters frame)
            #   Skip if not matched and increment outlier count
            #   
            #   If matched then compute tau and update filter
            # 
        pass