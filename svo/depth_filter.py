import numpy as np
from filter import Filter
from scipy.spatial.distance import cdist
from config import Config
from camera import Camera
from pathlib import Path
from matcher import Matcher

class DepthFilter:

    cam = Camera(str(Path(__name__).parent / "mav0/cam0/sensor.yaml"))

    def __init__(self):
        self.filters = []
        self.px_error_angle = np.arctan(1/(2*DepthFilter.cam.intrinsics[0, 0])*2)

    def processFrame(self, frame, map):

        if frame.is_keyframe_:
            self.addKeyFrame(frame, map)

        else: self.updateFilters(frame)

    def addKeyFrame(self, frame, map):
        # For each NEW feature (i.e those not matched to map):
        #   create new filter
        n, _ = map.points.shape
        P = DepthFilter.cam.getProjectionMatrix(frame.T_w_f_) # 3x4
        x = np.hstack((map.points, np.ones((n, 1)))) # nx4
        u = (P @ x.T).T # nx3
        u = u[:, :-1]/u[:, -1] # normalize image points - drop last column - nx2 - map points projected in current frame

        # Calc distances between all new feature points and projected map points.
        # Check if there is a feature close by - get the closest feature distance. If less than threshold, skip. Else, add a new filter
        min_dists = np.amin(cdist(frame.np_keypoints_, u, metric='euclidean'), axis=1)
        for idx, val in enumerate(min_dists):
            if val > Config.DepthFilter.DIST_THRESH: # feature not in map - no feature close by - create new filter
                f = Filter(map.avg_scene_depth, np.amin(map.points[:, -1]), np.amax(map.points[:, -1]), frame, frame.np_keypoints_[idx, :])
                self.filters.append(f)


    def updateFilters(self, frame, map):
        # For each filter currently stored:
            # check if filter is too old (remove if too old)
            # check if filter is of a point that is visible in this frame (skip if not in frame)
            # 
            # Attempt to match fillter to feature in frame (use epipolar line between this frame and the filters frame)
            #   Skip if not matched and increment outlier count
            #   
            #   If matched then compute tau and update filter
            # 
        
        updated_filters = []
        for f in self.filters:
            if f.ref_keyframe.id >= map.keyframe_ids[0]: # only consider non-old filters
                # check if filter is assoiated to the last added keyframe - if yes - check if point is visible in this frame
                if f.ref_keyframe.id == map.keyframe_ids[-1]:
                    x = DepthFilter.cam.backProjection(f.feature_point, f.mean, f.ref_keyframe.T_w_f_) # back project filter feature to world
                    u = DepthFilter.cam.project(x, frame.T_w_f_) # project world point to current frame
                    min_dist = np.amin(cdist(u.reshape(-1, 2), frame.np_keypoints_, metric='euclidean'))
                    
                    if min_dist <= Config.DepthFilter.DIST_THRESH: # point is visible in current frame
                        m = Matcher(f.ref_keyframe, frame) # initialize matcher
                        x2, y2, sad = m.searchEpipolarLine(f.feature_point, np.amin(map.points[:, -1]), np.amax(map.points[:, -1])) # search along the epipolar line
                        
                        # Triangulate to get estimated depth. Update the filter
                        # TODO: check x2, y2 ordering
                        triangulated_point = m.triangulate(f.feature_point.reshape(-1, 2), np.array([[x2, y2]]))

                        triangulated_point = triangulated_point[:-1]/triangulated_point[-1]

                        estimated_depth = triangulated_point[-1]

                        # TODO: update tau squared before update

                        f.update(estimated_depth, _)

                        # TODO: check if variance is low enough - if yes, add filter point to map, remove from list

                updated_filters.append(f) # append filter to updated list
                        
        
        self.filters = updated_filters # update filters
    
    def calcTau(self, transform, f, z):
        pass