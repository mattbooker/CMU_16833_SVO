import numpy as np
from filter import Filter
from scipy.spatial.distance import cdist
from config import Config
from camera import Camera
from pathlib import Path
from matcher import Matcher


class DepthFilter:

    cam = Camera()

    def __init__(self):
        self.filters = []

    def processFrame(self, frame, map):

        if frame.is_keyframe_:
            self.addKeyFrame(frame, map)

        else:
            self.updateFilters(frame)

    def addKeyFrame(self, frame, map):
        # For each NEW feature (i.e those not matched to map):
        #   create new filter
        n, _ = map.points.shape
        P = DepthFilter.cam.getProjectionMatrix(frame.T_w_f_)  # 3x4
        x = np.hstack((map.points, np.ones((n, 1))))  # nx4
        u = (P @ x.T).T  # nx3
        # normalize image points - drop last column - nx2 - map points projected in current frame
        u = u[:, :-1]/u[:, -1]

        # Calc distances between all new feature points and projected map points.
        # Check if there is a feature close by - get the closest feature distance. If less than threshold, skip. Else, add a new filter
        min_dists = np.amin(cdist(frame.np_keypoints_, u,
                            metric='euclidean'), axis=1)
        for idx, val in enumerate(min_dists):
            if val > Config.DepthFilter.DIST_THRESH:  # feature not in map - no feature close by - create new filter
                f = Filter(map.avg_scene_depth, np.amin(
                    map.points[:, -1]), np.amax(map.points[:, -1]), frame, frame.np_keypoints_[idx, :])
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
        for filter in self.filters:

            # If filters keyframe is older than oldest keyframe then ignore
            if filter.ref_keyframe.id < map.keyframe_ids[0]:
                # TODO: Should remove the filter
                continue

            # Check if point is behind camera
            point_in_cur_frame = DepthFilter.cam.backProjection(
                filter.feature_point, filter.getDepth(), filter.ref_keyframe.T_w_f_ @ np.inv(frame.T_f_w_))
            if point_in_cur_frame[3] < 0:
                continue

            # Check if the point is outside camera view
            world_pt = DepthFilter.cam.backProjection(
                filter.feature_point, filter.mean, filter.ref_keyframe.T_w_f_)
            if not DepthFilter.cam.isInFrame(frame, world_pt):
                continue

            updated_filters.append(filter)

            # check if filter is assoiated to the last added keyframe
            if filter.ref_keyframe.id == map.keyframe_ids[-1]:
                image_coords = DepthFilter.cam.project(world_pt, frame.T_w_f_)
                min_dist = np.amin(
                    cdist(image_coords.reshape(-1, 2), frame.np_keypoints_, metric='euclidean'))

                # Filter is close enough to a keypoint in the current frame to be processed
                if min_dist <= Config.DepthFilter.DIST_THRESH:
                    m = Matcher(filter.ref_keyframe, frame)

                    # We use 2 std dev away from mean to get min and max depths
                    min_depth = filter.getDepth() - 2*filter.getStdDev()
                    max_depth = filter.getDepth() + 2*filter.getStdDev()

                    # Search along the epipolar line to find best corresponding point
                    x2, y2, sad = m.searchEpipolarLine(
                        filter.feature_point, min_depth, max_depth)
                    
                    estimated_depth = m.triangulate(filter.feature_point, np.array([x2, y2]))

                    tau_sq = self.computeTauSq(filter.ref_keyframe, frame, world_pt)

                    filter.update(estimated_depth, tau_sq)

        self.filters = updated_filters  # update filters

    
    def computeTauSq(self, ref_frame, cur_frame, world_pt):
        '''
        Refer to https://www.researchgate.net/publication/262378171_REMODE_Probabilistic_Monocular_Dense_Reconstruction_in_Real_Time
        for derivation.
        '''

        T_ref_cur = ref_frame.T_w_f_ @ np.inv(cur_frame.T_f_w_)
        t = T_ref_cur[:, -1]
        a = world_pt - t
        f = (world_pt / world_pt[-1])

        # Use fx as focal length
        focal_length = DepthFilter.cam.intrinsics[0,0]

        alpha = np.arccos(f.dot(a) / np.linalg.norm(t))
        beta = np.arccos(-a.dot(t)) / (np.linalg.norm(a) * np.linalg.norm(t))
        beta_plus = beta + 2 * np.arctan(1/(2 * focal_length))
        gamma = np.pi - alpha - beta_plus

        rP_plus_norm = np.linalg.norm(t) * np.sin(beta_plus) / np.sin(gamma)

        return np.power(rP_plus_norm - np.linalg.norm(world_pt), 2)

