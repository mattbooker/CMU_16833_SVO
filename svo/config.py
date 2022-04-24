from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    PATCH_SIZE = 8
    HALF_PATCH_SIZE = 4

    MIN_NUMBER_FEATURES = 10
    REPROJECTION_THRESHOLD = 1.0
    
    class Matcher:
        WINDOW_SIZE = 9

    class FeatureDetector:
        BINS = 10
        NON_MAX_SUPPRESSION = True
        THRESHOLD = 30
        BORDER_THREHOLD = 10

    class FeatureTracker:
        EPSILON = 0.001
        ITERATIONS = 30
        WINDOW_SIZE = 30
        PYRAMID_MAX_LEVEL = 4

    class Map:
        VAR_THRESH = 10
        MAX_KEYFRAMES = 10
        KEYFRAME_THRESH = 0.12
        MIN_KEYPOINTS = 50


    class DepthFilter:
        DIST_THRESH = 10.0

    class Camera:
        # INTRINSICS = np.eye(3)
        INTRINSICS = np.array([[315.5, 0, 376.0],
                                [0, 315.5, 240.0],
                                [0, 0, 1]])

    class ImageAlignment:
        THRESHOLD = 0.025
        MAX_ITER = 100
        WINDOW_SIZE = 4