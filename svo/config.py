class Config:
    patch_size = 8
    half_patch_size = 4
    
    class Matcher:
        window_size = 15

    class FeatureDetector:
        BINS = 10
        NON_MAX_SUPPRESSION = True
        THRESHOLD = 5

    class FeatureTracker:
        EPSILON = 0.001
        ITERATIONS = 30
        WINDOW_SIZE = 30
<<<<<<< Updated upstream
        PYRAMID_MAX_LEVEL = 4
=======
        PYRAMID_MAX_LEVEL = 4

    class Map:
        VAR_THRESH = 0.1

    class DepthFilter:
        DIST_THRESH = 0.1

    class Camera:
        # INTRINSICS = np.eye(3)
        INTRINSICS = np.array([[315.5, 0, 376.0],
                                [0, 315.5, 240.0],
                                [0, 0, 1]])

    class ImageAlignment:
        THRESHOLD = 0.25
        MAX_ITER = 10
        WINDOW_SIZE = 4
>>>>>>> Stashed changes
