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
        PYRAMID_MAX_LEVEL = 4

    class Map:
        VAR_THRESH = 0.1

    class DepthFilter:
        DIST_THRESH = 0.1