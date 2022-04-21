class Config:
    PATCH_SIZE = 8
    HALF_PATCH_SIZE = 4
    
    class Matcher:
        WINDOW_SIZE = 9

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