class Config:
    patch_size = 8
    half_patch_size = 4
    
    class Matcher:
        window_size = 15

    class FeatureDetector:
        BINS = 10
        NON_MAX_SUPPRESSION = True
        THRESHOLD = 5