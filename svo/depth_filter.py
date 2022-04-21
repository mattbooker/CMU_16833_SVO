from filter import Filter

class DepthFilter:

    def __init__(self, map):
        self.ref_keyframe = 0
        self.filters = []

    def processFrame(self, frame):
        # self.updateFilters(frame)

        # if frame.isKeyFrame():
        #     self.addKeyFrame(frame)
        pass

    def addKeyFrame(self, frame, avg_depth, min_depth, max_depth):
        # For each NEW feature (i.e those not matched to map):
        #   create new filter
        pass

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