import numpy as np

class Map:

    def __init__(self):
        self.map_points = np.empty((0, 3)) # 3D points added to the map
        self.avg_scene_depth = 0 # current scene depth
    
    def initial_map(self):
        # triangulate initial map from first two views
        pass