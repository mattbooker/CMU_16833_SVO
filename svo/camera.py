import yaml
import numpy as np

class Camera:
    
    def __init__(self, filename):

        with open(filename) as file:
            self.data = yaml.safe_load(file)

        ext_rows = self.data['T_BS']['rows']
        ext_cols = self.data['T_BS']['cols']

        # Reshape and drop the last row
        # self.extrinsics = np.array(self.data['T_BS']['data']).reshape((ext_cols, ext_rows))[:-1,:]
        self.extrinsics = np.hstack([np.eye(3), np.zeros((3,1))])
        
        self.intrinsics = np.eye(3)
        self.intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]] = np.array(self.data['intrinsics'])
        
        self.P = self.intrinsics @ self.extrinsics

import pathlib
a = pathlib.Path(__name__)
b = a.parent / "mav0/cam0/sensor.yaml"

test = Camera(str(b))
