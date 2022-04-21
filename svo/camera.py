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
        # self.extrinsics = np.hstack([np.eye(3), np.zeros((3,1))])

        # self.R = self.extrinsics[:3, :3]
        # self.t = self.extrinsics[:, -1].reshape((3,1))
        
        self.intrinsics = np.eye(3)
        self.intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]] = np.array(self.data['intrinsics'])
        
        # self.P = self.intrinsics @ self.extrinsics

    # Project world point into image to get pixel coordinates
    def project(self, transform, world_pt):
        homogeneous_world_pt = np.vstack([world_pt, 1])

        homogeneous_image_pt = self.intrinsics @ transform @ homogeneous_world_pt
        image_pt = homogeneous_image_pt[:-1] / homogeneous_image_pt[-1]

        return image_pt
        
    # Back project pixel coordinates to a world point at given depth
    def backProjection(self, image_pt, depth, transform):
        R = transform[:3, :3]
        t = transform[:, -1].reshape((3,1))
        homogeneous_image_pt = np.vstack([image_pt, 1])

        return depth * R.T @ np.linalg.inv(self.intrinsics) @ homogeneous_image_pt - R.T @ t

    def getProjectionMatrix(self, transform_world_to_frame):
        return self.intrinsics @ transform_world_to_frame


if __name__ == "__main__":
    from pathlib import Path
    b = Path(__name__).parent / "mav0/cam0/sensor.yaml"

    test = Camera(str(b))
    p = np.array([[367.215], [248.375]])
    transform = np.hstack([np.eye(3), np.zeros((3,1))])

    print(test.backProjection(p, 5.5, transform))

    print(test.getProjectionMatrix(transform))
