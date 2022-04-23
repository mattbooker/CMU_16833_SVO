import numpy as np
from config import Config

class Camera:
    
    def __init__(self):
        
        self.intrinsics = Config.Camera.INTRINSICS

    # Project world point into image to get pixel coordinates
    def project(self, world_pt, transform):
        '''
        Transform: (4,4) np.array
            transform from world to camera frame
        world_pt: (3,) np.array
            3D point in world
        '''

        homogeneous_world_pt = np.vstack([world_pt.reshape((3,1)), 1])

        transform = transform[:-1]

        homogeneous_image_pt = self.intrinsics @ transform @ homogeneous_world_pt
        image_pt = homogeneous_image_pt[:-1] / homogeneous_image_pt[-1]

        return image_pt
        
    # Back project pixel coordinates to a world point at given depth
    def backProjection(self, image_pt, depth, transform):
        '''
        image_pt: (2,) np.array
            point in image coordinates
        depth: float
            Depth to scale ray up by
        transform: (4,4) np.array
            transform from world to camera frame
        '''

        R = transform[:3, :3]
        t = transform[:3, -1].reshape((3,1))

        homogeneous_image_pt = np.vstack([image_pt.reshape((2,1)), 1])

        return depth * R.T @ np.linalg.inv(self.intrinsics) @ homogeneous_image_pt - R.T @ t

    def getProjectionMatrix(self, transform_world_to_frame):
        '''
        transform: (4,4) np.array
            transform from world to camera frame
        '''

        return self.intrinsics @ transform_world_to_frame[:3, :]

    def isInFrame(self, frame, world_pt):
        h_pt = np.vstack([world_pt.reshape(3,1), 1])

        P = self.getProjectionMatrix(frame.T_w_f_)
        h_image_coords = P @ h_pt
        image_coords = (h_image_coords / h_image_coords[-1])[:-1]

        print(image_coords)

        x, y = image_coords

        if 0 <= x < frame.image_.shape[1] and 0 <= y < frame.image_.shape[0]:
            return True

        return False


if __name__ == "__main__":
    cam = Camera()
    cam.intrinsics = np.array([[200, 0, 320], 
                                [0, 200, 240], 
                                [0, 0, 1]])
    p = np.array([[320], [240]])
    transform = np.hstack([np.eye(3), np.zeros((3,1))])

    print(cam.backProjection(p, 5.5, transform))

    print(cam.getProjectionMatrix(transform))

    from frame import Frame
    test_frame = Frame(np.zeros([600, 400]))

    test_pt1 = np.array([1, 1, 10]).reshape(3,1)
    print(cam.isInFrame(test_frame, test_pt1))
