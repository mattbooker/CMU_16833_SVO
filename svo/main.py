import numpy as np
from pathlib import Path
from re import L
import cv2
from enum import Enum
from frame import Frame
from config import Config
from depth_filter import DepthFilter
from feature_detector import FeatureDetector
from feature_tracker import FeatureTracker
from map import Map
from image_alignment import ImageAlignment
from camera import Camera

data_dir = "../data/"

ft = FeatureTracker()
fd = None
map = Map()
depth_filter = DepthFilter(map)
image_aligner = ImageAlignment()
cam = Camera()

class Stage(Enum):
    PROCESS_FIRST_FRAME = 0
    PROCESS_SECOND_FRAME = 1
    PROCESS_FRAMES = 2

def processFirstFrame(cur_frame: Frame):
    '''
    current_frame: Frame
        with image set as grayscale
    '''

    print("Processing First Frame...")
    fd.detectKeypoints(cur_frame)

    if len(cur_frame.np_keypoints_) < Config.MIN_NUMBER_FEATURES:
        print(f'Not enough features in first frame. Have = {len(cur_frame.np_keypoints_)}')
        return False

    map.addKeyFrame(cur_frame)
    cur_frame.setKeyFrame()
    return True

def processSecondFrame(prev_frame:Frame, cur_frame: Frame):
    print("Processing Second Frame...")

    ft.trackFeatures(prev_frame, cur_frame)

    # TODO: Use homography to compute transform
    H, status = cv2.findHomography(prev_frame.np_keypoints_, cur_frame.np_keypoints_, cv2.RANSAC, Config.REPROJECTION_THRESHOLD)
    num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(H, cam.intrinsics)


    I = np.hstack([np.eye(3), np.zeros((3,1))])
    prev_P = cam.getProjectionMatrix(I)

    final_R = None
    final_t = None
    final_world_pts = None
    
    # TODO: Scale map
    # status = status.ravel() != 0
    # prev_inliers = prev_frame.np_keypoints_[status]
    # cur_inliers = cur_frame.np_keypoints_[status]

    for n, R,t  in zip(range(num), Rs, Ts):
        ext = np.hstack((R.T, -R.T @ t))
        cur_P = cam.getProjectionMatrix(ext)
        
        world_pts = cv2.triangulatePoints(prev_P, cur_P, prev_frame.np_keypoints_.T, cur_frame.np_keypoints_.T)
        world_pts = (world_pts[:-1]/world_pts[-1]).T

        if np.any(world_pts[:, -1] < 0):
            print(n)
            continue

        final_R = R
        final_t = t
        final_world_pts = world_pts

    T = np.vstack([np.hstack([final_R.T, -final_R @ final_t]), [0,0,0,1]])
    cur_frame.T_w_f_ = T

    map.initial_map(final_world_pts)
    
    cur_frame.setKeyFrame()
    map.addKeyFrame(cur_frame)
    depth_filter.addKeyFrame(cur_frame)

    return True

def processFrame(cur_frame: Frame):
    print("Processing Frame...")

    # Manage keyframes

    return True

def run(current_stage = Stage.PROCESS_FIRST_FRAME):
    cur_dir = Path(__name__)
    data_dir = cur_dir.parent / "data"
    
    last_frame = None
    current_frame = None

    for filename in sorted(data_dir.glob("*")):
        img = cv2.imread(str(filename), 0)

        global fd

        if fd is None:
            fd = FeatureDetector(img.shape[1], img.shape[0])

        current_frame = Frame(img)

        if current_stage == Stage.PROCESS_FIRST_FRAME:
            if processFirstFrame(current_frame):
                current_stage = Stage.PROCESS_SECOND_FRAME

        elif current_stage == Stage.PROCESS_SECOND_FRAME:
            if processSecondFrame(last_frame, current_frame):
                current_stage = Stage.PROCESS_FRAMES

        elif current_stage == Stage.PROCESS_FRAMES:
            if processFrame(current_frame):
                pass


        # cv2.imshow("op", current_frame.image_)
        debug_img = fd.drawKeypoints(current_frame)
        if last_frame is not None:
            cv2.imshow("a", ft.drawTrackedFeature(last_frame, current_frame, fd))
        else:
            cv2.imshow("a", debug_img)
        cv2.waitKey(0)

        last_frame = current_frame

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()