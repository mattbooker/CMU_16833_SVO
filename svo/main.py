<<<<<<< Updated upstream
=======
from sys import _current_frames
import numpy as np
>>>>>>> Stashed changes
from pathlib import Path
import cv2

<<<<<<< Updated upstream
data_dir = "/mav0/cam0/data"

def run():
    cur_dir = Path(__name__)
    data_dir = cur_dir.parent / "mav0/cam0/data"
    
=======
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

def processFrame(prev_frame: Frame, cur_frame: Frame):
    print("Processing Frame...")

    cur_frame.T_w_f_ = prev_frame.T_w_f_

    # Manage keyframes
    T = ImageAlignment().findAlignment(prev_frame, cur_frame)

    return True

def run(current_stage = Stage.PROCESS_FIRST_FRAME):
    cur_dir = Path(__file__)
    data_dir = cur_dir.parent.parent / "data"
    
    last_frame = None
    current_frame = None

    print(cur_dir.parent)

>>>>>>> Stashed changes
    for filename in sorted(data_dir.glob("*")):
        img = cv2.imread(str(filename))

        cv2.imshow("a", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()