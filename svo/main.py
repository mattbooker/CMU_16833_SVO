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
        ext = np.hstack((R,t))
        cur_P = cam.getProjectionMatrix(ext)
        
        world_pts = cv2.triangulatePoints(prev_P, cur_P, prev_frame.np_keypoints_.T, cur_frame.np_keypoints_.T)
        world_pts = (world_pts[:-1]/world_pts[-1]).T

        if np.any(world_pts[:, -1] < 0):
            continue

        final_R = R
        final_t = t
        final_world_pts = world_pts

    T = np.vstack([np.hstack([final_R, final_t]), [0,0,0,1]])
    cur_frame.T_w_f_ = T

    map.initial_map(final_world_pts)

    cur_frame.setKeyFrame()
    map.addKeyFrame(cur_frame)
    depth_filter.addKeyFrame(cur_frame)

    return True

def processFrame(prev_frame: Frame, cur_frame: Frame):
    print("Processing Frame...")

    ft.trackFeatures(prev_frame, cur_frame)

    cur_frame.T_w_f_ = prev_frame.T_w_f_

    # T = image_aligner.findAlignment(prev_frame, cur_frame)
    H, status = cv2.findHomography(prev_frame.np_keypoints_, cur_frame.np_keypoints_, cv2.RANSAC, Config.REPROJECTION_THRESHOLD)
    num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(H, cam.intrinsics)

    I = np.hstack([np.eye(3), np.zeros((3,1))])
    prev_P = cam.getProjectionMatrix(I)

    final_R = None
    final_t = None
    
    # TODO: Scale map

    status = status.ravel() != 0
    prev_inliers = prev_frame.np_keypoints_[status]
    cur_inliers = cur_frame.np_keypoints_[status]

    for n, R,t  in zip(range(num), Rs, Ts):
        ext = np.hstack((R, t))
        cur_P = cam.getProjectionMatrix(ext)
        
        world_pts = cv2.triangulatePoints(prev_P, cur_P, prev_inliers.T, cur_inliers.T)
        world_pts = (world_pts[:-1]/world_pts[-1]).T

        if np.any(world_pts[:, -1] < 0):

            continue

        final_R = R
        final_t = t

    final_t = final_t.reshape(3,1)
    T = np.vstack([np.hstack([final_R, final_t]), [0,0,0,1]])
    cur_frame.T_w_f_ = T

    map.checkKeyframe(cur_frame)

    if cur_frame.is_keyframe_:
        print("New keyframe")
        fd.detectKeypoints(cur_frame)

    depth_filter.processFrame(cur_frame)

    print(T[:-1, -1])

    return True

def decomposeHomography(H):
    norm = np.sqrt(np.power(H[0,0],2) + np.power(H[1,1], 2) + np.power(H[2,2],2))
    norm_H = H / norm

    c1 = norm_H[:, 0]
    c2 = norm_H[:, 1]
    c3 = np.cross(c1, c2)
    
    t = norm_H[:, 2]
    R = np.zeros((3,3))
    R[0] = np.array([c1[0], c2[0], c3[0]])
    R[1] = np.array([c1[1], c2[1], c3[1]])
    R[2] = np.array([c1[2], c2[2], c3[2]])

    U, S, Vh = np.linalg.svd(R)

    R = U @ Vh

    return R, t


def run(current_stage = Stage.PROCESS_FIRST_FRAME):
    cur_dir = Path(__file__)
    data_dir = cur_dir.parent.parent / "data"
    
    last_frame = None
    current_frame = None

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    f = open("predicted_path.txt", "a")

    cum_t = np.zeros((3,))

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
            if processFrame(last_frame, current_frame):
                pass


        # cv2.imshow("op", current_frame.image_)
        # debug_img = fd.drawKeypoints(current_frame)
        # if last_frame is not None and not current_frame.is_keyframe_:
        #     cv2.imshow("a", ft.drawTrackedFeature(last_frame, current_frame, fd))
        # else:
        #     cv2.imshow("a", debug_img)
        # cv2.waitKey(0)

        cum_t += current_frame.T_w_f_[:-1, -1]

        #save output to file to compare with the groundtruth
        file_name = str(filename)[str(filename).rfind("/") + 1 : ].replace("_0.png","")
        fileString = file_name + " " + str(cum_t[0]) + " " + str(cum_t[1]) + \
                        " " + str(cum_t[2]) + "\n"
        f.write(fileString)

        ax.plot3D(cum_t[0], cum_t[1], cum_t[2], "rx")
        ax.set_ylim([-5, 5])
        ax.set_xlim([0, -20])
        ax.set_zlim([-10, 10])
        # plt.pause(0.1)

        last_frame = current_frame

    f.close()

    plt.savefig("plot.png")
    plt.show()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()