import cv2
import numpy as np

from util import loadImage, debugBins, gray2RGB, saveImage, RGB2Gray

from frame import Frame
from config import Config

from feature_detector import FeatureDetector


class FeatureTracker:
    def __init__(self):
        self.lk_params = dict(
            winSize=(
                Config.FeatureTracker.WINDOW_SIZE,
                Config.FeatureTracker.WINDOW_SIZE,
            ),
            maxLevel=Config.FeatureTracker.PYRAMID_MAX_LEVEL,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                Config.FeatureTracker.ITERATIONS,
                Config.FeatureTracker.EPSILON,
            ),
        )

    def trackFeatures(self, prev_frame, curr_frame):
        kps, status, error = cv2.calcOpticalFlowPyrLK(
            prev_frame.image_,
            curr_frame.image_,
            prev_frame.np_keypoints_,
            None,
            **self.lk_params
        )

        status = status.reshape(status.shape[0])
        prev_frame.np_keypoints_ = prev_frame.np_keypoints_[status == 1]
        curr_frame.np_keypoints_ = kps[status == 1]

    def drawTrackedFeature(self, prev_frame, current_frame, feature_detector):
        debug_img = feature_detector.drawKeypoints(current_frame)
        for (x0, y0), (x1, y1) in zip(
            prev_frame.np_keypoints_.astype(int),
            current_frame.np_keypoints_.astype(int),
        ):
            cv2.line(debug_img, (x0, y0), (x1, y1), (0, 255, 0))
        return debug_img


if __name__ == "__main__":
    webcam = cv2.VideoCapture(0)

    if not webcam.read()[0]:
        print("Webcam not found.")
    else:
        [_, image] = webcam.read()
        initial_frame = Frame(RGB2Gray(image))
        height, width, channels = image.shape

        fd = FeatureDetector(width, height)
        fd.detectKeypoints(initial_frame)

        ft = FeatureTracker()

        while True:
            [_, image] = webcam.read()
            height, width, channels = image.shape

            gray_image = RGB2Gray(image)
            current_frame = Frame(gray_image)

            ft.trackFeatures(initial_frame, current_frame)

            print(current_frame.np_keypoints_.shape)
            print()

            debug_img = ft.drawTrackedFeature(initial_frame, current_frame, fd)

            cv2.imshow("tracked_features", debug_img)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27 or ch == ord("q"):  # escape key or q
                cv2.destroyAllWindows()
                break
            elif ch == ord("r"):
                print("Reset Tracked Frame")
                initial_frame = current_frame
                fd = FeatureDetector(width, height)
                fd.detectKeypoints(initial_frame)
