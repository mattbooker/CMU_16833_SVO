import cv2
import numpy as np

from util import createBins, loadImage, debugBins, gray2RGB, saveImage, RGB2Gray

from frame import Frame
from config import Config


class FeatureDetector:
    def __init__(self, width, height):
        self.fast = cv2.FastFeatureDetector_create()
        self.fast.setNonmaxSuppression(Config.FeatureDetector.NON_MAX_SUPPRESSION)
        self.fast.setThreshold(Config.FeatureDetector.THRESHOLD)

        self.col_bins, self.row_bins = createBins(
            width, height, number_of_bins=Config.FeatureDetector.BINS
        )
        
    def cvKeyPointsToNPArray(self, keypoints):
        kpts_np = np.array([kp.pt for kp in keypoints])
        kpts_scores = np.array([kp.response for kp in keypoints])
        return kpts_np, kpts_scores

    def detectKeypoints(self, frame):
        opencv_kps_list = []
        np_kps_list = []

        keypoints = self.fast.detect(frame.image_)
        np_kps, np_kps_scores = self.cvKeyPointsToNPArray(keypoints)

        # TODO Optimize (since bins are precalculated)
        for i_r in range(self.row_bins.shape[0] - 1):
            for i_c in range(self.col_bins.shape[0] - 1):
                start_row = self.row_bins[i_r]
                end_row = self.row_bins[i_r + 1]

                start_col = self.col_bins[i_c]
                end_col = self.col_bins[i_c + 1]

                col_indexes = (np_kps[:, 0] >= start_col) & (np_kps[:, 0] < end_col)
                row_indexes = (np_kps[:, 1] >= start_row) & (np_kps[:, 1] < end_row)

                index = row_indexes & col_indexes
                if np.any(index):
                    max_index = np.argmax(np_kps_scores[index])

                    np_kps_list.append(np_kps[index][max_index])

                    opencv_kps_list.append(keypoints[np.where(index)[0][max_index]])

        frame.opencv_keypoints_ = opencv_kps_list
        frame.np_keypoints_ = np.array(np_kps_list, dtype = np.float32)

    def drawKeypoints(self, frame):
        debug_img = gray2RGB(frame.image_.copy())

        for kp in frame.np_keypoints_.astype(int):

            cv2.circle(
                debug_img, kp, radius=2, color=(0, 255, 0), thickness=1
            )

            strt = kp - 5
            end = kp + 5

            cv2.line(debug_img, strt, end, color=(0, 255, 0), thickness=2)
            cv2.line(
                debug_img,
                (end[0], strt[1]),
                (strt[0], end[1]),
                color=(0, 255, 0),
                thickness=2,
            )

        debugBins(
            debug_img, self.col_bins, self.row_bins, debug_img.shape[1], debug_img.shape[0], draw_output=True
        )

        return debug_img


if __name__ == "__main__":
    webcam = cv2.VideoCapture(0)

    if not webcam.read()[0]:
        print("Webcam not found.")
    else:
        while True:
            [_ ,image] = webcam.read()
            height, width, channels = image.shape

            gray_image = RGB2Gray(image)
            frame = Frame(gray_image)

            fd = FeatureDetector(width=width, height=height)
            fd.detectKeypoints(frame)
            debug_img = fd.drawKeypoints(frame)

            cv2.imshow("features", debug_img)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27 or ch == ord('q'): # escape key or q
                cv2.destroyAllWindows()
                break
