import cv2
import numpy as np

from util import createBins, load_image, debug_bins, gray2RGB, save_image

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

    def cv_kpts_to_np_kp_and_scores(self, keypoints):
        kpts_np = np.array([kp.pt for kp in keypoints])
        kpts_scores = np.array([kp.response for kp in keypoints])
        return kpts_np, kpts_scores

    def detectKeypoints(self, frame):
        kps_list = []
        kps_scores_list = []

        keypoints = self.fast.detect(frame.image_)
        np_kps, np_kps_scores = self.cv_kpts_to_np_kp_and_scores(keypoints)

        # TODO Optimize (since bins are precalculated)
        for i_r in range(self.row_bins.shape[0] - 1):
            for i_c in range(self.col_bins.shape[0] - 1):
                start_row = self.row_bins[i_r]
                end_row = self.row_bins[i_r + 1]

                start_col = self.col_bins[i_c]
                end_col = self.col_bins[i_c + 1]

                row_indexes = (np_kps[:, 0] >= start_row) & (np_kps[:, 0] < end_row)
                col_indexes = (np_kps[:, 1] >= start_col) & (np_kps[:, 1] < end_col)

                max_index = np.argmax(np_kps_scores[row_indexes & col_indexes])

                max_kp = np_kps[row_indexes & col_indexes][max_index]
                max_kp_score = np_kps_scores[row_indexes & col_indexes][max_index]

                kps_list.append(max_kp)
                kps_scores_list.append(max_kp_score)

        frame.keypoints_ = kps_list
        frame.kp_scores_ = kps_scores_list

    def drawKeypoints(self, frame):
        debug_img = gray2RGB(frame.image_.copy())

        for kp in frame.keypoints_:
            cv2.circle(
                debug_img, kp.astype(int), radius=2, color=(0, 255, 0), thickness=1
            )

            strt = kp.astype(int) - 5
            end = kp.astype(int) + 5

            cv2.line(debug_img, strt, end, color=(0, 255, 0), thickness=2)
            cv2.line(
                debug_img,
                (end[0], strt[1]),
                (strt[0], end[1]),
                color=(0, 255, 0),
                thickness=2,
            )

        debug_bins(
            debug_img, self.col_bins, self.row_bins, width, height, draw_output=True
        )

        return debug_img


if __name__ == "__main__":
    image_path = "lena.jpg"
    image = load_image(path=image_path)
    width, height = image.shape
    frame = Frame(image)

    fd = FeatureDetector(width=width, height=height)
    fd.detectKeypoints(frame)
    debug_img = fd.drawKeypoints(frame)

    cv2.imshow("features", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
