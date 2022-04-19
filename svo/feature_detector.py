import cv2
import numpy as np

from util import createBins, load_image, debug_bins, gray2RGB

from frame import Frame
from config import Config

class FeatureDetector:
    def __init__(self, width, height):
        self.fast = cv2.FastFeatureDetector_create()
        self.fast.setNonmaxSuppression(Config.FeatureDetector.NON_MAX_SUPPRESSION)
        self.fast.setThreshold(Config.FeatureDetector.THRESHOLD)

        self.col_bins, self.row_bins = createBins(width, height, number_of_bins=Config.FeatureDetector.BINS)

    def cv_kpts_to_np_kp_and_scores(self, keypoints):
        kpts_np = np.array([kp.pt for kp in keypoints])
        kpts_scores = np.array([kp.response for kp in keypoints])
        return kpts_np, kpts_scores

    def detectKeypoints(self, frame):
        kps = []
        kps_scores = []

        # TODO Optimize (since bins are precalculated)
        for i_r in range(self.row_bins.shape[0] - 1):
            for i_c in range(self.col_bins.shape[0] - 1):
                start_row = self.row_bins[i_r]
                end_row = self.row_bins[i_r + 1]

                start_col = self.col_bins[i_c]
                end_col = self.col_bins[i_c + 1]

                img_patch = frame.image_[start_row:end_row, start_col:end_col]
                img_patch_kp_pts = self.fast.detect(img_patch)

                if img_patch_kp_pts:
                    kpts_np, kpts_scores = self.cv_kpts_to_np_kp_and_scores(img_patch_kp_pts)
                    highest_score_index = np.argmax(kpts_scores)
                    kp = kpts_np[highest_score_index]
                    kp[1] += start_row
                    kp[0] += start_col

                    kps.append(kp)
                    kps_scores.append(kpts_scores[highest_score_index])

        frame.keypoints_ = kps
        frame.kp_scores_ = kps_scores

    def drawKeypoints(self, frame):
        debug_img = gray2RGB(frame.image_.copy())
        
        for kp in frame.keypoints_:
            cv2.circle(debug_img, kp.astype(int), radius=2, color=(0, 255, 0), thickness=1)

            strt = kp.astype(int) - 5
            end = kp.astype(int)  + 5

            cv2.line(debug_img, strt, end, color=(0, 255, 0), thickness=2)
            cv2.line(debug_img, (end[0], strt[1]), (strt[0], end[1]), color=(0, 255, 0), thickness=2)

        debug_bins(debug_img, self.col_bins, self.row_bins, width, height, draw_output=True)

        return debug_img
    

if __name__ == "__main__":
    image_path = "lena.jpg"
    image = load_image(path = image_path)
    width, height = image.shape
    frame = Frame(image)

    fd = FeatureDetector(width=width, height=height)
    fd.detectKeypoints(frame)
    debug_img = fd.drawKeypoints(frame)

    cv2.imshow("features_binned", debug_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()