import cv2
import numpy as np
from util import createBins, load_image, debug_bins, gray2RGB, save_image

# image_path = "lena.jpg"
image_path = "blox.jpg"

class FeatureDetector:
    def __init__(self, image):
        self.fast = cv2.FastFeatureDetector_create()
        self.fast.setNonmaxSuppression(1)
        self.fast.setThreshold(5)

        self.kp_pts = None
        self.image = image

    def cv_kpts_to_np_kp_and_scores(self):
        kpts_np = np.array([kp.pt for kp in self.kp_pts])
        kpts_scores = np.array([kp.response for kp in self.kp_pts])
        return kpts_np, kpts_scores

    def detectKeypoints(self):
        self.kp_pts = self.fast.detect(self.image)
    
    def getFastKeypoints(self):
        return self.kp_pts

    def drawKeypoints(self, image):
        cv2.drawKeypoints(image, self.kp_pts, image, color = (255, 0, 0))

if __name__ == "__main__":
    image = load_image(path = image_path)
    color_image = gray2RGB(image)
    width, height = image.shape
    col_bins, row_bins = createBins(width, height, number_of_bins=10)
    debug_bins(color_image, col_bins, row_bins, width, height, draw_output=True)

    kps = []
    scores = []
    for i_r in range(row_bins.shape[0] - 1):
        for i_c in range(col_bins.shape[0] - 1):
            start_row = row_bins[i_r]
            end_row = row_bins[i_r + 1]
            start_col = row_bins[i_c]
            end_col = row_bins[i_c + 1]
            # print(f'{start_row=}, {end_row=}, {start_col=}, {end_col=}')
            img_patch = image[start_row:end_row, start_col:end_col]
            fd = FeatureDetector(img_patch)
            fd.detectKeypoints()

            if fd.getFastKeypoints():
                kpts_np, kpts_scores = fd.cv_kpts_to_np_kp_and_scores()
                highest_score_index = np.argmax(kpts_scores)
                kp = kpts_np[highest_score_index]
                kp[1] += start_row
                kp[0] += start_col

                kps.append(kp)
                scores.append(kpts_scores[highest_score_index])

            # cv2.imshow("patch", img_patch)       
            # cv2.waitKey(0)

    for kp in kps:
        cv2.circle(color_image, kp.astype(int), radius=2, color=(0, 255, 0), thickness=1)

        strt = kp.astype(int) - 5
        end = kp.astype(int)  + 5

        cv2.line(color_image, strt, end, color=(0, 255, 0), thickness=2)
        cv2.line(color_image, (end[0], strt[1]), (strt[0], end[1]), color=(0, 255, 0), thickness=2)
  

    
    cv2.imshow("features_binned", color_image)
    save_image("features_binned.jpg", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()