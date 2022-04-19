import cv2
import numpy as np

def createBins(width, height, number_of_bins = 30):
    horizontal_bins = np.linspace(start = 0, stop = width, num=number_of_bins, endpoint=True, dtype = 'int')
    vertical_bins = np.linspace(start = 0, stop = height, num=number_of_bins, endpoint=True, dtype = 'int')
    return horizontal_bins, vertical_bins

def load_image(path, ):
    return cv2.imread(path, 0)

def gray2RGB(image):
    return cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

def RGB2Gray(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

def debug_bins(image, horizontal_bins, vertical_bins, width, height, draw_output = False, print_bins = False):
    if print_bins:
        print(f'{horizontal_bins.shape=},{horizontal_bins=}')
        print(f'{vertical_bins.shape=}, {vertical_bins=}')

    # Coordinates must be a tuple - (x,y)
    if draw_output:
        for hor in horizontal_bins:
            cv2.line(image,(0, hor),(width, hor),(0,0,255),thickness=1) #Color is by default black
        for ver in vertical_bins:
            cv2.line(image,(ver, 0),(ver, height),(255,0,0),thickness=1) #Color is by default black
    
def save_image(filename, img):
    cv2.imwrite(filename, img)

def pre_binning(row_bins, col_bins):
    bins = []
    for i_r in range(row_bins.shape[0] - 1):
        for i_c in range(col_bins.shape[0] - 1):
            start_row = row_bins[i_r]
            end_row = row_bins[i_r + 1]
            start_col = row_bins[i_c]
            end_col = row_bins[i_c + 1]
            bins.append([start_row, end_row, start_col, end_col])
    return bins
        

# def display_image(image, window_name):
#     cv2.imshow(window_name, image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

if __name__ == "__main__":
    w, h = createBins(275, 350, 30)
    print(w)
    print(h)
