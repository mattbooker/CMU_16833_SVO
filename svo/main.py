from pathlib import Path
import cv2

data_dir = "/mav0/cam0/data"

def run():
    cur_dir = Path(__name__)
    data_dir = cur_dir.parent / "mav0/cam0/data"
    
    for filename in sorted(data_dir.glob("*")):
        img = cv2.imread(str(filename))

        cv2.imshow("a", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()