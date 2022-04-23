from pathlib import Path
import cv2
from enum import Enum
from frame import Frame

data_dir = "../data/"

class Stage(Enum):
    PROCESS_FIRST_FRAME = 0
    PROCESS_SECOND_FRAME = 1
    PROCESS_FRAMES = 2

def processFirstFrame(current_frame):
    print("processFirstFrame")
    return True

def processSecondFrame(current_frame):
    print("processSecondFrame")
    return True

def processFrame(current_frame):
    print("processFrame")
    return True

def run(current_stage = Stage.PROCESS_FIRST_FRAME):
    cur_dir = Path(__name__)
    data_dir = cur_dir.parent / "../data"
    
    last_frame = None
    current_frame = None

    for filename in sorted(data_dir.glob("*")):
        img = cv2.imread(str(filename), 0)

        current_frame = Frame(img)

        if current_stage == Stage.PROCESS_FIRST_FRAME:
            if processFirstFrame(current_frame):
                current_stage = Stage.PROCESS_SECOND_FRAME

        elif current_stage == Stage.PROCESS_SECOND_FRAME:
            if processSecondFrame(current_frame):
                current_stage = Stage.PROCESS_FRAMES

        elif current_stage == Stage.PROCESS_FRAMES:
            if processFrame(current_frame):
                pass

        last_frame = current_frame

        cv2.imshow("op", current_frame.image_)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()