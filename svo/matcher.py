import cv2
import numpy as np


class Matcher:

    def __init__():
        pass

    def searchEpipolarLine(self, ref_frame, cur_frame, ref_feature, depth_estimate, min_depth, max_depth):
        # Calculate fundamental matrix between ref_frame and cur_frame

        # Use fundamental matirx to get epipolar line

        # Use minimum depth to find start point of search on epipolar line
        # Use maximum depth to find end point of search on epipolar line
        #   1. Use feature point (u,v) and projection matrix to find (x,y,1) in world space
        #   2. Scale up point to required depth and project point into second camera
        #   3. Optional - If point not on line find closest point on line

        # Search from start to end point to find pixel with best patch correspondence
        # Optimization - step through by some step length?

        pass