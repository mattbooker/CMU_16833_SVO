import numpy as np 
from scipy.interpolate import RectBivariateSpline
from config import Config

class ImageAlignment:
    def __init__(self):
        self.Transform = None
        self.threshold = 0.025
        self.maxIter = 100
        self.windowSize = 4

    def LucasKanadeAffine(self, It, It1, rect, M, thresh=.025, maxIt=100):
        '''
        Lucas-Kanade Forward Additive Alignment with Affine MAtrix
        
        Inputs: 
            It: template image
            It1: Current image
            rect: Current position of the object
            (top left, bottom right coordinates, x1, y1, x2, y2)
            thresh: Stop condition when dp is too small
            maxIt: Maximum number of iterations to run
            
        Outputs:
            M: Affine mtarix (2x3)
        '''
        # Set thresholds (you probably want to play around with the values)
        M = np.zeros((2,3))
        threshold = thresh
        maxIters = maxIt
        i = 0
        x1, y1, x2, y2 = rect
        
        # ----- TODO -----
        # YOUR CODE HERE
        M = M
        x_spline = np.arange(0, It.shape[1])
        y_spline = np.arange(0, It.shape[0])

        #get gradients along x and y (to optimise run time)
        img_y, img_x = np.gradient(It1)

        #get complete images in the form of splines (to avoid indexing errors)
        interp_it = RectBivariateSpline(y_spline, x_spline, It)
        interp_it1 = RectBivariateSpline(y_spline, x_spline, It1)
        interp_it1_y = RectBivariateSpline(y_spline, x_spline, img_y)
        interp_it1_x = RectBivariateSpline(y_spline, x_spline, img_x)

        #define matrix for original template without movement 
        x_img_range = np.arange(0, It.shape[1])
        y_img_range = np.arange(0, It.shape[0])
        x_img, y_img = np.meshgrid(x_img_range, y_img_range)
        #t_img = interp_it.ev(y_img, x_img)
        x_img = x_img.flatten()
        y_img = y_img.flatten()
        i = 0
        while (i <= maxIters):       
            #define window around rect
            x_temp_range = np.arange(x1 , x2 + 0.5)
            y_temp_range = np.arange(y1 , y2 + 0.5)
            x_temp, y_temp = np.meshgrid(x_temp_range, y_temp_range)

            #take affine transform of x and y 
            one = np.ones((x_temp.shape))
            pts = np.vstack((x_temp.flatten(), y_temp.flatten(), one.flatten())).reshape(3,-1)
            pts_new = M@pts

            xs = pts_new[0].reshape(x_temp.shape)
            ys = pts_new[1].reshape(x_temp.shape)

            error_img = interp_it.ev(y_temp, x_temp) - interp_it1.ev(ys, xs)   

            dx = interp_it1.ev(ys, xs, dy = 1).reshape((-1, 1))
            dy = interp_it1.ev(ys, xs, dx = 1).reshape((-1, 1))
            
            xs = xs.flatten().reshape((-1, 1))
            ys = ys.flatten().reshape((-1, 1))

            A = np.hstack((xs*dx, ys*dx, dx, xs*dy, ys*dy, dy))
            B = error_img.flatten().reshape(-1,1)

            del_m = np.linalg.lstsq(A, B, rcond = None)[0].reshape(2,3)
            M = M + del_m
            if np.linalg.norm(del_m) < thresh :
                print(i)
                break
            i += 1
        return M
    
    def findAlignment(self, previous_frame, current_frame, threshold=Config.ImageAlignment.THRESHOLD, maxIters=Config.ImageAlignment.MAX_ITER):

        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        for i in range(previous_frame.np_keypoints_.shape[0]):
            
            x0, y0 = previous_frame.np_keypoints_[i].astype(int)

            rect = [x0 - Config.ImageAlignment.WINDOW_SIZE
                    //2, y0 - Config.ImageAlignment.WINDOW_SIZE
                    //2, x0 + Config.ImageAlignment.WINDOW_SIZE
                    //2, y0 + Config.ImageAlignment.WINDOW_SIZE
                    //2]

            M = self.LucasKanadeAffine(current_frame.image_, previous_frame.image_, rect, M, threshold, maxIters)
        
        print(f'M : {M.shape}')
        print(f'previous : {previous_frame.np_keypoints_.shape}')
    
        homogenized_keypts = np.vstack([previous_frame.np_keypoints_[:,0],previous_frame.np_keypoints_[:,1],np.ones(previous_frame.np_keypoints_.shape[0])]).T.copy() 
        warped_keypoints = M@homogenized_keypts.T
        warped_keypoints /= warped_keypoints[-1]

        print(f'homo : {homogenized_keypts.shape}')
        print(f'warped keypoints norm : {warped_keypoints}')
        
        warped_keypoints = warped_keypoints[:,:2]
        print(f'warped keypoints top 2 : {warped_keypoints.shape}')

        current_frame.np_keypoints_ = warped_keypoints

        return M

    def getRotTrans(self, H):
        # returns rotation and translation matrix
        '''
        K is the camera calibration matrix
        T is translation
        R is rotation
        '''
        H = H.T
        h1 = H[0]
        h2 = H[1]
        h3 = H[2]
        K_inv = np.linalg.inv(K)
        L = 1 / np.linalg.norm(np.dot(K_inv, h1))
        r1 = L * np.dot(K_inv, h1)
        r2 = L * np.dot(K_inv, h2)
        r3 = np.cross(r1, r2)
        T = L * (K_inv @ h3.reshape(3, 1))
        R = np.array([[r1], [r2], [r3]])
        R = np.reshape(R, (3, 3))

        return R, T