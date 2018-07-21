import cv2
from line import Line
import os
import glob
import numpy as np
class LaneDetectionPipeline(object):
    """
        Lane detection pipeline.
    """

    def __init__(self, debug=False):
        self._mtx = None # undistort matrice
        self._dist = None
        self._rvect = None
        self._tvect = None
        self._calibrated = False
        self._current_left_lane = None
        self._current_right_lane = None
        self._bad_frame_count = 0
        self._debug = debug

    def prepare_pipeline(self, chessboard_file_location, shape):
        self._current_left_lane = Line()
        self._current_right_lane = Line()
        assert(os.path.exists(chessboard_file_location))
        assert(shape is not None)
        chessboard_files = glob.glob(chessboard_file_location+"/*")
        print(chessboard_files)
        self._calibrate_camera_images(chessboard_files, shape)

    def _calibrate_camera_images(self, chessboard_files, shape):
        """
            Calibrate camera and assign value to
            @chessboard_files: checker board image files.
            @shape: image shape
        """
        # calibrate camera here
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane
        for chessboard_file in chessboard_files:
            img = cv2.imread(chessboard_file)
            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # find corners
            ret, corners = cv2.findChessboardCorners(gray, shape, None)
            # draw corners
            if ret:
                if self._debug:
                    cv2.drawChessboardCorners(img, shape, corners, ret)
                    plt.figure()
                    plt.imshow(img)
                # generate image coordinates using mgrid
                objp = np.zeros((shape[0]*shape[1],3),np.float32)
                # fix x,y coordinate
                objp[:,:2] = np.mgrid[0:shape[0],0:shape[1]].T.reshape(-1,2)
                objpoints.append(objp)
                imgpoints.append(corners)
        # now calibrate camera
        ret, self._mtx, self._dist, self._rvect, self._tvect = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
        assert ret
        assert self._mtx is not None
        assert self._dist is not None
        assert self._rvect is not None
        assert self._tvect is not None
        self._calibrated = True

    def _abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # Apply threshold
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if orient == 'x':
            grad_img = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        else:
            grad_img = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        scaled_img = np.uint8(grad_img / np.max(grad_img) * 255.0)
        final_img = np.zeros_like(scaled_img)
        final_img[(scaled_img > thresh[0]) & (scaled_img < thresh[1])] = 1
        return final_img

    def _mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply threshold
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binaryx_img = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        binaryy_img = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        binary_img = np.sqrt(binaryx_img**2 + binaryy_img**2)
        scaled_img = np.uint8(binary_img / np.max(binary_img) * 255.0)
        final_img = np.zeros_like(scaled_img)
        final_img[(scaled_img > mag_thresh[0]) & (scaled_img < mag_thresh[1])] = 1
        return final_img

    def _dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        # Apply threshold
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binaryx_img = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        binaryy_img = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        binary_img = np.arctan2(binaryy_img, binaryx_img)
        final_img = np.zeros_like(binary_img)
        final_img[(binary_img > thresh[0]) & (binary_img < thresh[1])] = 1
        return final_img

    def _s_channel_thresholding(self, img, threshold=(95,255)):
        """
            Using HLS color spaces to threshold the image
        """
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_chan = hsv_img[:,:,2]
        print(s_chan.shape)
        binary_img = np.zeros_like(s_chan)
        binary_img [(s_chan > threshold[0]) & (s_chan < threshold[1])] = 1
        print (s_chan > threshold[0])
        return binary_img

    def _threshold_image(self, img):
        """
            Apply all threshold techniques to filter out unneccessary noise.
        """
        print("Apply thresholding to image")
        abs_image = self._abs_sobel_thresh(img, thresh=(90,255))
        mag_grad_image = self._mag_thresh(img, mag_thresh=(110,255))
        dir_image = self._dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
        s_image = self._s_channel_thresholding(img, threshold=(95,255))
        shape = img.shape[:2]
        combine_mask_img = np.zeros(shape)
        combine_mask_img[(s_image > 0) | ((mag_grad_image >0) & (dir_image > 0)) | (abs_image > 0)] = 1
        return combine_mask_img


    def _warp_image_to_bird_eye_view(self, undistort_image):
        """
            Warp image to bird eye view so we can use sliding windows to search
            for the lanes.
        """
        print ("Warp image shape ", undistort_image.shape)
        width, height = undistort_image.shape[1], undistort_image.shape[0]
        src = np.float32([
            [undistort_image.shape[1]/2 - 120, undistort_image.shape[0]/1.8 + 50], # tl
            [undistort_image.shape[1]/2 + 120, undistort_image.shape[0]/1.8 + 50], # tr
            [undistort_image.shape[1], undistort_image.shape[0] - 20], # br
            [0, undistort_image.shape[0] - 20] # bl
        ])
        # reassign to variable names to keep code cleaner
        tl,tr,br,bl = src
        width = int(np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2))
        height = int(max(np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2), np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2)))
        # projected points
        dst = np.float32([
            [0,0], # tl
            [width - 1, 0], # tr
            [width - 1, height - 1], # br
            [0, height -1] # bl
        ], dtype='float32')

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(undistort_image, M, (width, height), flags=cv2.INTER_LINEAR)
        return warped, M, Minv, (tl,tr,br,bl)

    def _find_lanes_using_sliding_windows(self, warped_image, margin=100, minpix=50, debug=False):
        """

            margin: where to search for pixel
            minpix: number of nonzero pix that trigger re-center
        """
        # use histogram to find peaks
        # sum pixels along colums
        histogram = np.sum(warped_image, axis=0)
        # empty output image for visualizing
        output_img = np.dstack((warped_image, warped_image, warped_image)) * 255
        # find base of starting x positions for left and right lane
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # find number of windows
        nwindows = 10
        window_height = np.int(warped_image.shape[0] // nwindows)
        # get nonzero tuple
        print(warped_image.shape)
        nonzero_y, nonzero_x = warped_image.nonzero()
        l_x = leftx_base
        r_x = rightx_base
        left_lane_indices = []
        right_lane_indices = []

        for iw in range(nwindows):
            w_y_low = warped_image.shape[0] - (iw + 1)*window_height
            w_y_high = w_y_low + window_height
            # left lane
            w_x_left_low = l_x - margin
            w_x_left_high = l_x + margin
            # right lane
            w_x_right_low = r_x - margin
            w_x_right_high = r_x + margin
            # draw image for visualizing

    #         print("Draw rect x,y:" ,w_x_left_low, w_y_low, w_x_left_high, w_y_high)
            if debug:
                cv2.rectangle(output_img,(w_x_left_low, w_y_low),(w_x_left_high, w_y_high),(0,255,0), 4)
                cv2.rectangle(output_img,(w_x_right_low, w_y_low),(w_x_right_high, w_y_high),(0,255,0), 4)
    #         plt.figure()
    #         plt.imshow(output_img)
            # find good left land indices
            good_left_lane_indices = (nonzero_y >= w_y_low) & (nonzero_y < w_y_high) \
                        & (nonzero_x >= w_x_left_low) & (nonzero_x < w_x_left_high)

            good_right_lane_indices = (nonzero_y >= w_y_low) & (nonzero_y < w_y_high) \
                        & (nonzero_x >= w_x_right_low) & (nonzero_x < w_x_right_high)

            # remove all False value
            good_left_lane_indices = good_left_lane_indices.nonzero()[0]
            good_right_lane_indices = good_right_lane_indices.nonzero()[0]

            # append to full list
            left_lane_indices.append(good_left_lane_indices)
            right_lane_indices.append(good_right_lane_indices)

            # change the center if number of pix is at threshold
            #print ("len ",len(good_left_lane_indices), len(good_right_lane_indices))
            if len(good_left_lane_indices) > minpix:
                l_x = np.int(np.mean(nonzero_x[good_left_lane_indices]))
            if len(good_right_lane_indices) > minpix:
                r_x = np.int(np.mean(nonzero_x[good_right_lane_indices]))


        # now we have an array list of all good indices
        all_good_left_lanes = np.concatenate(left_lane_indices)
        all_good_right_lanes = np.concatenate(right_lane_indices)

        # extract x,y coordinates for each lane so we can fit a polynomial through them
        leftx = nonzero_x[all_good_left_lanes]
        lefty = nonzero_y[all_good_left_lanes]
        rightx = nonzero_x[all_good_right_lanes]
        righty = nonzero_y[all_good_right_lanes]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, warped_image.shape[0] - 1, warped_image.shape[0] )
        ploty = np.linspace(0, warped_image.shape[0] - 1, warped_image.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # FOR PLOTTING ONLY
        if debug:


            output_img[nonzero_y[all_good_left_lanes], nonzero_x[all_good_left_lanes]] = [255, 0, 0]
            output_img[nonzero_y[all_good_right_lanes], nonzero_x[all_good_right_lanes]] = [0, 0, 255]
            plt.figure(figsize=(10,10))
            # hacky way to write cv2 image to file
            cv2.imwrite('test.jpg', output_img)
            test = plt.imread("test.jpg")
            plt.imshow(test)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='green')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)

        return left_fit, right_fit, leftx, lefty, rightx, righty, ploty, left_fitx, right_fitx


    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = left_fit[0]*ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1] * ploty + right_fit[2]

        return left_fit, right_fit, leftx, lefty, rightx, righty, ploty, left_fitx ,right_fitx

    def search_around_poly(self, binary_warped, left_fit, right_fit, margin=100, threshold=0.8):
        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # search around the current left fit and right fit
        left_lane_x_margin_left = (left_fit[0] * (nonzero_y**2) + left_fit[1] * nonzero_y + left_fit[2] - margin)
        left_lane_x_margin_right = (left_fit[0] * (nonzero_y**2) + left_fit[1] * nonzero_y + left_fit[2] + margin)
        left_lane_ids = (nonzero_x >= left_lane_x_margin_left) & (nonzero_x < left_lane_x_margin_right)

        right_lane_x_margin_left = (right_fit[0] * (nonzero_y**2) + right_fit[1] * nonzero_y + right_fit[2] - margin)
        right_lane_x_margin_right = (right_fit[0] * (nonzero_y**2) + right_fit[1] * nonzero_y + right_fit[2] + margin)
        right_lane_ids = (nonzero_x >= right_lane_x_margin_left) & (nonzero_x < right_lane_x_margin_right)

        # TODO:
        # if number of pixels ids is smaller than some threshold
        # return none so we will use sliding windows to rediscover the windows
        if (len(left_lane_ids) + len(right_lane_ids)) / np.float(len(nonzero_x)) < threshold:
            print ("Bad lane detected")
            return None, None, None, None, None

        leftx = nonzero_x[left_lane_ids]
        lefty = nonzero_y[left_lane_ids]
        rightx = nonzero_x[right_lane_ids]
        righty = nonzero_y[right_lane_ids]

        # now we have the values
        return self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    def _draw_lanes_on_image(self, warped, undistort_img, ploty, left_fitx, right_fitx, Minv):
        # Create an image to draw the lines on
        img_shape = undistort_img.shape
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img_shape[1], img_shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undistort_img, 1, newwarp, 0.3, 0)
        return result

    def run(self, video_file):
        # run pipeline here
        assert self._calibrated
        assert os.path.exists(video_file)
        print ("Start Video Processing")
        # open the video and feed the frame here
        cap = cv2.VideoCapture(video_file)
        while cap.isOpened():
            ret, orig_frame = cap.read()
            # if frame is valid then run it through the pipe line
            if ret == True:
                print("Good frame", orig_frame.shape)
                frame = self._threshold_image(orig_frame)
                #obtain bird-eye view of the lane
                warped, M, Minv, _ = self._warp_image_to_bird_eye_view(frame)
                # if one of the lane is not detected use sliding windows to search it
                if (not self._current_left_lane.detected) or (not self._current_right_lane.detected):
                    left_fit, right_fit, leftx, lefty, rightx, righty, ploty, left_fitx ,right_fitx = self._find_lanes_using_sliding_windows(warped)
                    self._current_left_lane.update(True, allx=leftx, ally=lefty, fit_coeff=left_fit)
                    self._current_right_lane.update(True, allx=rightx, ally=righty, fit_coeff=right_fit)
                else: # just search around the current line here
                    print("search around")
                    left_fit = self._current_left_lane.best_fit
                    right_fit = self._current_right_lane.best_fit
                    left_fit, right_fit, leftx, lefty, rightx, righty, ploty, left_fitx ,right_fitx = \
                                        self.search_around_poly(warped, left_fit, right_fit)
                    self._current_left_lane.update(True, allx=leftx, ally=lefty, fit_coeff=left_fit)
                    self._current_right_lane.update(True, allx=rightx, ally=righty, fit_coeff=right_fit)

                result_frame = self._draw_lanes_on_image(warped, orig_frame, ploty, left_fitx, right_fitx, Minv)

                cv2.imshow('Frame', result_frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                         break

        cap.release()
        print("Quit Video Processing")
