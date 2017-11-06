import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle


def chessboard_corners( imdir , checkcols , checkrows, outdir='examples/calibration_output/' ):
    '''
    Find the chessboard corner correspondences 
    '''

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((checkcols*checkrows,3), np.float32)
    objp[:,:2] = np.mgrid[0:checkcols, 0:checkrows].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images. Each element in this list is
    # the image name has (9,6) number of corners
    images = [imdir+'calibration'+str(i)+'.jpg' for i in range(2,21)]

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (checkcols,checkrows), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            print("Image: {} calibration successful".format(fname))

            # Draw and display the corners        
            cv2.drawChessboardCorners(img, (checkcols,checkrows), corners, ret)
            write_name = outdir + '/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)

        else:
            print("Calibration Unsuccessful for {}".format(fname))
        
    return objpoints, imgpoints

def find_reclanes(img, gradthresh = (20,100), colorthresh = (170,255)):
    ''' 
    find_reclanes: find rectified lanes
    
    Input: img (which will be copied)
    '''

    # Copy the image so you're not operating on the original
    img = np.copy(img)
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min, thresh_max = gradthresh
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min, s_thresh_max = colorthresh
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return color_binary, combined_binary

def draw_box(im, pairs):
    ''' 
    Draw a box given four points and an image
    '''
    im = np.copy(im)
    cv2.line(im, pairs[0], pairs[1], color=[255,0,0], thickness=4)
    cv2.line(im, pairs[1], pairs[2], color=[255,0,0], thickness=4)
    cv2.line(im, pairs[2], pairs[3], color=[255,0,0], thickness=4)
    cv2.line(im, pairs[3], pairs[0], color=[255,0,0], thickness=4)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
           max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_poly_from_hist(binary_warped, margin=100, minpix=50):
    '''
    Inputs:
        binary_warped - warped binary image of thresholded pixels
        margin - the +/- width of the window being swept
        minpix - the minimum number of pixels to recenter the window
        
    Outputs:
        left_fit - polynomial for the left line
        right_fit - polynomial for the right line
        (leftx, lefty) - all sets of points used for the left line
        (rightx, righty) - all sets of points used for the right line
        out_img - if visualizing the output image
    '''

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                      (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                      (0,255,0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
     
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    left_all = (nonzerox[left_lane_inds], nonzeroy[left_lane_inds])
    right_all = ( nonzerox[right_lane_inds], nonzeroy[right_lane_inds] )

    return left_fit, right_fit, left_all, right_all, out_img

def find_poly_from_poly(binary_warped, left_fit, right_fit):
    
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                                   left_fit[2] - margin)) & 
                      (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                                   left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                                    right_fit[2] - margin)) & 
                       (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                                    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    window_img = np.zeros_like(out_img)
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
    return left_fit, right_fit, (leftx, lefty), (rightx, righty), out_img

def detect_lanes(imname, mtx, dist, M, Minv):
    '''
    Full pipeline to detect lanes
    Draw lanes on top of image given distortion and image warping matrix
    Arguments:
      imname - image name (text)
      mtx - camera calibration coefficient
      dist - distortion coefficient
      M - warping matrix
      Minv = inverse of warping matrix
      
    Returns:
      image with lanes overlayed on top
    '''
    if type(imname) == str:
        img = cv2.imread(imname)
    else:
        img = imname
    img_size = (img.shape[1],img.shape[0])
    calimg = cv2.undistort(img, mtx, dist, None, mtx)
    warped = cv2.warpPerspective(calimg, M, img_size, flags=cv2.INTER_LINEAR)
    color_binary, combined_binary = find_reclanes(warped)
    left_fit, right_fit, _, _, _ = find_poly_from_hist(combined_binary)
    lanedrawn = draw_lane(left_fit, right_fit, img_size=img_size)
    newwarp = cv2.warpPerspective(lanedrawn, Minv, (img.shape[1], img.shape[0]))     
    return cv2.addWeighted(calimg, 1, newwarp, 0.3, 0)

def draw_lane(left_fit, right_fit, img_size=(1280, 720)):
    '''
    Draw lane lines and fill in the lane with green
    '''
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_size[0]-1, img_size[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    color_warp = np.zeros((img_size[1],img_size[0], 3)).astype(np.uint8)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
        
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            
    return color_warp


def convert_curvature(left_fit, right_fit, ploty, maxy=720, meters=True):
    
    '''
    Arguments:
      leftx, rightx = left and right x points
      ploty = y points, corresponding to both leftx and rightx
      meters = Do this in meters or pixel space, defaults to meters
    '''
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ploty = np.linspace(0, maxy-1, maxy )
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    if meters:
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
    else:
        ym_per_pix = 1.0
        xm_per_pix = 1.0

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

def car_center_offset(left_fit, right_fit, maxy=720, meters=True):
    '''
    Determine the center offset for the vehicle. This assumes that the center of the car is the center of the image.:
    Arguments:
      - left_fit: line fit to left lane as  polynomial
      - right_fit: line fit to right lane as polynomial
    '''
    
    left_pos = left_fit[0]*maxy**2 + left_fit[1]*maxy + left_fit[2]
    right_pos = right_fit[0]*maxy**2 + right_fit[1]*maxy + right_fit[2]
    
    offset = 1260/2 - np.mean([left_pos, right_pos])
    
    if meters:
        xm_per_pix=3.7/700
    else:
        xm_per_pix=1.0

    return offset*xm_per_pix


