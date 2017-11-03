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





