import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (8,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

# cv.destroyAllWindows()



# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Calibration successful:", ret)
print("\nCamera matrix:\n", mtx)
print("\nDistortion coefficients:\n", dist)
print("\nRotation:\n", rvecs[0])
print("\nTranslation:\n", tvecs[0])

# Save calibration results
np.savez("camera_calibration_9x7.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

img = cv.imread(images[2])
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

def overlay_grid(image, step=50, color=(0, 0, 255), thickness=1):
    """Overlays a red grid onto an image."""
    img_with_grid = image.copy()
    h, w = image.shape[:2]

    # Draw vertical lines (Red color is BGR: (0, 0, 255))
    for x in range(0, w, step):
        cv.line(img_with_grid, (x, 0), (x, h), color, thickness)

    # Draw horizontal lines
    for y in range(0, h, step):
        cv.line(img_with_grid, (0, y), (w, y), color, thickness)

    return img_with_grid

# Define grid parameters
GRID_STEP = 50  # Distance between lines in pixels
GRID_COLOR = (0, 0, 255) # Red in BGR
GRID_THICKNESS = 1

# Apply grid to both images
original_with_grid = overlay_grid(img, GRID_STEP, GRID_COLOR, GRID_THICKNESS)
undistorted_with_grid = overlay_grid(dst, GRID_STEP, GRID_COLOR, GRID_THICKNESS)

# Display images with grid
cv.imshow('Original (Distorted) with Grid', original_with_grid)
cv.imshow('Undistorted with Grid', undistorted_with_grid)
cv.waitKey(0)
cv.destroyAllWindows()