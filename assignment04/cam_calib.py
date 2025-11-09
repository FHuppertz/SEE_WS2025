import numpy as np
import cv2 as cv
import glob
import os

CHECKERBOARD = (8, 6)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints, imgpoints = [], []

images = sorted(glob.glob("*.jpg"))
if not images:
    raise FileNotFoundError("No calibration images found in current directory!")

print(f"Found {len(images)} images.")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow("Corners", img)
        cv.waitKey(100)
cv.destroyAllWindows()

print(f"\nValid calibration images: {len(objpoints)}")

# ================================================================
# Regular Calibration
# ================================================================
gray_shape = gray.shape[::-1]
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray_shape, None, None,
    flags=cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL
)
print("\nRegular model RMS error:", ret)

# Compute reprojection error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    total_error += cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
mean_error = total_error / len(objpoints)
print(f"Mean reprojection error: {mean_error:.5f}")

# ================================================================
# Switch to fisheye calibration if distortion still bad
# ================================================================
use_fisheye = ret > 1.0 or mean_error > 0.6  # empirical thresholds
print(f"\nUsing Fisheye model? {'Yes' if use_fisheye else 'No'}")

if use_fisheye:
    # Fisheye calibration
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    objp_fish = [objp] * len(objpoints)
    rms, _, _, _, _ = cv.fisheye.calibrate(
        objp_fish,
        imgpoints,
        gray.shape[::-1],
        K, D,
        None, None,
        cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
        criteria
    )
    print("\nFisheye RMS:", rms)
    print("K (camera matrix):\n", K)
    print("D (distortion coeffs):\n", D.ravel())
    np.savez("fisheye_calibration.npz", K=K, D=D)

    # Undistortion
    img = cv.imread(images[0])
    h, w = img.shape[:2]
    map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv.CV_16SC2)
    undistorted = cv.remap(img, map1, map2, interpolation=cv.INTER_LANCZOS4)
else:
    # Standard undistortion
    img = cv.imread(images[0])
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1.0, (w, h))
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv.CV_32FC1)
    undistorted = cv.remap(img, mapx, mapy, cv.INTER_LANCZOS4)

cv.imshow("Original", img)
cv.imshow("Undistorted", undistorted)
cv.waitKey(0)
cv.destroyAllWindows()

print("\n✅ Undistortion complete. Inspect t")