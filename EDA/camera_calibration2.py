import cv2
import numpy as np
import glob
import os

# 체스보드 크기 설정
CHECKERBOARD = (6, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 체스보드의 3D 포인트 준비
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 3D 포인트와 2D 포인트 저장 리스트
objpoints = []
imgpoints = []

# 체스보드 이미지 파일 경로 설정 (여기서는 예시로 'chessboard_images' 폴더 사용)
images = glob.glob('/home/jinjuuk/dev_ws/EDA/*.jpg')

# 각 이미지에 대해 체스보드 코너 찾기
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # 코너 그리기 및 이미지 표시 (선택 사항)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 카메라 보정
if len(objpoints) > 0 and len(imgpoints) > 0:
    print("Calculating camera matrix and distortion coefficients...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Calibration completed.")

    print("\nInternal Parameters (Camera Matrix):")
    print(mtx)

    print("\nDistortion Coefficients:")
    print(dist)

    # 보정된 이미지 저장 폴더 설정
    save_path = "calibrated_images"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 각 이미지에 대해 왜곡 제거 및 저장
    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # 보정된 이미지 저장
        basename = os.path.basename(fname)
        save_filename = os.path.join(save_path, basename)
        cv2.imwrite(save_filename, dst)
        print(f"{save_filename} 저장 완료.")
else:
    print("체스보드 이미지를 충분히 인식하지 못했습니다. 다른 이미지를 사용해보세요.")
