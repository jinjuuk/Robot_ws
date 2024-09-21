import cv2
import numpy as np

CHECKERBOARD = (6, 5)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(0)

calibrated = False
frame_count = 0
required_frames = 20
undistortion_message_shown = False  # 이 변수를 루프 밖으로 이동

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if not calibrated:
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            print(f"Chessboard detected in frame {frame_count + 1}")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            frame = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
            frame_count += 1
        else:
            print("No chessboard detected in this frame")
            
        if frame_count >= required_frames:
            print("Calculating camera matrix and distortion coefficients...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            calibrated = True
            print("Calibration completed. Camera matrix and distortion coefficients calculated.")
            
            print("\nInternal Parameters (Camera Matrix):")
            print(mtx)
            
            print("\nDistortion Coefficients:")
            print(dist)
            
            print("\nExternal Parameters:")
            print("Rotation Vector:")
            print(rvecs[-1])
            print("Translation Vector:")
            print(tvecs[-1])
            
    else:
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow('Undistorted', dst)
        
        # 언디스토션 메시지를 한 번만 출력
        if not undistortion_message_shown:
            print("Undistortion applied to the frames")
            undistortion_message_shown = True
    
    cv2.imshow('Calibration', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








