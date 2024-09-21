import cv2
import os

# 사진 저장 경로 설정
save_path = "captured_images"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 사진 찍기
num_photos = 20
photo_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # 화면에 카메라 영상 출력
    cv2.imshow('Camera', frame)
    
    # 's' 키를 눌러 사진 촬영
    if cv2.waitKey(1) & 0xFF == ord('s'):
        photo_count += 1
        photo_filename = os.path.join(save_path, f'photo_{photo_count:03d}.jpg')
        cv2.imwrite(photo_filename, frame)
        print(f'{photo_filename} 저장 완료.')
        
        # 20장 다 찍으면 종료
        if photo_count >= num_photos:
            break

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 해제 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
