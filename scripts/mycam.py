from lib.options import BaseOptions
from scripts.mycam_mask import img2masks
# from scripts.img2strand import img2strand
# from scripts.img2depth import img2depth
import cv2
import numpy as np

if __name__ == "__main__":
    opt = BaseOptions().parse()
    # img2strand(opt)
    # img2depth(opt)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # 현재 프레임 읽기
        ret, frame = cap.read()
        # 프레임 읽기에 성공했는지 확인
        if ret:
            mask = img2masks(opt, frame)
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8) * 255
            print("mask:", mask.shape)
            mask = mask.reshape(512,512,-1)
            if len(mask.shape) == 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # 프레임을 화면에   표시
            cv2.imshow('Frame', mask)

            # 'q' 키를 누르면 루프를 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
