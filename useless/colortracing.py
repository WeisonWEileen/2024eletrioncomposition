import cv2
import numpy as np
# import serial
import struct

cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)

CAP_WIDTH = 320
CAP_HEIGHT = 240
# ROI
CAP_CENTER = (160, 135)
CAP_LENGTH = 180

color = {'red': (0, 0, 255),
         'green': (0, 255, 0),
         'blue': (255, 0, 0),
         'yellow': (0, 255, 255), }

# 4 members: red in white, red in black, green in white, green in black
color_range = {'color_red_in_white': {'Lower': np.array([0, 40, 240]), 'Upper': np.array([40, 255, 255])},
               'color_red_in_black': {'Lower': np.array([0, 40, 100]), 'Upper': np.array([40, 255, 250])},
               'color_green_in_white': {'Lower': np.array([60, 40, 100]), 'Upper': np.array([100, 255, 254])},
               'color_green_in_black': {'Lower': np.array([60, 40, 100]), 'Upper': np.array([100, 255, 250])}}



class IMG:
    def __init__(self, _name, _color, _img, _range, _erode_kernel):
        self.name = _name
        self.color = _color
        self.image = cv2.inRange(_img, _range['Lower'], _range['Upper'])
        self.image = cv2.erode(self.image, _erode_kernel)

        self.maxcnt = None
        self.center = None
        self.radius = None


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)  # 宽度
CAP_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)  # 宽度
CAP_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 宽度
# cap.set(cv2.CAP_PROP_FPS, 30)	            # 帧率 帧/秒
cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)  # 亮度
cap.set(cv2.CAP_PROP_CONTRAST, 50)  # 对比度
cap.set(cv2.CAP_PROP_SATURATION, 100)  # 饱和度
cap.set(cv2.CAP_PROP_HUE, 20)  # 色调 50
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 曝光

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if frame is not None:
            # 获取图像
            frame = frame[int(CAP_CENTER[1] - (CAP_LENGTH / 2)):int(CAP_CENTER[1] + (CAP_LENGTH / 2)),
                    int(CAP_CENTER[0] - (CAP_LENGTH / 2)):int(CAP_CENTER[0] + (CAP_LENGTH / 2)), :]
            cv2.imshow('camera', frame)

            # 预处理
            gs_frame = cv2.GaussianBlur(frame, (7, 7), 0)  # 高斯模糊
            # cv2.imshow('gs_frame', gs_frame)
            hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像

            # 二值化
            kernel = np.ones((3, 3), np.uint8)
            img_list = [IMG('red_in_white', color['yellow'], hsv, color_range['color_red_in_white'], kernel),
                        IMG('red_in_black', color['yellow'], hsv, color_range['color_red_in_black'], kernel),
                        IMG('green_in_white', color['blue'], hsv, color_range['color_green_in_white'], kernel),
                        IMG('green_in_black', color['blue'], hsv, color_range['color_green_in_black'], kernel)]

            cv2.imshow('red_in_white_pcs', img_list[0].image)
            cv2.imshow('red_in_black_pcs', img_list[1].image)
            cv2.imshow('green_in_white_pcs', img_list[2].image)
            cv2.imshow('green_in_black_pcs', img_list[3].image)

            for i, img in enumerate(img_list):
                if i%2 == 1:
                    if img_list[i-1].maxcnt is not None:
                        continue

                cnts, hierarchy = cv2.findContours(img.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                max_cnt = None
                max_cnt_area = 0
                for j in range(0, len(cnts)):
                    if hierarchy[0][j][3] != -1:
                        cnt_area = cv2.contourArea(cnts[j])
                        if cnt_area > max_cnt_area:
                            max_cnt = cnts[j]
                            max_cnt_area = cnt_area
                if max_cnt is None:
                    # print("No"+ img.name)
                    # if i%2 == 1:
                    for j in range(0, len(cnts)):
                        cnt_area = cv2.contourArea(cnts[j])
                        if cnt_area > max_cnt_area:
                            max_cnt = cnts[j]
                            max_cnt_area = cnt_area
                    if max_cnt is not None:
                        img.maxcnt = max_cnt
                        img.center, img.radius = cv2.minEnclosingCircle(img.maxcnt)
                        cv2.circle(frame, (int(img.center[0]), int(img.center[1])), int(img.radius), img.color, 2)
                        cv2.circle(frame, (int(img.center[0]), int(img.center[1])), 1, img.color, 2)

                else:
                    # print(img.name)
                    img.maxcnt = max_cnt
                    img.center, img.radius = cv2.minEnclosingCircle(img.maxcnt)
                    cv2.circle(frame, (int(img.center[0]), int(img.center[1])), int(img.radius), img.color, 2)
                    cv2.circle(frame, (int(img.center[0]), int(img.center[1])), 1, img.color, 2)


            # cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            #
            # maxarea = 0
            # maxcnt = None
            # for cnt in cnts:
            #     area = cv2.contourArea(cnt)
            #     if area > maxarea:
            #         maxcnt = cnt
            #         maxarea = area
            #
            # if maxcnt is not None:
            #     maxcnt = max(cnts, key=cv2.contourArea)
            #     cv2.drawContours(frame, [np.int0(maxcnt)], -1, (0, 255, 255), 2)
            #     (x, y), r = cv2.minEnclosingCircle(maxcnt)
            #     frame = cv2.circle(frame, (int(x), int(y)), int(r), (255, 255, 0), 2)
            #     print(x, y)
            #     #str = struct.pack("1B", int())
            #     x_h = int(x / 256)
            #     x_l = int(x % 256)
            #     y_h = int(y / 256)
            #     y_l = int(y % 256)
            #     str = struct.pack("14B", 0xFF, 1, 0, x_h, x_l, y_h, y_l, 0, 0, 0, 0,0,0x0D, 0x0A)
            #     print(str)
            #     com.write(str)

            cv2.imshow("processed frame", frame)
            print()

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cv2.imwrite("img.jpg", frame)
                break
            if key & 0xFF == ord('f'):
                cv2.imwrite("img.jpg", frame)
                break
        else:
            print("无画面")
    else:
        print("无法读取摄像头！")

cap.release()
cv2.destroyAllWindows()
