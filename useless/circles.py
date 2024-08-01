import cv2
import numpy as np
import math
# import serial
import struct



cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)

CAP_WIDTH = 1920
CAP_HEIGHT = 1080
# ROI
CAP_CENTER = (1080, 607)
CAP_LENGTH = 800

BOX_SIZE_RATIO = 0.1
EPSILON_RATIO = 0.05

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)  # 宽度
CAP_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)  # 宽度
CAP_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 宽度
# cap.set(cv2.CAP_PROP_FPS, 30)	            # 帧率 帧/秒
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)  # 亮度
# cap.set(cv2.CAP_PROP_CONTRAST, 50)  # 对比度
# cap.set(cv2.CAP_PROP_SATURATION, 100)  # 饱和度
# cap.set(cv2.CAP_PROP_HUE, 20)  # 色调 50
# cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 曝光
# Enable auto exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 turns ON auto exposure
print("SIZE:")
print((CAP_WIDTH, CAP_HEIGHT))

def findblackcircle(hsv, frame=None):
    if hsv is None:
        print("img_bin cannot be None")
        assert True
    # min_area = int(BOX_SIZE_RATIO * len(img_bin) * len(img_bin[0]))


    lower_black = np.array([0, 0, 0])  # 设定黑色的阈值下限
    upper_black = np.array([180, 255, 46])  # 设定黑色的阈值上限
    kernel = np.ones((3, 3), np.uint8)
    black_image = cv2.inRange(hsv, lower_black, upper_black)
    black_image = cv2.erode(black_image, kernel)
    cv2.imshow('black_image', black_image)
    black_circles = dict()
    circle_id = 0
    # cnts, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnts,hierachy = cv2.findContours(black_image, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None
    print(f"the number of counters detected is {len(cnts)}")
    for i, cnt in enumerate(cnts):
        area_cnt = cv2.contourArea(cnt)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        area_circle = 3.1415926 * radius * radius
        if abs(area_cnt - area_circle) < area_circle * 0.2 and radius>10:  # Adjust the threshold as needed
            print('Found a circle at ({}, {}), radius: {}'.format(x, y, radius))
            center = (int(x), int(y))
            radius = int(radius)
        # if radius < 10:
        #     continue
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            # cv2.putText(frame, f"{i}", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            x_start = max(0, int(x) - 5)
            y_start = max(0, int(y) - 5)
            x_end = min(hsv.shape[1], int(x) + 5)
            y_end = min(hsv.shape[0], int(y) + 5)
            region = hsv[y_start:y_end, x_start:x_end]#圆中间10*10的区域用来检测颜色
            mean_color = np.mean(region, axis=(0, 1))
            if np.all((lower_black <= mean_color) & (mean_color <= upper_black)):
                text = 'id:'+str(circle_id)+' color:black'
                cv2.putText(frame, text, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA, 0)  # 显示圆心位置
                circle_id += 1
                # black_circles[circle_id] = (x, y, r)
                black_circles[circle_id] = (center, radius)
    return black_circles

def findwhitecircle(hsv, frame=None):
    if hsv is None:
        print("img_bin cannot be None")
        assert True
    # min_area = int(BOX_SIZE_RATIO * len(img_bin) * len(img_bin[0]))

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换为HSV空间用于识别黑色
    lower_white = np.array([0, 0, 221])  # 设定黑色的阈值下限
    upper_white = np.array([180, 30, 255])  # 设定黑色的阈值上限
    kernel = np.ones((3, 3), np.uint8)
    white_image = cv2.inRange(hsv, lower_white, upper_white)
    white_image = cv2.erode(white_image, kernel)
    cv2.imshow('white_image', white_image)
    white_circles = dict()
    circle_id = 0
    # cnts, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnts,hierachy = cv2.findContours(white_image, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None
    print(f"the number of counters detected is {len(cnts)}")
    for i, cnt in enumerate(cnts):
        area_cnt = cv2.contourArea(cnt)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        area_circle = 3.1415926 * radius * radius
        if abs(area_cnt - area_circle) < area_circle * 0.2 and radius>10:  # Adjust the threshold as needed
            print('Found a circle at ({}, {}), radius: {}'.format(x, y, radius))
            center = (int(x), int(y))
            radius = int(radius)
        # if radius < 10:
        #     continue
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            # cv2.putText(frame, f"{i}", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            x_start = max(0, int(x) - 5)
            y_start = max(0, int(y) - 5)
            x_end = min(hsv.shape[1], int(x) + 5)
            y_end = min(hsv.shape[0], int(y) + 5)
            region = hsv[y_start:y_end, x_start:x_end]#圆中间10*10的区域用来检测颜色
            mean_color = np.mean(region, axis=(0, 1))
            if np.all((lower_white <= mean_color) & (mean_color <= upper_white)):
                text = 'id:'+str(circle_id)+' color:white'
                cv2.putText(frame, text, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA, 0)  # 显示圆心位置
                circle_id += 1
                # black_circles[circle_id] = (x, y, r)
                white_circles[circle_id] = (center, radius)
    return white_circles

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if frame is not None:
            # 获取图像
            frame = frame[int(CAP_CENTER[1] - (CAP_LENGTH / 2)):int(CAP_CENTER[1] + (CAP_LENGTH / 2)),
                    int(CAP_CENTER[0] - (CAP_LENGTH / 2)):int(CAP_CENTER[0] + (CAP_LENGTH / 2)), :]

            # 预处理
            gs_frame = cv2.GaussianBlur(frame, (7, 7), 0)  # 高斯模糊
            # cv2.imshow('gs_frame', gs_frame)
            frame_gray = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2GRAY)  # 转化成GRAY图像
            _, frame_bin = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            #hsv图像
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换为HSV空间


            # black_circles = findblackcircle(hsv, frame)
            # white_circles = findwhitecircle(hsv, frame)

            # cv2.imshow('frame_bin', frame_bin)
            cv2.imshow('camera', frame)

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