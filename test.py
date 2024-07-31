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
print("SIZE:")
print((CAP_WIDTH, CAP_HEIGHT))

def findcircle(img_bin, frame=None):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换为HSV空间

    kernel = np.ones((5, 5), np.uint8)  # 定义卷积核

    font = cv2.FONT_HERSHEY_SIMPLEX
    lower_black = np.array([0, 0, 0])  # 设定黑色的阈值下限
    upper_black = np.array([180, 255, 46])  # 设定黑色的阈值上限
    #  消除噪声
    mask = cv2.inRange(hsv, lower_black, upper_black)  # 设定掩膜取值范围
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 形态学开运算
    # bila = cv2.bilateralFilter(mask, 10, 200, 200)  # 双边滤波消除噪声
    edges = cv2.Canny(opening, 50, 100)  # 边缘识别
    # 识别圆形

    black_circles = dict()
    circle_id = 0
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, 1, 300, param1=100, param2=18, minRadius=50, maxRadius=500)
    if circles is not None:  # 如果识别出圆
        for circle in circles:
            #  获取圆的坐标与半径
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])
            cv2.circle(frame, (x, y), r, (0, 0, 255), 3)  # 标记圆
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)  # 标记圆心
            # Define the 10x10 pixel region around the circle center
            x_start = max(0, x - 5)
            y_start = max(0, y - 5)
            x_end = min(hsv.shape[1], x + 5)
            y_end = min(hsv.shape[0], y + 5)
            region = hsv[y_start:y_end, x_start:x_end]#圆中间10*10的区域用来检测颜色
            mean_color = np.mean(region, axis=(0, 1))
        if np.all((lower_black <= mean_color) & (mean_color <= upper_black)):
            text = 'x:'+str(x)+' color:black'
            cv2.putText(frame, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA, 0)  # 显示圆心位置
            black_circles[circle_id] = (x, y, r)

    else:
        # 如果识别不出，显示圆心不存在
        cv2.putText(frame, 'x: None y: None', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA, 0)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('edges', edges)
    return None

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


            circles = findcircle(frame_bin, frame)

            cv2.imshow('frame_bin', frame_bin)
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