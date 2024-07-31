import cv2
import numpy as np
import math
import serial
import struct
import time

# cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)

CAP_WIDTH = 640
CAP_HEIGHT = 480
# ROI
CAP_CENTER = [320, 240]
CAP_LENGTH = 240

GREEN_LASER_CENTER = [38, 54]
TRACING_DIS = 1

color = {'red': (0, 0, 255),
         'green': (0, 255, 0),
         'blue': (255, 0, 0),
         'yellow': (0, 255, 255), }

COM_ENABLE = 0

# 4 members: red in white, red in black, green in white, green in black
color_range = {'color_red_in_white': {'Lower': np.array([0, 90, 70]), 'Upper': np.array([40, 255, 255])},
               'color_red_in_black': {'Lower': np.array([0, 255, 30]), 'Upper': np.array([40, 255, 255])}}

def cap_center_x(value):
    CAP_CENTER[0] = value

def cap_center_y(value):
    CAP_CENTER[1] = value

def cap_length(value):
    global CAP_LENGTH
    CAP_LENGTH = value

def green_laser_center_x(value):
    GREEN_LASER_CENTER[0] = int(CAP_LENGTH/2 + (value - 50))
    if GREEN_LASER_CENTER[0] < 0:
        GREEN_LASER_CENTER[0] = 0

def green_laser_center_y(value):
    GREEN_LASER_CENTER[1] = int(CAP_LENGTH/2 + (value - 50))
    if GREEN_LASER_CENTER[1] < 0:
        GREEN_LASER_CENTER[1] = 0
def tracing_dis(value):
    global TRACING_DIS
    TRACING_DIS = value

def red_in_white_h_low(value):
    color_range['color_red_in_white']['Lower'][0] = value


def red_in_white_s_low(value):
    color_range['color_red_in_white']['Lower'][1] = value


def red_in_white_v_low(value):
    color_range['color_red_in_white']['Lower'][2] = value


def red_in_white_h_high(value):
    color_range['color_red_in_white']['Upper'][0] = value


def red_in_white_s_high(value):
    color_range['color_red_in_white']['Upper'][1] = value


def red_in_white_v_high(value):
    color_range['color_red_in_white']['Upper'][2] = value


def bound(x, minn, maxn):
    if x > maxn:
        return maxn
    elif x < minn:
        return minn
    else:
        return x


cv2.namedWindow('CAP', cv2.WINDOW_NORMAL)
cv2.createTrackbar('cap_center_x', 'CAP', 320, CAP_WIDTH, cap_center_x)
cv2.createTrackbar('cap_center_y', 'CAP', 240, CAP_HEIGHT, cap_center_y)
cv2.createTrackbar('cap_length', 'CAP', CAP_LENGTH, CAP_HEIGHT, cap_length)
cv2.createTrackbar('green_laser_center_x', 'CAP', GREEN_LASER_CENTER[0], 100, green_laser_center_x)
cv2.createTrackbar('green_laser_center_y', 'CAP', GREEN_LASER_CENTER[1], 100, green_laser_center_y)
cv2.createTrackbar('tracing_dis', 'CAP', TRACING_DIS, 50, tracing_dis)
cv2.namedWindow('Threshld', cv2.WINDOW_NORMAL)
cv2.createTrackbar('H low', 'Threshld', color_range['color_red_in_white']['Lower'][0], 179, red_in_white_h_low)
cv2.createTrackbar('H high', 'Threshld', color_range['color_red_in_white']['Upper'][0], 179, red_in_white_h_high)
cv2.createTrackbar('S low', 'Threshld', color_range['color_red_in_white']['Lower'][1], 255, red_in_white_s_low)
cv2.createTrackbar('S high', 'Threshld', color_range['color_red_in_white']['Upper'][1], 255, red_in_white_s_high)
cv2.createTrackbar('V low', 'Threshld', color_range['color_red_in_white']['Lower'][2], 255, red_in_white_v_low)
cv2.createTrackbar('V high', 'Threshld', color_range['color_red_in_white']['Upper'][2], 255, red_in_white_v_high)

cap = cv2.VideoCapture(0 + cv2.CAP_VFW)
print(cap.isOpened())
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)  # 帧宽根据自己的摄像头实际参数
# # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)  # 帧高根据自己的摄像头实际参数
# # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# cap.set(cv2.CAP_PROP_FPS, 30)  # 帧率
# cap.set(cv2.CAP_PROP_BRIGHTNESS, -40)  # 亮度
# cap.set(cv2.CAP_PROP_CONTRAST, 50)  # 对比度
# cap.set(cv2.CAP_PROP_SATURATION, 90)  # 饱和度
# cap.set(cv2.CAP_PROP_HUE, 90)  # 色相
# cap.set(cv2.CAP_PROP_GAMMA, 500)  # 伽马
# cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600) # 白平衡色温
# cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # 启用/禁用自动白平衡 0 关闭 1 打开
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 视频流格式


# cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)  # 宽度
# CAP_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)  # 宽度
# CAP_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 宽度
# # # cap.set(cv2.CAP_PROP_FPS, 30)	            # 帧率 帧/秒
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)  # 亮度
# cap.set(cv2.CAP_PROP_CONTRAST, 50)  # 对比度
# cap.set(cv2.CAP_PROP_SATURATION, 100)  # 饱和度
# cap.set(cv2.CAP_PROP_HUE, 20)  # 色调 50
# cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 曝光

class KAL:
    def __init__(self):
        self.enabled = False

        # 定义一些常量
        self.dt = 1  # 时间步长
        self.A = np.array([[1, self.dt], [0, 1]])  # 状态转移矩阵
        self.B = np.array([[0], [0]])  # 控制输入矩阵 (没有用到)
        self.H = np.array([[1, 0]])  # 测量矩阵
        self.q = 1e-5  # 系统噪声方差
        self.r = 0.1  # 测量噪声方差

        # 定义初始状态
        self.K = 0
        self.x = np.array([[0], [0]])  # 状态变量 (位置和速度)
        self.P = np.array([[0.01, 0.001], [0.001, 0.0001]])  # 状态协方差矩阵
        self.x_pred = 0
        self.p_pred = np.array([[0.01, 0.001], [0.001, 0.0001]])




class IMG:
    def __init__(self, _name, _color, _img, _range, _erode_kernel):
        self.name = _name
        self.color = _color
        self.image = cv2.inRange(_img, _range['Lower'], _range['Upper'])
        self.image = cv2.erode(self.image, _erode_kernel)
        self.image = cv2.dilate(self.image, _erode_kernel)

        self.maxcnt = None
        self.center = None
        self.radius = None


# color: 'green' or 'red'
hole_min_area_theshold = 20
# hole_max_area_theshold = 250
ring_min_area_theshold = 20
# ring_max_area_theshold = 600


def color_tracing(frame, tracing_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像

    # 二值化
    kernel = np.ones((3, 3), np.uint8)
    red_img_list = [IMG('red_in_white', color['red'], hsv, color_range['color_red_in_white'], kernel),
                    IMG('red_in_black', color['red'], hsv, color_range['color_red_in_black'], kernel), ]

    if (tracing_color == 'red'):
        cv2.imshow('red_in_white_pcs', red_img_list[0].image)
        # cv2.imshow('red_in_black_pcs', red_img_list[1].image)
        for i, img in enumerate(red_img_list[0:1]):
            # if i == 1:
            #     if red_img_list[i - 1].maxcnt is not None:
            #         continue

            cnts, hierarchy = cv2.findContours(img.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            max_cnt = None
            max_cnt_area = 0
            for j in range(0, len(cnts)):
                cv2.drawContours(frame, cnts, j, color['red'], 1)
                if hierarchy[0][j][3] != -1:
                    cnt_area = cv2.contourArea(cnts[j])
                    if cnt_area < hole_min_area_theshold:
                    # if cnt_area < hole_min_area_theshold or cnt_area > hole_max_area_theshold:
                        continue
                    if cnt_area > max_cnt_area:
                        max_cnt = cnts[j]
                        max_cnt_area = cnt_area
            # cv2.imshow('cnts', frame)
            if max_cnt is None:
                for j in range(0, len(cnts)):
                    cnt_area = cv2.contourArea(cnts[j])
                    if cnt_area < ring_min_area_theshold:
                    # if cnt_area < hole_min_area_theshold or cnt_area > hole_max_area_theshold:
                        continue
                    if cnt_area > max_cnt_area:
                        max_cnt = cnts[j]
                        max_cnt_area = cnt_area
                if max_cnt is not None:
                    img.maxcnt = max_cnt
                    img.center, img.radius = cv2.minEnclosingCircle(img.maxcnt)
                    cv2.circle(frame, (int(img.center[0]), int(img.center[1])), int(img.radius), img.color, 2)
                    cv2.circle(frame, (int(img.center[0]), int(img.center[1])), 1, img.color, 2)
                    return img.center
            else:
                img.maxcnt = max_cnt
                img.center, img.radius = cv2.minEnclosingCircle(img.maxcnt)
                cv2.circle(frame, (int(img.center[0]), int(img.center[1])), int(img.radius), img.color, 2)
                cv2.circle(frame, (int(img.center[0]), int(img.center[1])), 1, img.color, 2)
                return img.center
        return None
    return None


def data_transmission(x, y):
    x = x + 32768
    y = y + 32768
    x_h = int(x / 256)
    x_l = int(x % 256)
    y_h = int(y / 256)
    y_l = int(y % 256)
    str = struct.pack("7B", 0xFF, x_h, x_l, y_h, y_l, 0x0D, 0x0A)
    print(str)
    if COM_ENABLE:
        com.write(str)


# 卡尔曼滤波

def kalman_enable(value, kalman: KAL):
    kalman.enabled = True
    kalman.K = 0
    kalman.x = np.array([[value], [0]])  # 状态变量 (位置和速度)
    kalman.P = np.array([[0.01, 0.001], [0.001, 0.0001]])  # 状态协方差矩阵
    kalman.x_pred = value
    kalman.p_pred = np.array([[0.01, 0.001], [0.001, 0.0001]])


def kalman_update(measurement, kalman: KAL):
    if not kalman.enabled:
        kalman.x[0] = measurement
    else:
        kalman.x_pred = kalman.A @ kalman.x  # 预测状态
        kalman.P_pred = kalman.A @ kalman.P @ kalman.A.T + kalman.q  # 预测状态协方差矩阵

        # 更新步骤
        kalman.K = kalman.P_pred @ kalman.H.T / (kalman.H @ kalman.P_pred @ kalman.H.T + kalman.r)  # 卡尔曼增益
        kalman.x = kalman.x_pred + kalman.K * (measurement - kalman.H @ kalman.x_pred)  # 校正状态
        kalman.P = (np.eye(2) - kalman.K @ kalman.H) @ kalman.P_pred  # 校正状态协方差矩阵

    return kalman.x


bias_kalman_x = KAL()
bias_kalman_y = KAL()

lost_cnt = 0

if COM_ENABLE:
    com = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.001)
    # com = serial.Serial('COM9', 115200, timeout=0.1)
while cap.isOpened():

    start_time = time.time()
    ret, frame = cap.read()

    if ret:

        if frame is not None:
            # 获取图像
            frame = frame[bound(int(CAP_CENTER[1] - (CAP_LENGTH / 2)), 0, CAP_HEIGHT):bound(int(CAP_CENTER[1] + (CAP_LENGTH / 2)), 0, CAP_HEIGHT),
                          bound(int(CAP_CENTER[0] - (CAP_LENGTH / 2)), 0, CAP_WIDTH) :bound(int(CAP_CENTER[0] + (CAP_LENGTH / 2)), 0, CAP_WIDTH), :]
            # cv2.imshow('camera', frame)

            # 预处理
            frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊

            # 引导激光
            laser_center_t = color_tracing(frame, 'red')
            if laser_center_t is not None:
                lost_cnt = 0
                # print('Laser:   %.3f     %.3f' % (laser_center_t[0], laser_center_t[1]))
                laser_center = np.zeros(2, float)
                # if bias_kalman_x.enabled == 0:
                #     kalman_enable(laser_center[0], bias_kalman_x)
                #     kalman_enable(laser_center[1], bias_kalman_y)
                laser_center[0] = kalman_update(laser_center_t[0], bias_kalman_x)[0]
                laser_center[1] = kalman_update(laser_center_t[1], bias_kalman_y)[0]

                bias = [laser_center[0] - GREEN_LASER_CENTER[0],
                        laser_center[1] - GREEN_LASER_CENTER[1]]
                print('Bias:   %.3f     %.3f' % (bias[0], bias[1]))

                if(bias[0]*bias[0] + bias[1]*bias[1] > TRACING_DIS * TRACING_DIS):
                    data_transmission(bias[0], bias[1])
                else:
                    print('reached! DIS:    %.3f'%(math.sqrt(bias[0]*bias[0] + bias[1]*bias[1])))
                    data_transmission(0, 0)
                cv2.circle(frame, (int(laser_center[0]), int(laser_center[1])), TRACING_DIS, color['red'], 1)
                cv2.line(frame, (int(laser_center[0]), int(laser_center[1])), (GREEN_LASER_CENTER[0], GREEN_LASER_CENTER[1]), color['green'], 1)
            else:
                lost_cnt = lost_cnt + 1
                if lost_cnt >= 10:
                    data_transmission(0, 1000)

            cv2.circle(frame, (int(GREEN_LASER_CENTER[0]), int(GREEN_LASER_CENTER[1])), 5, color['green'], 1)
            cv2.circle(frame, (int(GREEN_LASER_CENTER[0]), int(GREEN_LASER_CENTER[1])), 1, color['green'], 1)
            cv2.imshow("result", frame)

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

    end_time = time.time()
    print('%.3fms,        %.3fFPS' % ((end_time - start_time) * 1000, 1 / (end_time - start_time)))

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
