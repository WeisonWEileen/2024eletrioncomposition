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
CAP_CENTER = [334, 263]
CAP_LENGTH = 405

BOX_SIZE_RATIO = 0.1
EPSILON_RATIO = 0.05

color = {'red': (0, 0, 255),
         'green': (0, 255, 0),
         'blue': (255, 0, 0),
         'yellow': (0, 255, 255), }

COM_ENABLE = 1
WAYPOINT_LENGTH = 3
WAYPOINT_DIS = 25
MIDPOINT_DIS = 5
VELOCITY = 10
MIDPOINT_VELOCITY = 3
midpoint_bias_ratio = 1.1

# 4 members: red in white, red in black, green in white, green in black
color_range = {'color_red_in_white': {'Lower': np.array([0, 90, 30]), 'Upper': np.array([40, 255, 255])},
               'color_red_in_black': {'Lower': np.array([0, 255, 30]), 'Upper': np.array([40, 255, 255])},
               'color_green_in_white': {'Lower': np.array([60, 40, 100]), 'Upper': np.array([100, 255, 254])},
               'color_green_in_black': {'Lower': np.array([60, 40, 100]), 'Upper': np.array([100, 255, 250])}}


def cap_center_x(value):
    CAP_CENTER[0] = value


def cap_center_y(value):
    CAP_CENTER[1] = value


def cap_length(value):
    global CAP_LENGTH
    CAP_LENGTH = value


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
cv2.createTrackbar('cap_center_x', 'CAP', CAP_CENTER[0], CAP_WIDTH, cap_center_x)
cv2.createTrackbar('cap_center_y', 'CAP', CAP_CENTER[1], CAP_HEIGHT, cap_center_y)
cv2.createTrackbar('cap_length', 'CAP', CAP_LENGTH, CAP_HEIGHT, cap_length)
cv2.namedWindow('Threshld', cv2.WINDOW_NORMAL)
cv2.createTrackbar('H low', 'Threshld', color_range['color_red_in_white']['Lower'][0], 179, red_in_white_h_low)
cv2.createTrackbar('H high', 'Threshld', color_range['color_red_in_white']['Upper'][0], 179, red_in_white_h_high)
cv2.createTrackbar('S low', 'Threshld', color_range['color_red_in_white']['Lower'][1], 255, red_in_white_s_low)
cv2.createTrackbar('S high', 'Threshld', color_range['color_red_in_white']['Upper'][1], 255, red_in_white_s_high)
cv2.createTrackbar('V low', 'Threshld', color_range['color_red_in_white']['Lower'][2], 255, red_in_white_v_low)
cv2.createTrackbar('V high', 'Threshld', color_range['color_red_in_white']['Upper'][2], 255, red_in_white_v_high)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(cap.isOpened())
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)  # 帧宽根据自己的摄像头实际参数
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)  # 帧高根据自己的摄像头实际参数
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# cap.set(cv2.CAP_PROP_FPS, 30) # 帧率
# cap.set(cv2.CAP_PROP_BRIGHTNESS, -40) # 亮度
# cap.set(cv2.CAP_PROP_CONTRAST, 50) # 对比度
# cap.set(cv2.CAP_PROP_SATURATION, 90) # 饱和度
# cap.set(cv2.CAP_PROP_HUE, 90) # 色相
# cap.set(cv2.CAP_PROP_GAMMA, 500) # 伽马
# # cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600) # 白平衡色温
# cap.set(cv2.CAP_PROP_AUTO_WB, 1) # 启用/禁用自动白平衡 0 关闭 1 打开
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 视频流格式

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


class BOX:
    def __init__(self, _box, _left_up_point, _right_up_point, _left_down_point, _right_down_point):
        self.left_up_point = _left_up_point
        self.right_up_point = _right_up_point
        self.left_down_point = _left_down_point
        self.right_down_point = _right_down_point


def findBox(img_bin, frame=None):
    if img_bin is None:
        print("img_bin cannot be None")
        assert True
    min_area = int(BOX_SIZE_RATIO * len(img_bin) * len(img_bin[0]))

    ret = dict()
    cnts, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if len(cnts) == 0:
        return None
    for i in range(0, len(cnts)):
        if cv2.contourArea(cnts[i]) >= min_area:
            epsilon = EPSILON_RATIO * cv2.arcLength(cnts[i], True)
            approx = cv2.approxPolyDP(cnts[i], epsilon, True)

            if len(approx) == 4:
                ret[i] = approx

    for i, cnt in enumerate(cnts):
        # 如果他是个矩形而且他有爹而且爹也是个矩形
        if i in ret.keys() and len(ret[i]) == 4 and hierarchy[0][i][3] != -1 and hierarchy[0][i][
            3] in ret.keys() and len(ret[hierarchy[0][i][3]]) == 4 and hierarchy[0][i][2] == -1:
            if frame is not None:
                cv2.drawContours(frame, [np.intp(cnts[i])], -1, color['red'], 2)
                cv2.drawContours(frame, [np.intp(cnts[hierarchy[0][i][3]])], -1, color['red'], 2)
                cv2.polylines(frame, ret[i], True, color['yellow'], 10)
                cv2.polylines(frame, ret[hierarchy[0][i][3]], True, color['yellow'], 10)
                cv2.imshow("BOX", frame)

            return ret[hierarchy[0][i][3]], ret[i]


def midpoint_detection(img):
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化成GRAY图像
    _, frame_bin = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    boxs = findBox(frame_bin)
    if boxs is None:
        print("No contours!")
        # continue
    else:
        # 匹配角点
        midpoint = np.zeros((4, 2))
        paired = [0, 0, 0, 0]
        cnt = 0
        for i in range(0, 4):
            min_pos = None
            min_dis = 9999
            for j in range(0, 4):
                if paired[j]:
                    continue
                dis = abs(boxs[0][i][0][0] - boxs[1][j][0][0]) + abs(boxs[0][i][0][1] - boxs[1][j][0][1])
                if dis < min_dis:
                    min_dis = dis
                    min_pos = j
            if min_pos is not None:
                paired[min_pos] = 1
                midpoint[i] = (
                    (boxs[0][i][0][0] + boxs[1][min_pos][0][0]) / 2, (boxs[0][i][0][1] + boxs[1][min_pos][0][1]) / 2)
                cnt = cnt + 1
            else:
                assert True
        if cnt != 4:
            return None

        midpoint_sorted = np.zeros_like(midpoint)
        visited = [0, 0, 0, 0]
        element_pre = None
        element_next = 0
        element_now = None
        cnt = 0

        center = [int((midpoint[0][0] + midpoint[1][0] + midpoint[2][0] + midpoint[3][0]) / 4),
                  int((midpoint[0][1] + midpoint[1][1] + midpoint[2][1] + midpoint[3][1]) / 4)]
        l = list()
        r = list()
        midpoint_sorted = list()
        for point in midpoint:
            if point[0] >= center[0]:
                r.append(point)
            else:
                l.append(point)

        if len(r) != 2 or len(l) != 2:
            return None

        r.sort(key=lambda x: x[1], reverse=False)
        l.sort(key=lambda x: x[1], reverse=True)

        if center[0] >= CAP_LENGTH/2 and center[1] < CAP_LENGTH/2:
            midpoint_sorted.append(l[0])
            midpoint_sorted.append(l[1])
            midpoint_sorted.append(r[0])
            midpoint_sorted.append(r[1])
            # midpoint_sorted = l + r
        if center[0] >= CAP_LENGTH/2 and center[1] >= CAP_LENGTH/2:
            midpoint_sorted.append(l[1])
            midpoint_sorted.append(r[0])
            midpoint_sorted.append(r[1])
            midpoint_sorted.append(l[0])
            # midpoint_sorted = l[1] + r + l[0]
        if center[0] < CAP_LENGTH/2 and center[1] < CAP_LENGTH/2:
            midpoint_sorted.append(r[1])
            midpoint_sorted.append(l[0])
            midpoint_sorted.append(l[1])
            midpoint_sorted.append(r[0])
            # midpoint_sorted = r[1] + l + r[0]
        if center[0] < CAP_LENGTH/2 and center[1] >= CAP_LENGTH/2:
            midpoint_sorted.append(r[0])
            midpoint_sorted.append(r[1])
            midpoint_sorted.append(l[0])
            midpoint_sorted.append(l[1])
            # midpoint_sorted = r + l

        # while cnt < 4:
        #     cnt = cnt + 1
        #     element_pre = element_now
        #     element_now = element_next
        #     element_next = None
        #     min_dis = 9999
        #     for j in range(0, 4):
        #         if element_now == j or element_pre == j:
        #             continue
        #         temp = math.sqrt((midpoint[element_now][0] - midpoint[j][0]) * (
        #                 midpoint[element_now][0] - midpoint[j][0]) + (
        #                                  midpoint[element_now][1] - midpoint[j][1]) * (
        #                                  midpoint[element_now][1] - midpoint[j][1]))
        #         if temp < min_dis:
        #             min_dis = temp
        #             element_next = j
        #     if element_next is not None:
        #         midpoint_sorted[cnt - 1] = midpoint[element_now]
        #     else:
        #         return None
        return midpoint_sorted


class IMG:
    def __init__(self, _name, _color, _img, _range, _erode_kernel):
        self.name = _name
        self.color = _color
        self.image = cv2.inRange(_img, _range['Lower'], _range['Upper'])
        self.image = cv2.erode(self.image, _erode_kernel)

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
            cv2.imshow('cnts', frame)
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


# # color: 'green' or 'red'
# min_area_theshold = 10
# def color_tracing(frame, tracing_color):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
#
#     # 二值化
#     kernel = np.ones((3, 3), np.uint8)
#     red_img_list = [IMG('red_in_white', color['red'], hsv, color_range['color_red_in_white'], kernel),
#                     IMG('red_in_black', color['red'], hsv, color_range['color_red_in_black'], kernel), ]
#     green_img_list = [IMG('green_in_white', color['green'], hsv, color_range['color_green_in_white'], kernel),
#                       IMG('green_in_black', color['green'], hsv, color_range['color_green_in_black'], kernel)]
#
#     if (tracing_color == 'red'):
#         cv2.imshow('red_in_white_pcs', red_img_list[0].image)
#         # cv2.imshow('red_in_black_pcs', red_img_list[1].image)
#         for i, img in enumerate(red_img_list):
#             if i == 1:
#                 if red_img_list[i - 1].maxcnt is not None:
#                     continue
#
#             cnts, hierarchy = cv2.findContours(img.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
#             max_cnt = None
#             max_cnt_area = 0
#             for j in range(0, len(cnts)):
#                 if hierarchy[0][j][3] != -1:
#                     cnt_area = cv2.contourArea(cnts[j])
#                     if cnt_area < min_area_theshold:
#                         continue
#                     if cnt_area > max_cnt_area:
#                         max_cnt = cnts[j]
#                         max_cnt_area = cnt_area
#             if max_cnt is None:
#                 for j in range(0, len(cnts)):
#                     cnt_area = cv2.contourArea(cnts[j])
#                     if cnt_area > max_cnt_area:
#                         max_cnt = cnts[j]
#                         max_cnt_area = cnt_area
#                 if max_cnt is not None:
#                     img.maxcnt = max_cnt
#                     img.center, img.radius = cv2.minEnclosingCircle(img.maxcnt)
#                     cv2.circle(frame, (int(img.center[0]), int(img.center[1])), int(img.radius), img.color, 2)
#                     cv2.circle(frame, (int(img.center[0]), int(img.center[1])), 1, img.color, 2)
#                     return img.center
#             else:
#                 img.maxcnt = max_cnt
#                 img.center, img.radius = cv2.minEnclosingCircle(img.maxcnt)
#                 cv2.circle(frame, (int(img.center[0]), int(img.center[1])), int(img.radius), img.color, 2)
#                 cv2.circle(frame, (int(img.center[0]), int(img.center[1])), 1, img.color, 2)
#                 return img.center
#         return None
#
#     if (tracing_color == 'green'):
#         # cv2.imshow('green_in_white_pcs', green_img_list[0].image)
#         # cv2.imshow('green_in_black_pcs', green_img_list[1].image)
#         for i, img in enumerate(green_img_list):
#             if i == 1:
#                 if green_img_list[i - 1].maxcnt is not None:
#                     continue
#
#             cnts, hierarchy = cv2.findContours(img.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
#             max_cnt = None
#             max_cnt_area = 0
#             for j in range(0, len(cnts)):
#                 if hierarchy[0][j][3] != -1:
#                     cnt_area = cv2.contourArea(cnts[j])
#                     if cnt_area < min_area_theshold:
#                         continue
#                     if cnt_area > max_cnt_area:
#                         max_cnt = cnts[j]
#                         max_cnt_area = cnt_area
#             if max_cnt is None:
#                 for j in range(0, len(cnts)):
#                     cnt_area = cv2.contourArea(cnts[j])
#                     if cnt_area > max_cnt_area:
#                         max_cnt = cnts[j]
#                         max_cnt_area = cnt_area
#                 if max_cnt is not None:
#                     img.maxcnt = max_cnt
#                     img.center, img.radius = cv2.minEnclosingCircle(img.maxcnt)
#                     cv2.circle(frame, (int(img.center[0]), int(img.center[1])), int(img.radius), img.color, 2)
#                     cv2.circle(frame, (int(img.center[0]), int(img.center[1])), 1, img.color, 2)
#                     return img.center
#             else:
#                 img.maxcnt = max_cnt
#                 img.center, img.radius = cv2.minEnclosingCircle(img.maxcnt)
#                 cv2.circle(frame, (int(img.center[0]), int(img.center[1])), int(img.radius), img.color, 2)
#                 cv2.circle(frame, (int(img.center[0]), int(img.center[1])), 1, img.color, 2)
#                 return img.center
#         return None


def data_transmission(x, y):
    # x = x * VELOCITY
    # y = y * VELOCITY
    x = x + 32768
    y = y + 32768
    x_h = int(x / 256)
    x_l = int(x % 256)
    y_h = int(y / 256)
    y_l = int(y % 256)
    str = struct.pack("7B", 0xFF, x_h, x_l, y_h, y_l, 0x0D, 0x0A)
    # print(str)
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


def midpoint_cal(midpoint, lastpoint, dis):
    bias = [midpoint[0] - lastpoint[0], midpoint[1] - lastpoint[1]]
    ratio = dis / math.sqrt(bias[0]*bias[0]+bias[1]*bias[1])
    return [bias[0] * ratio + midpoint[0], bias[1] * ratio + midpoint[1]]


def waypoint_create(_midpoint):
    _waypoint_list = list()
    _midpoint_list = list(_midpoint)
    _midpoint_list.append(_midpoint[0])
    _waypoint_list.append([midpoint_cal(_midpoint_list[0], [CAP_LENGTH/2, CAP_LENGTH/2], MIDPOINT_DIS * midpoint_bias_ratio), MIDPOINT_DIS, MIDPOINT_VELOCITY])
    for i in range(0, len(_midpoint_list) - 1):
        _bias = [_midpoint_list[i + 1][0] - _midpoint_list[i][0], _midpoint_list[i + 1][1] - _midpoint_list[i][1]]
        _dis = math.sqrt(_bias[0] * _bias[0] + _bias[1] * _bias[1])
        _waypoint_num = int(_dis / WAYPOINT_LENGTH) + 1

        for j in range(1, _waypoint_num):
            ratio = j / _waypoint_num
            # ratio_2 = 1 - (1 - (j / _waypoint_num)) * (1 - (j / _waypoint_num))
            # ratio_3 = 1 - (1 - (j / _waypoint_num)) * (1 - (j / _waypoint_num)) * (1 - (j / _waypoint_num))
            # ratio_3 = ((j / _waypoint_num)) * ((j / _waypoint_num)) * ((j / _waypoint_num))
            # _waypoint_list.append([[_midpoint_list[i][0] + _bias[0] * ratio,
            #                         _midpoint_list[i][1] + _bias[1] * ratio],
            #                        WAYPOINT_DIS[1] - (WAYPOINT_DIS[1] - WAYPOINT_DIS[0]) * ratio_3, VELOCITY])
            if ratio <= 0.5:
                ratio_t = 1 - ratio * 2
                ratio_2_t = ratio_t ** 2
                _velocity = MIDPOINT_VELOCITY + (VELOCITY - MIDPOINT_VELOCITY) * (1 - ratio_2_t)
                _waypoint_list.append([[_midpoint_list[i][0] + _bias[0] * (1 - ratio_2_t) / 2,
                                        _midpoint_list[i][1] + _bias[1] * (1 - ratio_2_t) / 2],
                                       WAYPOINT_DIS, VELOCITY])
            else:
                ratio_t = (ratio-0.5) * 2
                ratio_2_t = ratio_t ** 2
                _velocity = MIDPOINT_VELOCITY + (VELOCITY - MIDPOINT_VELOCITY) * (1 - ratio_2_t)
                _waypoint_list.append([[_midpoint_list[i][0] + _bias[0] * (ratio_2_t / 2 + 0.5),
                                        _midpoint_list[i][1] + _bias[1] * (ratio_2_t / 2 + 0.5)],
                                       WAYPOINT_DIS, VELOCITY])
        _waypoint_list.append([midpoint_cal(_midpoint_list[i+1], _midpoint_list[i], MIDPOINT_DIS * midpoint_bias_ratio), MIDPOINT_DIS,
                               MIDPOINT_VELOCITY])

    # _waypoint_list.append([_midpoint_list[0], MIDPOINT_DIS, MIDPOINT_VELOCITY])
    return _waypoint_list


box_found = 0

last_time = time.time()

if COM_ENABLE:
    # com = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
    com = serial.Serial('COM11', 115200, timeout=0.001)
print(cap.isOpened())
while cap.isOpened():

    ret, frame = cap.read()
    start_time = time.time()

    if ret:

        if frame is not None:
            # 获取图像
            frame = frame[bound(int(CAP_CENTER[1] - (CAP_LENGTH / 2)), 0, CAP_HEIGHT):bound(
                int(CAP_CENTER[1] + (CAP_LENGTH / 2)), 0, CAP_HEIGHT),
                    bound(int(CAP_CENTER[0] - (CAP_LENGTH / 2)), 0, CAP_WIDTH):bound(
                        int(CAP_CENTER[0] + (CAP_LENGTH / 2)), 0, CAP_WIDTH), :]
            # cv2.imshow('camera', frame)

            # 预处理
            frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
            getimage_time = time.time()
            if COM_ENABLE:
                data = com.read()
                if data:
                    if data[0] == 0xFE:
                        box_found = 1

                        guide_index = 0
                        waypoint_index = 0
                        bias_kalman_x = KAL()
                        bias_kalman_y = KAL()
            elif box_found == 0:

                guide_index = 0
                waypoint_index = 0
                bias_kalman_x = KAL()
                bias_kalman_y = KAL()
                box_found = 1

            uartreceive_time = time.time()

            if box_found == 1:
                # 寻找矩形
                midpoint = midpoint_detection(frame)
                if midpoint is not None:
                    # print('midpoint:')
                    # print(midpoint)
                    box_found = 2
                    waypoint = waypoint_create(midpoint)

                    # print
                    for i in range(0, len(waypoint)):
                        cv2.circle(frame, (int(waypoint[i][0][0]), int(waypoint[i][0][1])), int(waypoint[i][1]),
                                   color['blue'], 1)
                        # cv2.circle(frame, (int(waypoint[i][0][0]), int(waypoint[i][0][1])), 1,
                        #            color['blue'], 1)
                    cv2.imshow("waypoint", frame)
                    # cv2.waitKey(0)

            elif box_found == 2:
                # 引导激光
                laser_center_t = color_tracing(frame, 'red')
                lasertracing_time = time.time()
                if laser_center_t is not None:
                    # print('Laser:   %.3f     %.3f'%(laser_center_t[0], laser_center_t[1]))
                    laser_center = np.zeros(2, float)
                    laser_center[0] = kalman_update(int(laser_center_t[0]), bias_kalman_x)[0]
                    laser_center[1] = kalman_update(int(laser_center_t[1]), bias_kalman_y)[0]
                    cv2.line(frame, (int(waypoint[waypoint_index][0][0]), int(waypoint[waypoint_index][0][1])),
                             (int(laser_center[0]), int(laser_center[1])),
                             color['blue'], 1)
                    bias = [(waypoint[waypoint_index][0][0] - laser_center[0]) * waypoint[waypoint_index][2],
                            (waypoint[waypoint_index][0][1] - laser_center[1]) * waypoint[waypoint_index][2]]
                    cv2.circle(frame, (int(waypoint[waypoint_index][0][0]), int(waypoint[waypoint_index][0][1])),
                               int(waypoint[waypoint_index][1]),
                               color['blue'], 1)

                    # UART transmission
                    data_transmission(bias[0], bias[1])
                    if math.sqrt(bias[0] * bias[0] + bias[1] * bias[1]) < waypoint[waypoint_index][1] * \
                            waypoint[waypoint_index][2]:
                        waypoint_index = waypoint_index + 1
                        if waypoint_index == 1:
                            kalman_enable(laser_center[0], bias_kalman_x)
                            kalman_enable(laser_center[1], bias_kalman_y)
                    if waypoint_index == len(waypoint):
                        print("complete!")
                        if COM_ENABLE:
                            str = struct.pack("7B", 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0D, 0x0A)
                            com.write(str)
                        box_found = 3
                uarttransmit_time = time.time()
                print('getimage_time:   %.3fms' % ((getimage_time - start_time) * 1000))
                print('uartreceive_time:   %.3fms' % ((uartreceive_time - getimage_time) * 1000))
                print('lasertracing_time:   %.3fms' % ((lasertracing_time - uartreceive_time) * 1000))
                print('uarttransmit_time:   %.3fms' % ((uarttransmit_time - lasertracing_time) * 1000))


            cv2.circle(frame, (int(CAP_LENGTH/2), int(CAP_LENGTH/2)), 1, color['red'], 2)

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
    print('all_time:    %.3fms,        %.3fFPS' % ((end_time - last_time) * 1000, 1 / (end_time - last_time)))
    last_time = end_time

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
