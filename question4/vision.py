
import serial
import struct
from time import sleep
import cv2 
import numpy as np 

lower_white = np.array([0, 0, 150])  # 设定白色的阈值下限
upper_white = np.array([180, 40, 255])  # 设定白色的阈值上限

lower_black = np.array([0, 0, 0])  # 设定黑色的阈值下限
upper_black = np.array([180, 255, 46])  # 设定黑色的阈值上限

# 标定的情况下(写死的情况下得到的9的像素点的坐标)
centers = [ 
    [182, 150],[398, 155],
    [614, 160],[178, 365],
    [394, 369],[609, 374],
    [174, 578],[390, 583],
    [603, 586]  
]

def show_phase(phase):
    """显示局面"""
    
    for i in range(3):
        for j in range(3):
            if phase[i,j] == 1: 
                chessman = chr(0x25cf)
            elif phase[i,j] == 2:
                chessman = chr(0x25cb)
            elif phase[i,j] == 0:
                chessman = chr(0x2606)
            print('\033[0;30;43m' + chessman + '\033[0m', end='')
        print()

def get_board(boxs,hsv):
    phase = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=np.ubyte)

    for i in range(9):
        x = boxs[i][0]
        y = boxs[i][1]
        x_start = max(0, int(x) - 20)
        y_start = max(0, int(y) - 20)
        x_end = min(hsv.shape[1], int(x) + 20)
        y_end = min(hsv.shape[0], int(y) + 20)
        region = hsv[y_start:y_end, x_start:x_end]#棋格中间的区域用来检测颜色
        mean_color = np.mean(region, axis=(0, 1))
        if np.all((lower_black <= mean_color) & (mean_color <= upper_black)):
            phase[i//3][i%3] = 1 #黑色1
        if np.all((lower_white <= mean_color) & (mean_color <= upper_white)):
            phase[i//3][i%3] = 2 #白色2/
    # show_phase(phase)
    # print(phase)
    return phase


def detect(frame):
        # 预处理
    gs_frame = cv2.GaussianBlur(frame, (7, 7), 0)  # 高斯模糊
    # cv2.imshow('gs_frame', gs_frame)
    frame_gray = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2GRAY)  # 转化成GRAY图像
    # _, frame_bin = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold_value = cv2.getTrackbarPos('Threshold', 'camera')
    _, frame_bin = cv2.threshold(frame_gray, threshold_value, 255, cv2.THRESH_BINARY)

    #hsv图像
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换为HSV空间 

    phase = get_board(centers,hsv)
