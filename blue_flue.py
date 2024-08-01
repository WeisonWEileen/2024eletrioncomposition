from math import inf as infinity
from random import choice
import platform
import time
from os import system
import cv2
import numpy as np

CAP_WIDTH = 1920
CAP_HEIGHT = 1080
CAP_CENTER = (792,595)
CAP_LENGTH = 760
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
print((f"now the CAP_WIDTH and CAP_HEIGHT{CAP_WIDTH},{CAP_HEIGHT}"))

print((f"start to init thres params"))
lower_white = np.array([0, 0, 150])  # 设定白色的阈值下限
upper_white = np.array([180, 40, 255])  # 设定白色的阈值上限

lower_black = np.array([0, 0, 0])  # 设定黑色的阈值下限
upper_black = np.array([180, 255, 46])  # 设定黑色的阈值上限

# 标定的情况下(写死的情况下得到的9的像素点的坐标)
centers = [ [182, 150]  ,  [398, 155]  ,  [614, 160]  ,  [178, 365]  ,  [394, 369]  ,  [609, 374]  ,  [174, 578]  ,  [390, 583]  ,  [603, 586]  ]