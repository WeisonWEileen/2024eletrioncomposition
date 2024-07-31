import cv2
import numpy as np
import math
# import serial
import struct



cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)

CAP_WIDTH = 1920
CAP_HEIGHT = 1080


# ROI
# 原来zcb的在没有装上装置的时候
# CAP_CENTER = (1080, 607)
# CAP_LENGTH = 800


CAP_CENTER = (910,607)
CAP_LENGTH = 1050

BOX_SIZE_RATIO = 0.1
EPSILON_RATIO = 0.05


# color = {'red': (0, 0, 255),
#          'green': (0, 255, 0),
#          'blue': (255, 0, 0),
#          'yellow': (0, 255, 255), }

# # 4 members: red in white, red in black, green in white, green in black
# color_range = {'color_red_in_white': {'Lower': np.array([0, 40, 240]), 'Upper': np.array([40, 255, 255])},
#                'color_red_in_black': {'Lower': np.array([0, 40, 100]), 'Upper': np.array([40, 255, 250])},
#                'color_green_in_white': {'Lower': np.array([60, 40, 100]), 'Upper': np.array([100, 255, 254])},
#                'color_green_in_black': {'Lower': np.array([60, 40, 100]), 'Upper': np.array([100, 255, 250])}}

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

lower_white = np.array([0, 0, 150])  # 设定白色的阈值下限
upper_white = np.array([180, 40, 255])  # 设定白色的阈值上限

lower_black = np.array([0, 0, 0])  # 设定黑色的阈值下限
upper_black = np.array([180, 255, 46])  # 设定黑色的阈值上限

def findBox(img_bin, frame=None):
    if img_bin is None:
        print("img_bin cannot be None")
        assert True
    min_area = int(BOX_SIZE_RATIO * len(img_bin) * len(img_bin[0]))

    ret = []
    # cnts, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnts,hierachy = cv2.findContours(img_bin, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None
    print(f"the number of counters detected is {len(cnts)}")
    
    i=0
    
    ret_num=0
    hierach_id = 0
    inner_class =[]
    while(hierachy[0][i][2]!=-1 and hierachy[0][i][1]==-1 and i<len(cnts) and hierachy[0][i][3]!=-1):
        hierach_id = i
        i+=1
    while(hierachy[0][hierach_id][0]!=-1):
        inner_class.append(cnts[hierach_id])
        # if (cv2.contourArea(cnts[hierach_id])>min_area):
        #     print("larger than min_area")
        hierach_id=hierachy[0][hierach_id][0]
        # print(hierach_id,hierachy[0][hierach_id])
    # for cnt in inner_class: # 画图
    #     cv2.drawContours(frame,[cnt],0,(122,122,0),3)
    # for i in range(len(cnts)):
    #     cv2.drawContours(frame, cnts, i, (0, 255, 0), 2) 
    approx = None
    for cnt in inner_class:
        if cv2.contourArea(cnt) >= 1000:
            epsilon = EPSILON_RATIO * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
        # epsilon = EPSILON_RATIO * cv2.arcLength(cnt, True) #直接检测矩形跳过面积阈值
        # approx = cv2.approxPolyDP(cnt, epsilon, True)
        # print(f"len of approx is {len(approx)}")

        if approx is not None:
            if len(approx) == 4: #如果是矩形
                ret.append(np.round(np.add(np.add(approx[0][0],approx[1][0]),np.add(approx[2][0],approx[3][0]))*0.25).astype(int))
                cv2.circle(frame, tuple(ret[ret_num]), 5, (0, 0, 255), -1)
                ret_num += 1
                # print(approx)
                for point in approx:
                    cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)
            y = [item[1] for item in ret]
            y_indices = np.argsort(y)
            ret = [ret[i] for i in y_indices] #按y坐标排序
            if len(ret)>=9:
                x1 = [item[0] for item in ret[:3]]
                x2 = [item[0] for item in ret[3:6]]
                x3 = [item[0] for item in ret[6:]]
                x1_indices = np.argsort(x1)
                x2_indices = np.argsort(x2)
                x3_indices = np.argsort(x3)
                ret[:3] = [ret[i] for i in x1_indices]
                ret[3:6] = [ret[i+3] for i in x2_indices]
                ret[6:] = [ret[i+6] for i in x3_indices]
                for i in range(9):
                    cv2.putText(frame, f"{i+1}", tuple(ret[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return ret

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

def getchessphase(boxs,hsv):
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
            phase[i//3,i%3] = 1 #黑色1
        if np.all((lower_white <= mean_color) & (mean_color <= upper_white)):
            phase[i//3,i%3] = 2 #白色2/
    show_phase(phase)
    print(phase)

    return phase

boad_detect = 0

def blue_thres(frame):
        # Convert the image from RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Split the image into B, G, R channels
    b, g, r = cv2.split(frame)

    # Find the maximum value among the B, G, R channels
    max_val = np.maximum(np.maximum(b, g), r)

    # Create a mask where the B channel is the maximum
    mask = np.where(b == max_val, 255, 0).astype(np.uint8)

    return mask





while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if frame is not None:
            # 获取图像
            frame = frame[int(CAP_CENTER[1] - (CAP_LENGTH / 2)):int(CAP_CENTER[1] + (CAP_LENGTH / 2)),
                    int(CAP_CENTER[0] - (CAP_LENGTH / 2)):int(CAP_CENTER[0] + (CAP_LENGTH / 2)), :]
            
            mask = blue_thres(frame)
            

            


            # 预处理
            # gs_frame = cv2.GaussianBlur(frame, (7, 7), 0)  # 高斯模糊
            # # cv2.imshow('gs_frame', gs_frame)
            # frame_gray = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2GRAY)  # 转化成GRAY图像
            # _, frame_bin = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            #hsv图像
            # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换为HSV空间
            # boxs = findBox(frame_bin, frame)
            # if len(boxs) == 9:
            #     for i in boxs:
            #         cv2.circle(frame, tuple(i), 5, (0, 0, 255), -1)
            #     phase = getchessphase(boxs,hsv)
            # if boad_detect < 50:
            #     boxs = findBox(frame_bin, frame)
            #     if len(boxs) == 9:
            #         boad_detect += 1
            # if boad_detect == 50:#跳过前10帧检测到的棋盘
            #     for i in boxs:
            #         cv2.circle(frame, tuple(i), 5, (0, 0, 255), -1)
            #     phase = getchessphase(boxs,hsv)
            # print(boxs)

            cv2.imshow('camera', frame)
            # cv2.imshow('gray scale image', frame_gray)

            cv2.imshow('frame_bin', mask)

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
