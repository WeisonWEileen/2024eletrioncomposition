"""
An implementation of Minimax AI Algorithm in Tic Tac Toe,
using Python.
This software is available under GPL license.
Author: Weison Pan 
Year: 2024
License: GNU GENERAL PUBLIC LICENSE (GPL)
"""

# 问题描述
# 装置执黑棋(1)先行与人对弈（第 1 步方格可设置）,1
# 若人(-1)应对的第 1 步白, -1
# 棋有错误，装置能获胜。
# X对应的是白色棋子,O对应的是黑色棋子


#!/usr/bin/env python3
from math import inf as infinity
from random import choice
import platform
import time
from os import system
import cv2
import numpy as np

import stm32_serial
import vision
import utils
import struct  



HUMAN = -1
COMP = +1
board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

# 映射关系
moves = {
    1: [0, 0], 2: [0, 1], 3: [0, 2],
    4: [1, 0], 5: [1, 1], 6: [1, 2],
    7: [2, 0], 8: [2, 1], 9: [2, 2],
}


# 设置摄像头的参数
print((f"start init camera"))
CAP_WIDTH = 1920
CAP_HEIGHT = 1080

CAP_CENTER = (663,575)
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
# centers = [ [182, 150]  ,  [398, 155]  ,  [619, 131]  ,  [178, 365]  ,  [394, 369]  ,  [609, 374]  ,  [174, 578]  ,  [390, 583]  ,  [603, 586]  ]

# 鼠标点击得到
centers = [ 
    [148, 134]  ,  [384, 128]  ,  [619, 134]  ,  
    [145, 374]  ,  [389, 369]  ,  [628, 363]  ,  
    [145, 614]  ,  [390, 610]  ,  [631, 603]  ]




def preprocess_image(frame):
    gs_frame = cv2.GaussianBlur(frame, (7, 7), 0)  # 高斯模糊
    # cv2.imshow('gs_frame', gs_frame)
    frame_gray = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2GRAY)  # 转化成GRAY图像
    # _, frame_bin = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold_value = cv2.getTrackbarPos('Threshold', 'camera')


    _, frame_bin = cv2.threshold(frame_gray, threshold_value, 255, cv2.THRESH_BINARY)

    #hsv图像
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换为HSV空间
    return hsv





def evaluate(state):
    """
    Function to heuristic evaluation of state.
    :param state: the state of the current board
    :return: +1 if the computer wins; -1 if the human wins; 0 draw
    """
    if wins(state, COMP):
        score = +1
    elif wins(state, HUMAN):
        score = -1
    else:
        score = 0

    return score


def wins(state, player):
    """
    This function tests if a specific player wins. Possibilities:
    * Three rows    [X X X] or [O O O]
    * Three cols    [X X X] or [O O O]
    * Two diagonals [X X X] or [O O O]
    :param state: the state of the current board
    :param player: a human or a computer
    :return: True if the player wins
    """
    win_state = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[2][0], state[1][1], state[0][2]],
    ]
    if [player, player, player] in win_state:
        return True
    else:
        return False


def game_over(state):
    """
    This function test if the human or computer wins
    :param state: the state of the current board
    :return: True if the human or computer wins
    """
    return wins(state, HUMAN) or wins(state, COMP)


def empty_cells(state):
    """
    Each empty cell will be added into cells' list
    :param state: the state of the current board
    :return: a list of empty cells
    """
    cells = []

    for x, row in enumerate(state):
        for y, cell in enumerate(row):
            if cell == 0:
                cells.append([x, y])

    return cells


def valid_move(x, y):
    """
    A move is valid if the chosen cell is empty
    :param x: X coordinate
    :param y: Y coordinate
    :return: True if the board[x][y] is empty
    """
    if [x, y] in empty_cells(board):
        return True
    else:
        return False


def set_move(x, y, player):
    """
    Set the move on board, if the coordinates are valid
    :param x: X coordinate
    :param y: Y coordinate
    :param player: the current player
    """
    if valid_move(x, y):
        board[x][y] = player
        return True
    else:
        return False


def minimax(state, depth, player):
    """
    AI function that choice the best move
    :param state: current state of the board
    :param depth: node index in the tree (0 <= depth <= 9),
    but never nine in this case (see iaturn() function)
    :param player: an human or a computer
    :return: a list with [the best row, best col, best score]
    """
    if player == COMP:
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]

    if depth == 0 or game_over(state):
        score = evaluate(state)
        return [-1, -1, score]

    for cell in empty_cells(state):
        x, y = cell[0], cell[1]
        state[x][y] = player
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == COMP:
            if score[2] > best[2]:
                best = score  # max value
        else:
            if score[2] < best[2]:
                best = score  # min value

    return best


def clean():
    """
    Clears the console
    """
    os_name = platform.system().lower()
    if 'windows' in os_name:
        system('cls')
    else:
        system('clear')


def render(state, c_choice, h_choice):
    """
    Print the board on console
    :param state: current state of the board
    """

    chars = {
        -1: h_choice,
        +1: c_choice,
        0: ' '
    }
    str_line = '---------------'

    print('\n' + str_line)
    for row in state:
        for cell in row:
            symbol = chars[cell]
            print(f'| {symbol} |', end='')
        print('\n' + str_line)


def ai_turn(ser,c_choice, h_choice):
    """
    It calls the minimax function if the depth < 9,
    else it choices a random coordinate.
    :param c_choice: computer's choice X or O
    :param h_choice: human's choice X or O
    :return:
    """
    # 通过摄像头获取当前的 boards 的变量状态,先保证读到的图像是正常的,过滤掉前面几帧
    count = 0
    while count < 10:
        ret, frame = cap.read()
        count += 1
        cv2.imshow('image_raw', frame)
    
    ret, frame = cap.read()
    if ret:
        if frame is not None:
            # 获取图像
            frame = frame[int(CAP_CENTER[1] - (CAP_LENGTH / 2)):int(CAP_CENTER[1] + (CAP_LENGTH / 2)),
                    int(CAP_CENTER[0] - (CAP_LENGTH / 2)):int(CAP_CENTER[0] + (CAP_LENGTH / 2)), :]

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv_frame)
    board = vision.get_board(centers,hsv_frame)
    print("get from camera")
    render(board, c_choice, h_choice)
    print("get from camera")


    depth = len(empty_cells(board))
    if depth == 0 or game_over(board):
        return
    
    if depth == 9:
        x = choice([0, 1, 2])
        y = choice([0, 1, 2])
        ai_next_key = utils.get_key(moves,[x,y])

        print(f"决策出来,下一步棋子是{ai_next_key},串口发送启动")
        stm32_serial.send_data(ser,ai_next_key[0])
    else:

        # clean()
        print('')
        print(f'Computer turn [{c_choice}]')
        render(board, c_choice, h_choice)



        move = minimax(board, depth, COMP)
        x, y = move[0], move[1]
        
        # 通过串口发送数据给stm32
        ai_next_key = utils.get_key(moves,[x,y])


        print(f"决策出来,下一步棋子是{ai_next_key},串口发送启动")
        stm32_serial.send_data(ser,ai_next_key[0])
    

    set_move(x, y, COMP)

def human_turn(ser):
    """
    直接使用串口延时
    """
    depth = len(empty_cells(board))
    if depth == 0 or game_over(board):
        return

    # clean()

    print('')
    print(f'人类等待按键按下,轮到ai下棋 -----')

    # 收到一帧就进入这个循环(案件按下,推出发送设定的东西)
    while True:   
        if ser.in_waiting > 0:  
            break



def main():

    
    ser = stm32_serial.init_serial()
    
    """
    Main function that calls all functions
    """
    # 手动设置成两个的标识符

    # clean()
    print('')
    h_choice = 'X'  # X or O
    c_choice = "O"  # X or O
    first_flag = True  #第四题,让AI随机先行,然后第一步棋子由用户输入来设置
    


    # Human may starts first
    # clean()
    print('')

    # Main loop of this game
    while len(empty_cells(board)) > 0 and not game_over(board):
        
        ser.flushInput()
        ser.flushOutput()

        while(True):
            if ser.in_waiting > 0:
                frame_header = ser.read(1)
                if frame_header != b'\x0A':
                    continue
                    # return None  # 帧头不匹配
                data = ser.read(1)
                print(f"收到串口数据: {data[0]}")
                if data[0] == 2:
                    break





                # if data[0] == 0x0A and data[2] == 0x0B:
                #     print(f"收到串口数据: {data[1]}")
                #     if(data[1] == 0x01):
                #         break

        ser.flushInput()
        ser.flushOutput()

        ai_turn(ser,c_choice, h_choice)

        ser.flushInput()
        ser.flushOutput()
         
        # human_turn(ser)

    
    # Game over message
    if wins(board, HUMAN):
        # clean()
        print('')
        print(f'Human turn [{h_choice}]')
        render(board, c_choice, h_choice)
        print('YOU WIN!')
    elif wins(board, COMP):
        # clean()
        print('')
        print(f'Computer turn [{c_choice}]')
        render(board, c_choice, h_choice)
        print('YOU LOSE!')
    else:
        # clean()
        print('')
        render(board, c_choice, h_choice)
        print('DRAW!')

    exit()


if __name__ == '__main__':
    main()