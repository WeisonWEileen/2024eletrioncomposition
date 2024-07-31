import numpy as np

# 定义一些常量
dt = 1        # 时间步长
A = np.array([[1, dt], [0, 1]])       # 状态转移矩阵
B = np.array([[0], [0]])              # 控制输入矩阵 (没有用到)
H = np.array([[1, 0]])                # 测量矩阵
q = 1e-5     # 系统噪声方差
r = 0.1      # 测量噪声方差

# 定义初始状态
x = np.array([[0], [0]])   # 状态变量 (位置和速度)
P = np.diag([10, 10])  # 状态协方差矩阵

# 定义测量结果
measurements = [1, 2, 3, 4, 5]

# 卡尔曼滤波
for measurement in measurements:
    # 预测步骤
    x_pred = A @ x             # 预测状态
    P_pred = A @ P @ A.T + q   # 预测状态协方差矩阵

    # 更新步骤
    K = P_pred @ H.T / (H @ P_pred @ H.T + r)   # 卡尔曼增益
    x = x_pred + K * (measurement - H @ x_pred)  # 校正状态
    P = (np.eye(2) - K @ H) @ P_pred            # 校正状态协方差矩阵

    # 输出结果
    print("Measurement:", measurement)
    print("Predicted state:", x_pred)
    print("Updated state:", x)
    print("---------------------------")