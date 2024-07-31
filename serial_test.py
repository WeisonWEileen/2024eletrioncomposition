import serial  
import time  
  
# 串口配置  
ser = serial.Serial(  
    # port='/dev/ttyUSB0',  # 串口名称，根据你的设备修改  
    port='/dev/ttyS0',  # 串口名称，根据你的设备修改  
    baudrate=115200,       # 波特率  
    parity=serial.PARITY_NONE, # 校验位  
    stopbits=serial.STOPBITS_ONE, # 停止位  
    bytesize=serial.EIGHTBITS,   # 数据位  
    timeout=1              # 读取超时设置  
)  
  
try:  
    while True:
    # 发送数字0到9  
    # for num in range(10):  
        # 将数字转换为字节串发送  
        # 注意：发送的是字节，所以需要将数字转换为字节类型  
        # 例如，使用str(num).encode()将数字转换为字符串，然后编码为字节  
        ser.write(str(19).encode())  # 发送数字  
        ser.write(b'\n')  # 可选：发送换行符作为分隔  
        time.sleep(1)  # 等待一秒  
  
finally:  
    ser.close()  # 关闭串口  
  
print("发送完成，串口已关闭。")