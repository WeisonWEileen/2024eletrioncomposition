import serial  
import struct  
from time import sleep  
  
def init_serial():  
    ser = serial.Serial('/dev/ttyUSB0', 115200)  
    if not ser.isOpen():  
        ser.open()  
    return ser  
  
def send_data(ser, data):  
    # 假设数据格式是：0x10 + 数据字节 + 0x0A  
    packed_data = struct.pack('BBB', 0x0A, data, 0x0B)  
    ser.write(packed_data)  
    ser.flush()  
    print(f"send data: {packed_data.hex()}")  
  
def read_data(ser):  
    # 读取直到遇到期望的帧头（这里假设为0x10）  
    while True:  
        if ser.in_waiting > 0:  
            byte = ser.read(1)  
            if byte == b'\x0A':  # 帧头  
                data = ser.read(1)  # 读取数据字节  
                end_byte = ser.read(1)  # 读取结束字节  
                if end_byte == b'\x0B':  
                    return data[0]  
                else:  
                    print("Invalid frame end")  
  
def main():  
    ser = init_serial()  
    try:  
        while True:  
            # 假设我们发送一个示例数据 0x42  
            send_data(ser, 0x42)  
  
            # 读取数据  
            received_data = read_data(ser)  
            if received_data:  
                print(f"Received data: {received_data:02X}")  
  
            sleep(1)  # 等待一秒后再次发送  
  
    except KeyboardInterrupt:  
        print("Program interrupted by user")  
    finally:  
        ser.close()  
  
if __name__ == "__main__":  
    main()