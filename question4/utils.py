# 用于得到目标位置的序列号,发送到stm32,让其移动到目标位置
def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]