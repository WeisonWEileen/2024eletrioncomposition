import os
import numpy as np

os.system('')

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
    
phase = np.array([
    [1, 0, 0],
    [0, 1, 2],
    [0, 0, 0]
], dtype=np.ubyte)

show_phase(phase)