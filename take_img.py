import time

import cv2 as cv
import numpy as np
import os
import keyboard

ttv = 0

cam = cv.VideoCapture(0)

if __name__ == "__main__":
    while True:
        _, frame = cam.read()
        frame = cv.resize(frame, (640, 640))
        cv.imshow('windows', frame)
        cv.waitKey(1)

        if keyboard.is_pressed('t'):
            cv.imwrite(f'img/{ttv}.jpg', frame)
            ttv += 1
            print(f'Taken Images {ttv}')
            time.sleep(2)

        if cv.waitKey(1) == ord('q'):
            print(f'Taken Images {ttv} !!! action break')
            break
