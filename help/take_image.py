import pyrealsense2 as rs
import cv2 as cv
import numpy as np  
import sys
import time

try:
    name = sys.argv[1]
except IndexError:
    name = "test"



pip = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pip.start(config)
try:
    count = 1
    while True:
        frames = pip.wait_for_frames()

        color_frame = frames.get_color_frame()

        img = np.asanyarray(color_frame.get_data())

        cv.namedWindow("take_photo", cv.WINDOW_AUTOSIZE)
        cv.imshow("take_photo", img)
        key = cv.waitKey(1)

        if key == ord("t"):
            # cv.destroyAllWindows()
            cv.imwrite("photos/photo_%s_%s.png" % (name,count), img)
            count += 1
        elif key == ord("q"):
            cv.destroyAllWindows()
            break

        time.sleep(0.001)

finally:
    pip.stop()