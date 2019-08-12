"""
WORLD
storing the global setting to contorl:
1.print
2.parameters
"""
import time
import cv2 as cv

__version__ = "1.0"
#environment#####################################
resolution_width = 1280
resolution_height = 720
depth_width = 1280
depth_height = 720
frame_rate = 30
stablisation = 30
#chessboard fot calibrations#####################
chessboard_size = 0.0243
chessboard_width = 8
chessboard_height = 6
chessboard_params = [chessboard_height, chessboard_width, chessboard_size]

##################################################
qrcode_debug = False
qrcode_num = 4
qrcode_verbose = False
##################################################
world2camera = None
device_manager = None
calibrate_debug = False
##################################################
now = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
#opencv window####################################
cv.namedWindow("world", cv.WINDOW_AUTOSIZE)
# cv.namedWindow("debug", cv.WINDOW_AUTOSIZE)
#model############################################
model_verbose = False
bestCnt = 10
conf_thresh = 0.3
use_gpu = False