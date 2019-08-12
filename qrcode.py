"""
QRCODE:
using Kalman filter to locate the qrcode corner points.
"""

import pyrealsense2 as rs
import cv2 as cv
import numpy as np  
import sys
import time
from pyzbar import pyzbar
from pykalman import KalmanFilter
from numpy import ma
import world
from pprint import pprint
from utils import post_process_depth_frame, getDevice
from realsense_device_manager import DeviceManager


try:
    name = sys.argv[1]
except IndexError:
    name = "test"



class QRCODE:
    def __init__(self, ppx, ppy, fx, fy, qrnum=4, tracking_code = None):
        """ 
        need to use camera intrinsics to setup
        """
        self.qrnum = qrnum
        self.index = [i for i in range(1,qrnum+1)]
        if tracking_code is None:
            self.tracking_code = self.index
        else:
            try:
                tracking_code = tuple(tracking_code)
            except TypeError:
                raise TypeError("tracking code list should be converted to tuple")
            self.tracking_code = tracking_code
        self.QR_position = {}
        self.QR_cov = {}
        self.QR_now = {}
        self.tracking = {}
        self.kf = {}
        self.color = (255,0, 0)
        self.ppx = ppx
        self.ppy = ppy
        self.fx = fx
        self.fy = fy
        self.height = world.resolution_height
        self.width = world.resolution_width
        print(">SCANING START")

    def update(self, image, depth, frames):
        """
        update qrcode states \n
        @parameter: image(ndarray), depth(ndarray), frames(depth_frames)
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        codes = pyzbar.decode(gray)
        if world.qrcode_debug:
            print(codes)
        self.QR_now = {}
        if len(codes) == 0:
            return False
        for code in codes:
            index = int(code.data.decode("utf8"))
            xyz = self.getXYZ(code, depth, depth_frame=frames)
            if world.qrcode_debug:
                print("getxyz:", xyz)
            # if np.isnan(xyz).any():
            #     continue
            # else:
            #     self.QR_now[index] = xyz
            self.QR_now[index] = xyz
            if world.qrcode_verbose:
                print(self.QR_now[index])
        for i in self.QR_now:
            if self.QR_position.get(i) is None: #or np.isnan(self.QR_position[i]).any():
                self.QR_position[i] = self.QR_now[i]
                self.QR_cov[i] = np.eye(12, dtype=np.float32)
                self.kf[i] = KalmanFilter(transition_matrices=np.eye(12, dtype=np.float32),
                                        observation_matrices = np.eye(12, dtype=np.float32),
                                        transition_covariance= 0.03*np.eye(12))
                print(">CONNECT TO NO.%d !!!!!" % (i))
                if i in self.tracking_code:
                    print(">>TRACKING")
                    self.tracking[index] = self.QR_now[i]
            else:
                if i in self.tracking_code:
                    self.tracking[index] = self.QR_position[i]
                self.QR_position[i], self.QR_cov[i] = self.kf[i].filter_update(self.QR_position[i], self.QR_cov[i], self.QR_now[i])
        if world.qrcode_verbose:
            pprint(self.QR_position)
        return True

    def draw(self, img):
        for i in self.QR_now:
            try:
                for j in range(0, 12, 3):
                    # print(self.QR_position[i][j:j+3])
                    x, y = self.back_pixel(*self.QR_position[i][j:j+3])
                    x, y = int(x), int(y)
                    x, y = x if x<self.width else self.width, y if y < self.height else self.height
                    cv.circle(img, (x,y), 3, self.color, 2)
                X,Y,Z = QRCODE.center(self.QR_position[i])
                x,y = self.back_pixel(X,Y,Z)
                x,y = int(x), int(y)
                cv.circle(img, (x,y), 5, self.color, 2)
                cv.putText(img, "%d (%.3f)" % (i, Z), (x-20,y),  cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),2)
            except OverflowError:
                print("qrcode.py: encounter error in QRCODE.draw")
            finally:
                pass
        return img

    def getPosition(self): 
        return self.QR_position

    def getLatestPosition(self):
        return self.QR_now
    
    def getCenter(self, code_id):
        if code_id in self.QR_position:
            return QRCODE.center(self.QR_position[code_id])
        else:
            return None

    def converge(self, threshold = 1.0, code=None):
        if code is None:
            errors = []
            for i in self.tracking_code:
                error = np.mean(np.abs(self.QR_position[i] - self.tracking[i]))
                errors.append(error)
            return (np.mean(errors) < threshold)
        else:
            error = np.mean(np.abs(self.QR_position[code] - self.tracking[code]))
            return error < threshold


    @staticmethod
    def center(pos):
        midx = (pos[0]+pos[3]+pos[6]+pos[9])/4
        midy = (pos[1]+pos[4]+pos[7]+pos[10])/4
        midz = (pos[2]+pos[5]+pos[8]+pos[11])/4
        return midx, midy, midz

    def cal_point(self, x,y,Z):
        X = (x-self.ppx)*Z/self.fx
        Y = (y-self.ppy)*Z/self.fy
        return X,Y,Z

    def back_pixel(self, X, Y, Z):
        if abs(Z - 0) <1e-6:
            # print("z = 0.0!!")
            return 0.0, 0.0
        x = self.fx/Z*X + self.ppx
        y = self.fy/Z*Y + self.ppy
        return x,y

    def getXYZ(self, code, depth, depth_frame = None,kernal=5):
        xyz = []
        missing = []
        for i, point in enumerate(code.polygon):
            try:
                if depth_frame is None:
                    z = np.mean(depth[point.x-kernal:point.x+kernal, point.y-kernal:point.y+kernal])
                else:
                    z = depth_frame.as_depth_frame().get_distance(point.x, point.y)
            except:
                    z = np.nan
            if np.isnan(z):
                xyz.extend([point[0], point[1], 0.0])
                missing.extend([i+2])
                if world.qrcode_verbose:
                    print("z is nan,replace by 0.0")
                continue
            xyz.extend(self.cal_point(point.x, point.y, z))
        xyz = np.array(xyz)
        if len(missing):
            xyz[missing] = ma.masked
        return xyz

if __name__ == "__main__":

    device_manager = getDevice()
    device_manager.enable_device()
    frames = device_manager.poll_frames()
    intrinsics = device_manager.get_device_intrinsics(frames)

    # pip = rs.pipeline()

    # config = rs.config()
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)


    # profile = pip.start(config)

    # align_to = rs.stream.color
    # align = rs.align(align_to)

    depth_intrinsics = intrinsics[rs.stream.depth]
    ppx, ppy, fx, fy = depth_intrinsics.ppx, depth_intrinsics.ppy, depth_intrinsics.fx, depth_intrinsics.fy

    try:
        count = 1
        qr = QRCODE(ppx,ppy, fx, fy)
        while True:
            frames = device_manager.poll_frames()
            # frames = align.process(frames)


            color_frame = frames[rs.stream.color]
            depth_frame = frames[rs.stream.depth]
            depth_frame = post_process_depth_frame(depth_frame)

            if not color_frame or not depth_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            # print("main is:", np.mean(depth))
            if qr.update(img, depth, depth_frame):
                # print("got")
                img = qr.draw(img)
                # codo_shape = [int(code.data.decode("utf8")):]
                # for i in codes:
                #     rect = codes[i].rect
                #     cv.rectangle(img, (rect.left, rect.top), 
                #                 (rect.left+rect.width, rect.top+rect.height), color, 1)
                #     cv.putText(img, str(i) + "(%.1f,%.1f,%.1f)" % codes[i], (rect.left, rect.top),  cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0),1)                        
                    # print(code.data.decode("utf8") + "  " + str(distance))
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

            time.sleep(0.01)

    finally:
        device_manager.disable_streams()