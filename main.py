import world
from qrcode import QRCODE
from calibrate import CALIBRATION
from predict import Seg
from utils import getDevice
import pyrealsense2 as rs
import cv2 as cv
import numpy as np

def main():
    device = getDevice()
    device.enable_device()
    for i in range(world.stablisation):
        frame = device.poll_frames()
    color_intrinsics = device.get_device_intrinsics(frame)[rs.stream.color]
    depth_intrinsics = device.get_device_intrinsics(frame)[rs.stream.depth]
    print(">RGB:", color_intrinsics.width, color_intrinsics.height)
    print(">DEPTH:", depth_intrinsics.width, depth_intrinsics.height)
    calibrator = CALIBRATION()
    calibrator.run_calibration()
    qrcode = QRCODE(depth_intrinsics.ppx, depth_intrinsics.ppy, depth_intrinsics.fx, depth_intrinsics.fy)
    seg = Seg("ycb", color_intrinsics)
    # seg = Seg("linemod", color_intrinsics)
    flag = False
    already = False
    while True:
        frame = device.poll_frames()
        color = frame[rs.stream.color]
        depth = frame[rs.stream.depth]
        img = np.asanyarray(color.get_data())
        img_axis = np.copy(img)
        img_axis = calibrator.draw_axis(img_axis)
        img_qr = np.copy(img)
        if flag == False:
            img_pred = np.copy(img)
            img_pred = cv.rectangle(img_pred, (320, 120), (960, 600), (0,0,0), 2)
        else:
            if already == False:
                pred, img_test = seg.predict(img_pred[120:600, 320:960], draw=True)
                img_pred[120:600, 320:960] = img_test
                already = True
                if len(pred) == 0:
                    print("No Object")
                    continue
                # print(pred)
                # a = input(">WHICH: %s \n" % ([seg.names[a[0]] for a in pred]))
                # pos = pred[0][1][:, 3].reshape((-1,1))
                # out = np.matmul(seg.intrinsics, pos)/pos[2,0]
                # x, y = int(out[0,0]), int(out[1,0])
                # img_pred[y-5:y+5, x-5:x+5] = 0
            else:
                pass
        if qrcode.update(img, depth, frame[rs.stream.depth]):
            img_qr = qrcode.draw(img_qr)
        all_img = np.vstack((np.hstack((img_axis, img_qr)), np.hstack((img_pred,img) )))
        all_img = cv.resize(all_img,(1280,720))
        cv.imshow("world", all_img)
        key = cv.waitKey(1)
        if key == ord('q'):
            cv.destroyAllWindows()
            return
        elif key == ord('p'):
            flag = True if flag == False else False
            already = False
        elif key == ord('t'):
            cv.imwrite("./photos/main_%s.png" % (world.now), all_img)
        del img_axis, img, img_qr
    # img = cv.imread("./segmentation_driven_pose/data/OcclusionChallengeICCV2015/RGB-D/rgb_noseg/color_00000.png")
    # print(img.shape)
    # pred, img = seg.predict(img, draw=True)
    # cv.imshow("debug", img)
    # cv.waitKey(0)

if __name__ == "__main__":
    main()