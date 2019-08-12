import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2 

import time
# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale


pc = rs.pointcloud()


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        
        # Align the depth frame to color frame
        # aligned_frames = align.process(frames)
        aligned_frames = frames
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # print("depth", depth_image.shape)
        color_image = np.asanyarray(color_frame.get_data())
        # color_image[:50, :265, :] = 255
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        center = depth_image[80:400, 80:560]
        print("mean depth is:", np.mean(center), np.std(center))
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        
        # Render images
        points = pc.calculate(aligned_depth_frame)
        mask = np.abs(depth_image - np.mean(depth_image)) > np.std(depth_image)
        # depth_image = mask + depth_image
        # depth_image[:20, :265] = 2000
        if abs(np.mean(center) - 1100) < 20 and np.std(center) < 8:
            cv2.imwrite("ok.png", color_image)
            cv2.imwrite("ok_depth.png", np.dstack((depth_image,depth_image,depth_image)))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            a = cv2.Canny(color_image, 100, 200)
            cv2.imwrite("ok2.png", a)
            break
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        time.sleep(0.01)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            cv2.imwrite("ok.png", color_image)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            a = cv2.Canny(color_image, 100, 200)
            cv2.imwrite("ok2.png", a)
            break
finally:
    pipeline.stop()