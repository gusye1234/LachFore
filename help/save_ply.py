## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##                  Export to PLY                  ##
#####################################################

# First import the library
import pyrealsense2 as rs
import numpy as np

# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()
# print("shape: ", points.shape)

# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#Start streaming with default recommended configuration
pipe.start(config);

try:
    # Wait for the next set of frames from the camera
    while True:
        frames = pipe.wait_for_frames()

        # Fetch color and depth frames
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()

        # Tell pointcloud object to map to this color frame
        pc.map_to(color)
        print(np.asanyarray(depth.get_data()).shape)
        # Generate the pointcloud and texture mappings
        points = pc.calculate(depth)
        k = np.asanyarray(points.get_data())
        print(k)
        if len(k) == 0:
            continue
        else:
            break
    print("Saving to 1.ply...")
    points.export_to_ply("1.ply", color)
    print("Done")
finally:
    pipe.stop()