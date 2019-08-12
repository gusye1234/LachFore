##################################################################################################
##       License: Apache 2.0. See LICENSE file in root directory.                             ####
##################################################################################################
##                  Box Dimensioner with multiple cameras: Helper files                       ####
##################################################################################################
import pyrealsense2 as rs
import numpy as np
import world

"""
  _   _        _                      _____                     _    _                    
 | | | |  ___ | | _ __    ___  _ __  |  ___|_   _  _ __    ___ | |_ (_)  ___   _ __   ___ 
 | |_| | / _ \| || '_ \  / _ \| '__| | |_  | | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
 |  _  ||  __/| || |_) ||  __/| |    |  _| | |_| || | | || (__ | |_ | || (_) || | | |\__ \
 |_| |_| \___||_|| .__/  \___||_|    |_|    \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
                 |_|                                                                      
"""


class Device:
    def __init__(self, pipeline, pipeline_profile):
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile


def enumerate_connected_devices(context):
    """
    Enumerate the connected Intel RealSense devices
    Parameters:
    -----------
    context 	   : rs.context()
                     The context created for using the realsense library
    Return:
    -----------
    connect_device : array
                     Array of enumerated devices which are connected to the PC
    """
    connect_device = []
    for d in context.devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device.append(d.get_info(rs.camera_info.serial_number))
    return connect_device


"""
  __  __         _           ____               _                _   
 |  \/  |  __ _ (_) _ __    / ___| ___   _ __  | |_  ___  _ __  | |_ 
 | |\/| | / _` || || '_ \  | |    / _ \ | '_ \ | __|/ _ \| '_ \ | __|
 | |  | || (_| || || | | | | |___| (_) || | | || |_|  __/| | | || |_ 
 |_|  |_| \__,_||_||_| |_|  \____|\___/ |_| |_| \__|\___||_| |_| \__|
"""


class DeviceManager:
    def __init__(self, pipeline_configuration = None):
        """
        Class to manage the Intel RealSense devices
        Parameters:
        -----------
        context                 : rs.context()
                                  The context created for using the realsense library
        pipeline_configuration  : rs.config()
                                  The realsense library configuration to be used for the application
        """
        # assert isinstance(context, type(rs.context()))
        if pipeline_configuration is None:
            config = rs.config()
            config.enable_stream(rs.stream.depth, world.resolution_width, world.resolution_height, rs.format.z16, world.frame_rate)
            config.enable_stream(rs.stream.color, world.resolution_width, world.resolution_height, rs.format.bgr8, world.frame_rate)
            pipeline_configuration = config
        assert isinstance(pipeline_configuration, type(rs.config()))
        self._enabled_devices = None
        self.align = None
        self._config = pipeline_configuration
        self._frame_counter = 0

    def enable_device(self, enable_ir_emitter=True):
        """
        Enable an Intel RealSense Device
        Parameters:
        -----------
        device_serial     : string
                            Serial number of the realsense device
        enable_ir_emitter : bool
                            Enable/Disable the IR-Emitter of the device
        """
        if self._enabled_devices is not None:
            return
        pipeline = rs.pipeline()

        # Enable the device
        if world.resolution_width != world.depth_width:
            align_to = rs.stream.color
            self.align = rs.align(align_to)
        pipeline_profile = pipeline.start(self._config)

        # Set the acquisition parameters
        sensor = pipeline_profile.get_device().first_depth_sensor()
        sensor.set_option(rs.option.emitter_enabled, 1 if enable_ir_emitter else 0)
        self._enabled_devices = Device(pipeline, pipeline_profile)
        if self._enabled_devices is not None:
            print(">CAMERA start")

    def enable_emitter(self, enable_ir_emitter=True):
        """
        Enable/Disable the emitter of the intel realsense device
        """
        # for (device_serial, device) in self._enabled_devices.items():
            # Get the active profile and enable the emitter for all the connected devices
        sensor = self._enabled_devices.pipeline_profile.get_device().first_depth_sensor()
        sensor.set_option(rs.option.emitter_enabled, 1 if enable_ir_emitter else 0)
        if enable_ir_emitter:
            sensor.set_option(rs.option.laser_power, 330)

    def poll_frames(self):
        """
        Poll for frames from the enabled Intel RealSense devices.
        If temporal post processing is enabled, the depth stream is averaged over a certain amount of frames
        Parameters:
        -----------
        """
        device = self._enabled_devices
        frames  ={}
        frame = device.pipeline.wait_for_frames()
        if self.align is not None:
            frame = self.align.process(frame)
        frames[rs.stream.color] = frame.get_color_frame()
        frames[rs.stream.depth] = frame.get_depth_frame()
        # print(np.asanyarray(frame.get_depth_frame().get_data())[100:105, 100:105])
        return frames

    def get_depth_shape(self):
        """ 
        Retruns width and height of the depth stream for one arbitrary device
        Returns:
        -----------
        width : int
        height: int
        """
        width = -1
        height = -1
        device = self._enabled_devices
        for stream in device.pipeline_profile.get_streams():
            if (rs.stream.depth == stream.stream_type()):
                width = stream.as_video_stream_profile().width()
                height = stream.as_video_stream_profile().height()

        return width, height

    def get_device_intrinsics(self, frames):
        """
        Get the intrinsics of the imager using its frame delivered by the realsense device
        Parameters:
        -----------
        frames : rs::frame
                 The frame grabbed from the imager inside the Intel RealSense for which the intrinsic is needed
        Return:
        -----------
        device_intrinsics : dict
        keys  : serial
                Serial number of the device
        values: [key]
                Intrinsics of the corresponding device
        """
        device_intrinsics = {}
        frameset = frames
        for key, value in frameset.items():
            device_intrinsics[key] = value.get_profile().as_video_stream_profile().get_intrinsics()
        return device_intrinsics

    def get_depth_to_color_extrinsics(self, frames):
        """
        Get the extrinsics between the depth imager 1 and the color imager using its frame delivered by the realsense device
        Parameters:
        -----------
        frames : rs::frame
                 The frame grabbed from the imager inside the Intel RealSense for which the intrinsic is needed
        Return:
        -----------
        device_intrinsics : dict
        keys  : serial
                Serial number of the device
        values: [key]
                Extrinsics of the corresponding device
        """
        device_extrinsics = {}
        frameset = frames
        device_extrinsics = frameset[
            rs.stream.depth].get_profile().as_video_stream_profile().get_extrinsics_to(
            frameset[rs.stream.color].get_profile())
        return device_extrinsics

    def disable_streams(self):
        self._config.disable_all_streams()


"""
  _____           _    _               
 |_   _|___  ___ | |_ (_) _ __    __ _ 
   | | / _ \/ __|| __|| || '_ \  / _` |
   | ||  __/\__ \| |_ | || | | || (_| |
   |_| \___||___/ \__||_||_| |_| \__, |
                                  |___/ 
"""
if __name__ == "__main__":
    try:
        c = rs.config()
        c.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        # c.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 6)
        # c.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 6)
        c.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 6)
        device_manager = DeviceManager(c)
        device_manager.enable_device()
        for k in range(150):
            frames = device_manager.poll_frames()
        device_manager.enable_emitter(True)
        device_extrinsics = device_manager.get_depth_to_color_extrinsics(frames)
    finally:
        device_manager.disable_streams()