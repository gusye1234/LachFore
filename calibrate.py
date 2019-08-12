import pyrealsense2 as rs
import utils as rmsd
import numpy as np
from utils import cv_find_chessboard, get_chessboard_points_3D, get_depth_at_pixel, convert_depth_pixel_to_metric_coordinate, post_process_depth_frame
from utils import convert_pointcloud_to_depth
import world
from realsense_device_manager import DeviceManager
import cv2 as cv


"""
  _   _        _                      _____                     _    _                    
 | | | |  ___ | | _ __    ___  _ __  |  ___|_   _  _ __    ___ | |_ (_)  ___   _ __   ___ 
 | |_| | / _ \| || '_ \  / _ \| '__| | |_  | | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
 |  _  ||  __/| || |_) ||  __/| |    |  _| | |_| || | | || (__ | |_ | || (_) || | | |\__ \
 |_| |_| \___||_|| .__/  \___||_|    |_|    \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
				 _|                                                                      
"""


def calculate_transformation_kabsch(src_points, dst_points):
	"""
	Calculates the optimal rigid transformation from src_points to
	dst_points
	(regarding the least squares error)
	Parameters:
	-----------
	src_points: array
		(3,N) matrix
	dst_points: array
		(3,N) matrix
	
	Returns:
	-----------
	rotation_matrix: array
		(3,3) matrix
	
	translation_vector: array
		(3,1) matrix
	rmsd_value: float
	"""
	assert src_points.shape == dst_points.shape
	if src_points.shape[0] != 3:
		raise Exception("The input data matrix had to be transposed in order to compute transformation.")
		
	src_points = src_points.transpose()
	dst_points = dst_points.transpose()
	
	src_points_centered = src_points - rmsd.centroid(src_points)
	dst_points_centered = dst_points - rmsd.centroid(dst_points)

	rotation_matrix = rmsd.kabsch(src_points_centered, dst_points_centered)
	rmsd_value = rmsd.kabsch_rmsd(src_points_centered, dst_points_centered)

	translation_vector = rmsd.centroid(dst_points) - np.matmul(rmsd.centroid(src_points), rotation_matrix)

	return rotation_matrix.transpose(), translation_vector.transpose(), rmsd_value



"""
  __  __         _           ____               _                _   
 |  \/  |  __ _ (_) _ __    / ___| ___   _ __  | |_  ___  _ __  | |_ 
 | |\/| | / _` || || '_ \  | |    / _ \ | '_ \ | __|/ _ \| '_ \ | __|
 | |  | || (_| || || | | | | |___| (_) || | | || |_|  __/| | | || |_ 
 |_|  |_| \__,_||_||_| |_|  \____|\___/ |_| |_| \__|\___||_| |_| \__|																	 
"""

class Transformation:
	def __init__(self, rotation_matrix, translation_vector, ask_inv = True):
		self.pose_mat = np.zeros((4,4))
		self.pose_mat[:3,:3] = rotation_matrix
		self.pose_mat[:3,3] = translation_vector.flatten()
		self.pose_mat[3,3] = 1
		if ask_inv:
			self.inv = self._inverse()
			self.camera2world = self.inv

	
	def __str__(self):
		return str(self.pose_mat)

	def apply_transformation(self, points):
		""" 
		Applies the transformation to the pointcloud
		
		Parameters:
		-----------
		points : array
			(3, N) matrix where N is the number of points
		
		Returns:
		----------
		points_transformed : array
			(3, N) transformed matrix
		"""
		assert(points.shape[0] == 3)
		n = points.shape[1] 
		points_ = np.vstack((points, np.ones((1,n))))
		points_trans_ = np.matmul(self.pose_mat, points_)
		points_transformed = np.true_divide(points_trans_[:3,:], points_trans_[[-1], :])
		return points_transformed
	
	def _inverse(self):
		"""
		Computes the inverse transformation and returns a new Transformation object
		Returns:
		-----------
		inverse: Transformation
		"""
		rotation_matrix = self.pose_mat[:3,:3]
		translation_vector = self.pose_mat[:3,3]
		
		rot = np.transpose(rotation_matrix)
		trans = - np.matmul(np.transpose(rotation_matrix), translation_vector)
		return Transformation(rot, trans, ask_inv=False)
	
	def XYZ2xy(self, pointcloud, camera_intrinsics):
		assert (pointcloud.shape[0] == 3)
		pointcloud = self.apply_transformation(pointcloud)
		x_ = pointcloud[0,:]
		y_ = pointcloud[1,:]
		z_ = pointcloud[2,:]
		m = x_[np.nonzero(z_)]/z_[np.nonzero(z_)]
		n = y_[np.nonzero(z_)]/z_[np.nonzero(z_)]

		x = m*camera_intrinsics.fx + camera_intrinsics.ppx
		y = n*camera_intrinsics.fy + camera_intrinsics.ppy

		return x, y

	def xy2XYZ(self, xy, depth, camera_intrinsics):
		assert xy.shape[0] == 2
		X, Y, Z = convert_depth_pixel_to_metric_coordinate(depth, xy[0], xy[1], camera_intrinsics)
		point = np.vstack((X,Y))
		point = np.vstack((point, Z))
		point = self.inverse().apply_transformation(point)
		return point


	

class PoseEstimation:

	def __init__(self, frames, intrinsic, chessboard_params):
		assert(len(chessboard_params) == 3)
		self.frames = frames
		self.intrinsic = intrinsic
		self.chessboard_params = chessboard_params		

	def get_chessboard_corners_in3d(self):
		"""
		Searches the chessboard corners in the infrared images of 
		every connected device and uses the information in the 
		corresponding depth image to calculate the 3d 
		coordinates of the chessboard corners in the coordinate system of 
		the camera
		Returns:
		-----------
		corners3D : dict
			keys: str
				Serial number of the device
			values: [success, points3D, validDepths] 
				success: bool
					Indicates wether the operation was successfull
				points3d: array
					(3,N) matrix with the coordinates of the chessboard corners
					in the coordinate system of the camera. N is the number of corners
					in the chessboard. May contain points with invalid depth values
				validDephts: [bool]*
					Sequence with length N indicating which point in points3D has a valid depth value
		"""
		depth_frame = post_process_depth_frame(self.frames[rs.stream.depth])
		depth_intrinsics = self.intrinsic[rs.stream.depth]
		infrared_frame = self.frames[rs.stream.color]
		found_corners, points2D = cv_find_chessboard(depth_frame, infrared_frame, self.chessboard_params)
		if found_corners:
			points3D = np.zeros((3, len(points2D[0])))
			validPoints = [False] * len(points2D[0])
			for index in range(len(points2D[0])):
				corner = points2D[:,index].flatten()
				depth = get_depth_at_pixel(depth_frame, corner[0], corner[1])
				# print(depth)
				if depth != 0 and depth is not None:
					validPoints[index] = True
					[X,Y,Z] = convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1], depth_intrinsics)
					points3D[0, index] = X
					points3D[1, index] = Y
					points3D[2, index] = Z
			return [found_corners, points2D, points3D, validPoints]
		else:
			return [found_corners, None, None, None]


	def perform_pose_estimation(self):
		"""
		Calculates the extrinsic calibration from the coordinate space of the camera to the 
		coordinate space spanned by a chessboard by retrieving the 3d coordinates of the 
		chessboard with the depth information and subsequently using the kabsch algortihm 
		for finding the optimal rigid transformation between the two coordinate spaces
		Returns:
		-----------
		retval : dict
		keys: str
			Serial number of the device
		values: [success, transformation, points2D, rmsd]
			success: bool
			transformation: Transformation
				Rigid transformation from the coordinate system of the camera to 
				the coordinate system of the chessboard
			points2D: array
				[2,N] array of the chessboard corners used for pose_estimation
			rmsd:
				Root mean square deviation between the observed chessboard corners and 
				the corners in the local coordinate system after transformation
		"""
		corners3D = self.get_chessboard_corners_in3d()
		[found_corners, points2D, points3D, validPoints] = corners3D
		objectpoints = get_chessboard_points_3D(self.chessboard_params)
		if found_corners == True:
			#initial vectors are just for correct dimension
			valid_object_points = objectpoints[:,validPoints]
			valid_observed_object_points = points3D[:,validPoints]
			
			#check for sufficient points
			if valid_object_points.shape[1] < 5:
				print("Not enough points have a valid depth for calculating the transformation")
			else:
				[rotation_matrix, translation_vector, rmsd_value] = calculate_transformation_kabsch(valid_object_points, valid_observed_object_points)
				# retval[serial] =[True, Transformation(rotation_matrix, translation_vector), points2D, rmsd_value]
				print("RMS error for calibration with device number", 1 , "is :", rmsd_value, "m")
				return [True, Transformation(rotation_matrix, translation_vector), points2D, rmsd_value]
		return [False, None, None, None]


	def find_chessboard_boundary_for_depth_image(self):

		depth_frame = post_process_depth_frame(self.frame[rs.steam.depth])
		infrared_frame = self.frames[rs.steam.color]
		_, points2D = cv_find_chessboard(depth_frame, infrared_frame, self.chessboard_params)
		boundary = [np.floor(np.amin(points2D[0,:])).astype(int), np.floor(np.amax(points2D[0,:])).astype(int), np.floor(np.amin(points2D[1,:])).astype(int), np.floor(np.amax(points2D[1,:])).astype(int)]

		return boundary


class CALIBRATION:
	def __init__(self):
		self.axis = None
		self.xyAxis = None

	def draw_axis(self, img):
		if self.xyAxis is not None:
			axises = ["x", "y", "z"]
			color = [(255,0,0), (0,255,0), (0,0,255)]
			x, y = self.xyAxis[0], self.xyAxis[1]
			for i in range(3):
				cv.line(img, (x[0], y[0]), (x[i+1], y[i+1]), color[i], 3)
				cv.putText(img, axises[i], (x[i+1], y[i+1]), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0),2)
			return img


	def run_calibration(self):
		try:
			while True:
				if world.device_manager is None:
					device_manager = DeviceManager()
				else:
					device_manager = world.device_manager
				device_manager.enable_device()
				print(">CALIBRATION STARTING")
				for i in range(world.stablisation):
					frames = device_manager.poll_frames()
				assert (device_manager._enabled_devices is not None)
				intrinsic =  device_manager.get_device_intrinsics(frames)
				# print(type(intrinsic),intrinsic)
				calibrated = False
				cv.namedWindow("CALIBRATE", cv.WINDOW_AUTOSIZE)
				print(">SETTING IMAGE")
				while True:
					frames = device_manager.poll_frames()
					img = np.asanyarray(frames[rs.stream.color].get_data())
					gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
					found, corners = cv.findChessboardCorners(gray, (world.chessboard_width, world.chessboard_height))
					if found:
						cv.drawChessboardCorners(img, (world.chessboard_width, world.chessboard_height), corners, found)
					cv.imshow("CALIBRATE", img)
					key = cv.waitKey(1)
					if key == ord('q'):
						break
				while calibrated == False:
					frames = device_manager.poll_frames()
					pose_estimator = PoseEstimation(frames, intrinsic, world.chessboard_params)
					result = pose_estimator.perform_pose_estimation()
					object_point = pose_estimator.get_chessboard_corners_in3d()
					if not result[0]:
						print("Place the chessboard on the plane where the object needs to be detected..")
					else:
						calibrated = True
					img = np.asanyarray(frames[rs.stream.color].get_data())
					cv.imshow("CALIBRATE", img)
					key = cv.waitKey(1)
					if key == ord('q'):
						device_manager.disable_streams()
						cv.destroyAllWindows()
						return
				trans = {}
				if world.calibrate_debug:
					print("matrix is: \n", result[1])
				trans = result[1]
				points3d = np.array([[0.0,0.3,0,0],[0.0,0,0.3,0],[0.0,0,0,-0.1]], dtype="float32")
				if world.calibrate_debug:
					print("world axis is:")
					print(points3d)
				points3d = trans.apply_transformation(points3d)
				x,y = convert_pointcloud_to_depth(points3d, intrinsic[rs.stream.depth])
				if world.calibrate_debug:
					print("camera axis is")
					print(x,y)
					print("Image axis is:")
				x, y = x.astype("int32"), y.astype("int32")
				if world.calibrate_debug:
					print(x,'\n',y)
					print(object_point[2][:, object_point[3]][:, :10])
					print("Chess corners is(in camera):")
					print(trans.inv.apply_transformation(
					object_point[2][:, object_point[3]]).T[:10])
				#plot axises
				while True:
					color = [(255,0,0), (0,255,0), (0,0,255)]
					axises = ["x", "y", "z"]
					frames = device_manager.poll_frames()
					img = np.asanyarray(frames[rs.stream.color].get_data())
					for i in range(3):
						cv.line(img, (x[0], y[0]), (x[i+1], y[i+1]), color[i], 2)
						cv.putText(img, axises[i], (x[i+1], y[i+1]), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0),2)
					cv.imshow("CALIBRATE", img)
					key = cv.waitKey(1)
					if key == ord('q'):
						print("Calibration completed... \nPlace stuffs in the field of view of the devices...")
						world.world2camera = trans
						self.xyAxis = np.vstack((x,y))
						return
					elif key == ord('r'):
						break
					elif key == ord('t'):
						cv.imwrite("./photos/calibrate_"+world.now, img)
		


			
		finally:
			device_manager.disable_streams()
			cv.destroyAllWindows()

if __name__ == "__main__":
	a = CALIBRATION()
	a.run_calibration()