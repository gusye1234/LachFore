import cv2 as cv
import numpy as np
import re
import pyrealsense2 as rs
import world
from realsense_device_manager import DeviceManager


__version__ = world.__version__



def calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2d, depth_threshold = 0.01):
	"""
 Calculate the cumulative pointcloud from the multiple devices
	Parameters:
	-----------
	frames_devices : dict
		The frames from the different devices
		keys: str
			Serial number of the device
		values: [frame]
			frame: rs.frame()
				The frameset obtained over the active pipeline from the realsense device
				
	calibration_info_devices : dict
		keys: str
			Serial number of the device
		values: [transformation_devices, intrinsics_devices]
			transformation_devices: Transformation object
					The transformation object containing the transformation information between the device and the world coordinate systems
			intrinsics_devices: rs.intrinscs
					The intrinsics of the depth_frame of the realsense device
					
	roi_2d : array
		The region of interest given in the following order [minX, maxX, minY, maxY]
		
	depth_threshold : double
		The threshold for the depth value (meters) in world-coordinates beyond which the point cloud information will not be used.
		Following the right-hand coordinate system, if the object is placed on the chessboard plane, the height of the object will increase along the negative Z-axis
	
	Return:
	----------
	point_cloud_cumulative : array
		The cumulative pointcloud from the multiple devices
	"""
	# Use a threshold of 5 centimeters from the chessboard as the area where useful points are found
	point_cloud_cumulative = np.array([-1, -1, -1]).transpose()
	for (device, frame) in frames_devices.items() :
		# Filter the depth_frame using the Temporal filter and get the corresponding pointcloud for each frame
		filtered_depth_frame = post_process_depth_frame(frame[rs.stream.depth], temporal_smooth_alpha=0.1, temporal_smooth_delta=80)	
		point_cloud = convert_depth_frame_to_pointcloud( np.asarray( filtered_depth_frame.get_data()), calibration_info_devices[device][1][rs.stream.depth])
		point_cloud = np.asanyarray(point_cloud)

		# Get the point cloud in the world-coordinates using the transformation
		point_cloud = calibration_info_devices[device][0].apply_transformation(point_cloud)

		# Filter the point cloud based on the depth of the object
		# The object placed has its height in the negative direction of z-axis due to the right-hand coordinate system
		point_cloud = get_clipped_pointcloud(point_cloud, roi_2d)
		point_cloud = point_cloud[:,point_cloud[2,:]<-depth_threshold]
		point_cloud_cumulative = np.column_stack( ( point_cloud_cumulative, point_cloud ) )
	point_cloud_cumulative = np.delete(point_cloud_cumulative, 0, 1)
	return point_cloud_cumulative




def calculate_boundingbox_points(point_cloud, calibration_info_devices, depth_threshold = 0.01):
	"""
	Calculate the top and bottom bounding box corner points for the point cloud in the image coordinates of the color imager of the realsense device
	
	Parameters:
	-----------
	point_cloud : ndarray
		The (3 x N) array containing the pointcloud information
		
	calibration_info_devices : dict
		keys: str
			Serial number of the device
		values: [transformation_devices, intrinsics_devices, extrinsics_devices]
			transformation_devices: Transformation object
					The transformation object containing the transformation information between the device and the world coordinate systems
			intrinsics_devices: rs.intrinscs
					The intrinsics of the depth_frame of the realsense device
			extrinsics_devices: rs.extrinsics
					The extrinsics between the depth imager 1 and the color imager of the realsense device
					
	depth_threshold : double
		The threshold for the depth value (meters) in world-coordinates beyond which the point cloud information will not be used
		Following the right-hand coordinate system, if the object is placed on the chessboard plane, the height of the object will increase along the negative Z-axis
		
	Return:
	----------
	bounding_box_points_color_image : dict
		The bounding box corner points in the image coordinate system for the color imager
		keys: str
				Serial number of the device
			values: [points]
				points: list
					The (8x2) list of the upper corner points stacked above the lower corner points 
					
	length : double
		The length of the bounding box calculated in the world coordinates of the pointcloud
		
	width : double
		The width of the bounding box calculated in the world coordinates of the pointcloud
		
	height : double
		The height of the bounding box calculated in the world coordinates of the pointcloud
	"""
	# Calculate the dimensions of the filtered and summed up point cloud
	# Some dirty array manipulations are gonna follow
	if point_cloud.shape[1] > 500:
		# Get the bounding box in 2D using the X and Y coordinates
		coord = np.c_[point_cloud[0,:], point_cloud[1,:]].astype('float32')
		min_area_rectangle = cv.minAreaRect(coord)
		bounding_box_world_2d = cv.boxPoints(min_area_rectangle)
		# Caculate the height of the pointcloud
		height = max(point_cloud[2,:]) - min(point_cloud[2,:]) + depth_threshold

		# Get the upper and lower bounding box corner points in 3D
		height_array = np.array([[-height], [-height], [-height], [-height], [0], [0], [0], [0]])
		bounding_box_world_3d = np.column_stack((np.row_stack((bounding_box_world_2d,bounding_box_world_2d)), height_array))

		# Get the bounding box points in the image coordinates
		bounding_box_points_color_image={}
		for (device, calibration_info) in calibration_info_devices.items():
			# Transform the bounding box corner points to the device coordinates
			bounding_box_device_3d = calibration_info[0].inverse().apply_transformation(bounding_box_world_3d.transpose())
			
			# Obtain the image coordinates in the color imager using the bounding box 3D corner points in the device coordinates
			color_pixel=[]
			bounding_box_device_3d = bounding_box_device_3d.transpose().tolist()
			for bounding_box_point in bounding_box_device_3d: 
				bounding_box_color_image_point = rs.rs2_transform_point_to_point(calibration_info[2], bounding_box_point)			
				color_pixel.append(rs.rs2_project_point_to_pixel(calibration_info[1][rs.stream.color], bounding_box_color_image_point))
			
			bounding_box_points_color_image[device] = np.row_stack( color_pixel )
		return bounding_box_points_color_image, min_area_rectangle[1][0], min_area_rectangle[1][1], height
	else : 
		return {},0,0,0



def visualise_measurements(frames_devices, bounding_box_points_devices, length, width, height):
	"""
 Calculate the cumulative pointcloud from the multiple devices
	
	Parameters:
	-----------
	frames_devices : dict
		The frames from the different devices
		keys: str
			Serial number of the device
		values: [frame]
			frame: rs.frame()
				The frameset obtained over the active pipeline from the realsense device
				
	bounding_box_points_color_image : dict
		The bounding box corner points in the image coordinate system for the color imager
		keys: str
				Serial number of the device
			values: [points]
				points: list
					The (8x2) list of the upper corner points stacked above the lower corner points 
					
	length : double
		The length of the bounding box calculated in the world coordinates of the pointcloud
		
	width : double
		The width of the bounding box calculated in the world coordinates of the pointcloud
		
	height : double
		The height of the bounding box calculated in the world coordinates of the pointcloud
	"""
	for (device, frame) in frames_devices.items():
		color_image = np.asarray(frame[rs.stream.color].get_data())
		if (length != 0 and width !=0 and height != 0):
			bounding_box_points_device_upper = bounding_box_points_devices[device][0:4,:]
			bounding_box_points_device_lower = bounding_box_points_devices[device][4:8,:]
			box_info = "Length, Width, Height (mm): " + str(int(length*1000)) + ", " + str(int(width*1000)) + ", " + str(int(height*1000))

			# Draw the box as an overlay on the color image		
			bounding_box_points_device_upper = tuple(map(tuple,bounding_box_points_device_upper.astype(int)))
			for i in range(len(bounding_box_points_device_upper)):	
				cv.line(color_image, bounding_box_points_device_upper[i], bounding_box_points_device_upper[(i+1)%4], (0,255,0), 4)

			bounding_box_points_device_lower = tuple(map(tuple,bounding_box_points_device_lower.astype(int)))
			for i in range(len(bounding_box_points_device_upper)):	
				cv.line(color_image, bounding_box_points_device_lower[i], bounding_box_points_device_lower[(i+1)%4], (0,255,0), 1)
				
			cv.line(color_image, bounding_box_points_device_upper[0], bounding_box_points_device_lower[0], (0,255,0), 1)
			cv.line(color_image, bounding_box_points_device_upper[1], bounding_box_points_device_lower[1], (0,255,0), 1)
			cv.line(color_image, bounding_box_points_device_upper[2], bounding_box_points_device_lower[2], (0,255,0), 1)
			cv.line(color_image, bounding_box_points_device_upper[3], bounding_box_points_device_lower[3], (0,255,0), 1)
			cv.putText(color_image, box_info, (50,50), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0) )
			
		# Visualise the results
		cv.imshow('Color image from RealSense Device Nr: ' + device, color_image)
		cv.waitKey(1)


def calculate_rmsd(points1, points2, validPoints=None):
	"""
	calculates the root mean square deviation between to point sets
	Parameters:
	-------
	points1, points2: numpy matrix (K, N)
	where K is the dimension of the points and N is the number of points
	validPoints: bool sequence of valid points in the point set.
	If it is left out, all points are considered valid
	"""
	assert(points1.shape == points2.shape)
	N = points1.shape[1]

	if validPoints == None:
		validPoints = [True]*N

	assert(len(validPoints) == N)

	points1 = points1[:,validPoints]
	points2 = points2[:,validPoints]

	N = points1.shape[1]

	dist = points1 - points2
	rmsd = 0
	for col in range(N):
		rmsd += np.matmul(dist[:,col].transpose(), dist[:,col]).flatten()[0]

	return np.sqrt(rmsd/N)


def get_chessboard_points_3D(chessboard_params):
	"""
	Returns the 3d coordinates of the chessboard corners
	in the coordinate system of the chessboard itself.
	Returns
	-------
	objp : array
		(3, N) matrix with N being the number of corners
	"""
	assert(len(chessboard_params) == 3)
	width = chessboard_params[0]
	height = chessboard_params[1]
	square_size = chessboard_params[2]
	objp = np.zeros((width * height, 3), np.float32)
	objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)
	return objp.transpose() * square_size


def cv_find_chessboard(depth_frame, infrared_frame, chessboard_params):
    """
    Searches the chessboard corners using the set infrared image and the
    checkerboard size
    Returns:
    -----------
    chessboard_found : bool
                            Indicates wheather the operation was successful
    corners          : array
                            (2,N) matrix with the image coordinates of the chessboard corners
    """
    assert(len(chessboard_params) == 3)
    infrared_image = np.asanyarray(infrared_frame.get_data())
    img = cv.cvtColor(infrared_image, cv.COLOR_BGR2GRAY) 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_found = False
    chessboard_found, corners = cv.findChessboardCorners(img, (chessboard_params[0], chessboard_params[1]))
    corners1 = corners
    if chessboard_found:
        corners = cv.cornerSubPix(img, corners, (11,11),(-1,-1), criteria)
        corners = np.transpose(corners, (2,0,1))
    # cv.drawChessboardCorners(infrared_image, (8,6), corners1, True)
    # cv.imshow("chess", img)
    # cv.waitKey(2)
    return chessboard_found, corners


def getDevice(config=None):
    if world.device_manager is None:
        world.device_manager =  DeviceManager(config)
        return world.device_manager
    else:
        return world.device_manager

def get_depth_at_pixel(depth_frame, pixel_x, pixel_y):
	"""
	Get the depth value at the desired image point
	Parameters:
	-----------
	depth_frame 	 : rs.frame()
						   The depth frame containing the depth information of the image coordinate
	pixel_x 	  	 	 : double
						   The x value of the image coordinate
	pixel_y 	  	 	 : double
							The y value of the image coordinate
	Return:
	----------
	depth value at the desired pixel
	"""
	return depth_frame.as_depth_frame().get_distance(round(pixel_x), round(pixel_y))



def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, camera_intrinsics):
	"""
	Convert the depth and image point information to metric coordinates
	Parameters:
	-----------
	depth 	 	 	 : double
						   The depth value of the image point
	pixel_x 	  	 	 : double
						   The x value of the image coordinate
	pixel_y 	  	 	 : double
							The y value of the image coordinate
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	X : double
		The x value in meters
	Y : double
		The y value in meters
	Z : double
		The z value in meters
	"""
	X = (pixel_x - camera_intrinsics.ppx)/camera_intrinsics.fx *depth
	Y = (pixel_y - camera_intrinsics.ppy)/camera_intrinsics.fy *depth
	return X, Y, depth



def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics ):
	"""
	Convert the depthmap to a 3D point cloud
	Parameters:
	-----------
	depth_frame 	 	 : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters
	"""
	
	[height, width] = depth_image.shape

	nx = np.linspace(0, width-1, width)
	ny = np.linspace(0, height-1, height)
	u, v = np.meshgrid(nx, ny)
	x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
	y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy

	z = depth_image.flatten() / 1000
	x = np.multiply(x,z)
	y = np.multiply(y,z)

	x = x[np.nonzero(z)]
	y = y[np.nonzero(z)]
	z = z[np.nonzero(z)]

	return x, y, z


def convert_pointcloud_to_depth(pointcloud, camera_intrinsics):
	"""
	Convert the world coordinate to a 2D image coordinate
	Parameters:
	-----------
	pointcloud 	 	 : numpy array with shape 3xN
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x coordinate in image
	y : array
		The y coordiante in image
	"""

	assert (pointcloud.shape[0] == 3)
	x_ = pointcloud[0,:]
	y_ = pointcloud[1,:]
	z_ = pointcloud[2,:]

	m = x_[np.nonzero(z_)]/z_[np.nonzero(z_)]
	n = y_[np.nonzero(z_)]/z_[np.nonzero(z_)]

	x = m*camera_intrinsics.fx + camera_intrinsics.ppx
	y = n*camera_intrinsics.fy + camera_intrinsics.ppy

	return x, y



def get_boundary_corners_2D(points):
	"""
	Get the minimum and maximum point from the array of points
	
	Parameters:
	-----------
	points 	 	 : array
						   The array of points out of which the min and max X and Y points are needed
	
	Return:
	----------
	boundary : array
		The values arranged as [minX, maxX, minY, maxY]
	
	"""
	padding=0.05
	if points.shape[0] == 3:
		assert (len(points.shape)==2)
		minPt_3d_x = np.amin(points[0,:])
		maxPt_3d_x = np.amax(points[0,:])
		minPt_3d_y = np.amin(points[1,:])
		maxPt_3d_y = np.amax(points[1,:])

		boudary = [minPt_3d_x-padding, maxPt_3d_x+padding, minPt_3d_y-padding, maxPt_3d_y+padding]

	else:
		raise Exception("wrong dimension of points!")

	return boudary



def get_clipped_pointcloud(pointcloud, boundary):
	"""
	Get the clipped pointcloud withing the X and Y bounds specified in the boundary
	
	Parameters:
	-----------
	pointcloud 	 	 : array
						   The input pointcloud which needs to be clipped
	boundary      : array
										The X and Y bounds 
	
	Return:
	----------
	pointcloud : array
		The clipped pointcloud
	
	"""
	assert (pointcloud.shape[0]>=2)
	pointcloud = pointcloud[:,np.logical_and(pointcloud[0,:]<boundary[1], pointcloud[0,:]>boundary[0])]
	pointcloud = pointcloud[:,np.logical_and(pointcloud[1,:]<boundary[3], pointcloud[1,:]>boundary[2])]
	return pointcloud




# Python 2/3 compatibility
# Make range a iterator in Python 2
try:
    range = xrange
except NameError:
    pass


def kabsch_rmsd(P, Q):
    """
    Rotate matrix P unto Q using Kabsch algorithm and calculate the RMSD.
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rmsd : float
        root-mean squared deviation
    """
    P = kabsch_rotate(P, Q)
    return rmsd(P, Q)


def kabsch_rotate(P, Q):
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm.
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    P : array
        (N,D) matrix, where N is points and D is dimension,
        rotated
    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P


def kabsch(P, Q):
    """
    The optimal rotation matrix U is calculated and then used to rotate matrix
    P unto matrix Q so the minimum root-mean-square deviation (RMSD) can be
    calculated.
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a translation of P and Q
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    Example
    -----
    TODO
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


def quaternion_rmsd(P, Q):
    """
    Rotate matrix P unto Q and calculate the RMSD
    based on doi:10.1016/1049-9660(91)90036-O
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rmsd : float
    """
    rot = quaternion_rotate(P, Q)
    P = np.dot(P, rot)
    return rmsd(P, Q)


def quaternion_transform(r):
    """
    Get optimal rotation
    note: translation will be zero when the centroids of each molecule are the
    same
    """
    Wt_r = makeW(*r).T
    Q_r = makeQ(*r)
    rot = Wt_r.dot(Q_r)[:3, :3]
    return rot


def makeW(r1, r2, r3, r4=0):
    """
    matrix involved in quaternion rotation
    """
    W = np.asarray([
             [r4, r3, -r2, r1],
             [-r3, r4, r1, r2],
             [r2, -r1, r4, r3],
             [-r1, -r2, -r3, r4]])
    return W


def makeQ(r1, r2, r3, r4=0):
    """
    matrix involved in quaternion rotation
    """
    Q = np.asarray([
             [r4, -r3, r2, r1],
             [r3, r4, -r1, r2],
             [-r2, r1, r4, r3],
             [-r1, -r2, -r3, r4]])
    return Q


def quaternion_rotate(X, Y):
    """
    Calculate the rotation
    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.
    Y: array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rot : matrix
        Rotation matrix (D,D)
    """
    N = X.shape[0]
    W = np.asarray([makeW(*Y[k]) for k in range(N)])
    Q = np.asarray([makeQ(*X[k]) for k in range(N)])
    Qt_dot_W = np.asarray([np.dot(Q[k].T, W[k]) for k in range(N)])
    W_minus_Q = np.asarray([W[k] - Q[k] for k in range(N)])
    A = np.sum(Qt_dot_W, axis=0)
    eigen = np.linalg.eigh(A)
    r = eigen[1][:, eigen[0].argmax()]
    rot = quaternion_transform(r)
    return rot


def centroid(X):
    """
    Calculate the centroid from a vectorset X.
    https://en.wikipedia.org/wiki/Centroid
    Centroid is the mean position of all the points in all of the coordinate
    directions.
    C = sum(X)/len(X)
    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    C : float
        centeroid
    """
    C = X.mean(axis=0)
    return C


def rmsd(V, W):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.
    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rmsd : float
        Root-mean-square deviation
    """
    D = len(V[0])
    N = len(V)
    rmsd = 0.0
    for v, w in zip(V, W):
        rmsd += sum([(v[i] - w[i])**2.0 for i in range(D)])
    return np.sqrt(rmsd/N)


def write_coordinates(atoms, V, title=""):
    """
    Print coordinates V with corresponding atoms to stdout in XYZ format.
    Parameters
    ----------
    atoms : list
        List of atomic types
    V : array
        (N,3) matrix of atomic coordinates
    title : string (optional)
        Title of molecule
    """
    N, D = V.shape

    print(str(N))
    print(title)

    for i in range(N):
        atom = atoms[i]
        atom = atom[0].upper() + atom[1:]
        print("{0:2s} {1:15.8f} {2:15.8f} {3:15.8f}".format(
                atom, V[i, 0], V[i, 1], V[i, 2]))


def get_coordinates(filename, fmt):
    """
    Get coordinates from filename in format fmt. Supports XYZ and PDB.
    Parameters
    ----------
    filename : string
        Filename to read
    fmt : string
        Format of filename. Either xyz or pdb.
    Returns
    -------
    atoms : list
        List of atomic types
    V : array
        (N,3) where N is number of atoms
    """
    if fmt == "xyz":
        return get_coordinates_xyz(filename)
    elif fmt == "pdb":
        return get_coordinates_pdb(filename)
    exit("Could not recognize file format: {:s}".format(fmt))


def get_coordinates_pdb(filename):
    """
    Get coordinates from the first chain in a pdb file
    and return a vectorset with all the coordinates.
    Parameters
    ----------
    filename : string
        Filename to read
    Returns
    -------
    atoms : list
        List of atomic types
    V : array
        (N,3) where N is number of atoms
    """
    # PDB files tend to be a bit of a mess. The x, y and z coordinates
    # are supposed to be in column 31-38, 39-46 and 47-54, but this is
    # not always the case.
    # Because of this the three first columns containing a decimal is used.
    # Since the format doesn't require a space between columns, we use the
    # above column indices as a fallback.
    x_column = None
    V = list()
    # Same with atoms and atom naming.
    # The most robust way to do this is probably
    # to assume that the atomtype is given in column 3.
    atoms = list()

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("TER") or line.startswith("END"):
                break
            if line.startswith("ATOM"):
                tokens = line.split()
                # Try to get the atomtype
                try:
                    atom = tokens[2][0]
                    if atom in ("H", "C", "N", "O", "S", "P"):
                        atoms.append(atom)
                    else:
                        # e.g. 1HD1
                        atom = tokens[2][1]
                        if atom == "H":
                            atoms.append(atom)
                        else:
                            raise Exception
                except:
                        exit("Error parsing atomtype for the following line: \n{0:s}".format(line))

                if x_column == None:
                    try:
                        # look for x column
                        for i, x in enumerate(tokens):
                            if "." in x and "." in tokens[i + 1] and "." in tokens[i + 2]:
                                x_column = i
                                break
                    except IndexError:
                        exit("Error parsing coordinates for the following line: \n{0:s}".format(line))
                # Try to read the coordinates
                try:
                    V.append(np.asarray(tokens[x_column:x_column + 3], dtype=float))
                except:
                    # If that doesn't work, use hardcoded indices
                    try:
                        x = line[30:38]
                        y = line[38:46]
                        z = line[46:54]
                        V.append(np.asarray([x, y ,z], dtype=float))
                    except:
                        exit("Error parsing input for the following line: \n{0:s}".format(line))


    V = np.asarray(V)
    atoms = np.asarray(atoms)
    assert(V.shape[0] == atoms.size)
    return atoms, V


def get_coordinates_xyz(filename):
    """
    Get coordinates from filename and return a vectorset with all the
    coordinates, in XYZ format.
    Parameters
    ----------
    filename : string
        Filename to read
    Returns
    -------
    atoms : list
        List of atomic types
    V : array
        (N,3) where N is number of atoms
    """

    f = open(filename, 'r')
    V = list()
    atoms = list()
    n_atoms = 0

    # Read the first line to obtain the number of atoms to read
    try:
        n_atoms = int(f.readline())
    except ValueError:
        exit("Could not obtain the number of atoms in the .xyz file.")

    # Skip the title line
    f.readline()

    # Use the number of atoms to not read beyond the end of a file
    for lines_read, line in enumerate(f):

        if lines_read == n_atoms:
            break

        atom = re.findall(r'[a-zA-Z]+', line)[0]
        atom = atom.upper()

        numbers = re.findall(r'[-]?\d+\.\d*(?:[Ee][-\+]\d+)?', line)
        numbers = [float(number) for number in numbers]

        # The numbers are not valid unless we obtain exacly three
        if len(numbers) == 3:
            V.append(np.array(numbers))
            atoms.append(atom)
        else:
            exit("Reading the .xyz file failed in line {0}. Please check the format.".format(lines_read + 2))

    f.close()
    atoms = np.array(atoms)
    V = np.array(V)
    return atoms, V


def post_process_depth_frame(depth_frame, decimation_magnitude=1.0, spatial_magnitude=2.0, spatial_smooth_alpha=0.5,
                             spatial_smooth_delta=20, temporal_smooth_alpha=0.4, temporal_smooth_delta=20):
    """
    Filter the depth frame acquired using the Intel RealSense device
    Parameters:
    -----------
    depth_frame          : rs.frame()
                           The depth frame to be post-processed
    decimation_magnitude : double
                           The magnitude of the decimation filter
    spatial_magnitude    : double
                           The magnitude of the spatial filter
    spatial_smooth_alpha : double
                           The alpha value for spatial filter based smoothening
    spatial_smooth_delta : double
                           The delta value for spatial filter based smoothening
    temporal_smooth_alpha: double
                           The alpha value for temporal filter based smoothening
    temporal_smooth_delta: double
                           The delta value for temporal filter based smoothening
    Return:
    ----------
    filtered_frame : rs.frame()
                     The post-processed depth frame
    """

    # Post processing possible only on the depth_frame
    assert (depth_frame.is_depth_frame())

    # print("before:", np.asanyarray(depth_frame.get_data())[200:205, 200:205])
    # Available filters and control options for the filters
    decimation_filter = rs.decimation_filter()
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()

    filter_magnitude = rs.option.filter_magnitude
    filter_smooth_alpha = rs.option.filter_smooth_alpha
    filter_smooth_delta = rs.option.filter_smooth_delta

    # Apply the control parameters for the filter
    decimation_filter.set_option(filter_magnitude, decimation_magnitude)
    spatial_filter.set_option(filter_magnitude, spatial_magnitude)
    spatial_filter.set_option(filter_smooth_alpha, spatial_smooth_alpha)
    spatial_filter.set_option(filter_smooth_delta, spatial_smooth_delta)
    temporal_filter.set_option(filter_smooth_alpha, temporal_smooth_alpha)
    temporal_filter.set_option(filter_smooth_delta, temporal_smooth_delta)

    # Apply the filters
    filtered_frame = decimation_filter.process(depth_frame)
    filtered_frame = spatial_filter.process(filtered_frame)
    filtered_frame = temporal_filter.process(filtered_frame)

    # print("after:",np.asanyarray(filtered_frame.get_data())[200:205, 200:205])
    return filtered_frame


def main():

    import argparse
    import sys

    description = """
Calculate Root-mean-square deviation (RMSD) between structure A and B, in XYZ
or PDB format, using transformation and rotation. The order of the atoms *must*
be the same for both structures.
For more information, usage, example and citation read more at
https://github.com/charnley/rmsd
"""

    epilog = """output:
  Normal - RMSD calculated the straight-forward way, no translation or rotation.
  Kabsch - RMSD after coordinates are translated and rotated using Kabsch.
  Quater - RMSD after coordinates are translated and rotated using quaternions.
"""

    parser = argparse.ArgumentParser(
                    usage='%(prog)s [options] structure_a structure_b',
                    description=description,
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    epilog=epilog)

    parser.add_argument('-v', '--version', action='version', version='rmsd ' + __version__ + "\nhttps://github.com/charnley/rmsd")

    parser.add_argument('structure_a', metavar='structure_a', type=str, help='Structure in .xyz or .pdb format')
    parser.add_argument('structure_b', metavar='structure_b', type=str)
    parser.add_argument('-o', '--output', action='store_true', help='print out structure A, centered and rotated unto structure B\'s coordinates in XYZ format')
    parser.add_argument('-f', '--format', action='store', help='Format of input files. Valid format are XYZ and PDB', metavar='fmt')

    parser.add_argument('-m', '--normal', action='store_true', help='Use no transformation')
    parser.add_argument('-k', '--kabsch', action='store_true', help='Use Kabsch algorithm for transformation')
    parser.add_argument('-q', '--quater', action='store_true', help='Use Quaternion algorithm for transformation')

    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument('-n', '--no-hydrogen', action='store_true', help='ignore hydrogens when calculating RMSD')
    index_group.add_argument('-r', '--remove-idx', nargs='+', type=int, help='index list of atoms NOT to consider', metavar='idx')
    index_group.add_argument('-a', '--add-idx', nargs='+', type=int, help='index list of atoms to consider', metavar='idx')


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # As default use all three methods
    if not args.normal and not args.kabsch and not args.quater:
        args.normal = True
        args.kabsch = True
        args.quater = True

    # As default, load the extension as format
    if args.format == None:
        args.format = args.structure_a.split('.')[-1]

    p_atoms, p_all = get_coordinates(args.structure_a, args.format)
    q_atoms, q_all = get_coordinates(args.structure_b, args.format)

    if np.count_nonzero(p_atoms != q_atoms):
        exit("Atoms not in the same order")

    if args.no_hydrogen:
        not_hydrogens = np.where(p_atoms != 'H')
        P = p_all[not_hydrogens]
        Q = q_all[not_hydrogens]

    elif args.remove_idx:
        N, = p_atoms.shape
        index = range(N)
        index = set(index) - set(args.remove_idx)
        index = list(index)
        P = p_all[index]
        Q = q_all[index]

    elif args.add_idx:
        P = p_all[args.add_idx]
        Q = q_all[args.add_idx]

    else:
        P = p_all[:]
        Q = q_all[:]


    # Calculate 'dumb' RMSD
    if args.normal and not args.output:
        normal_rmsd = rmsd(P, Q)
        print("Normal RMSD: {0}".format(normal_rmsd))

    # Create the centroid of P and Q which is the geometric center of a
    # N-dimensional region and translate P and Q onto that center.
    # http://en.wikipedia.org/wiki/Centroid
    Pc = centroid(P)
    Qc = centroid(Q)
    P -= Pc
    Q -= Qc

    if args.output:
        U = kabsch(P, Q)
        p_all -= Pc
        p_all = np.dot(p_all, U)
        p_all += Qc
        write_coordinates(p_atoms, p_all, title="{} translated".format(args.structure_a))
        quit()

    if args.kabsch:
        print("Kabsch RMSD: {0}".format(kabsch_rmsd(P, Q)))

    if args.quater:
        print("Quater RMSD: {0}".format(quaternion_rmsd(P, Q)))


if __name__ == "__main__":
    main()
