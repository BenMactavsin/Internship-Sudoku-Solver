import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from typing import Sequence

from typing import cast
import math

def apply_invert(image: MatLike) -> MatLike:
	return cv.bitwise_not(
		src=image,
		dst=None,
		mask=None
	)

def apply_gaussian_blur(image: MatLike, kernel_size: int) -> MatLike:
	return cv.GaussianBlur(
		src=image, 
		ksize=(kernel_size, kernel_size),
		sigmaX=0,
		dst=None,
		sigmaY=0,
		borderType=cv.BORDER_DEFAULT,
		hint=cv.ALGO_HINT_DEFAULT
	)

def apply_bilateral_filtering(image: MatLike, diameter: int, sigmaColor: float, sigmaSpace: float) -> MatLike:
	return cv.bilateralFilter(
		src=image, 
		d=diameter,
		sigmaColor=sigmaColor,
		sigmaSpace=sigmaSpace,
		dst=None,
		borderType=cv.BORDER_DEFAULT
	)

def apply_laplacian(image: MatLike, kernel_size: int, scale: float, delta: float) -> MatLike:
	return cv.Laplacian(
		src=image,
		ddepth=cv.CV_8U,
		dst=None,
		ksize=kernel_size,
		scale=scale,
		delta=delta,
		borderType=cv.BORDER_DEFAULT
	)

def apply_otsu_threshold(image: MatLike) -> MatLike:
	return cv.threshold(
		src=image,
		thresh=0,
		maxval=255,
		type=cv.THRESH_BINARY_INV + cv.THRESH_OTSU,
		dst=None
	)[1]

def apply_adaptive_gaussian_threshold(image: MatLike, block_size: int, constant: float) -> MatLike:
	return cv.adaptiveThreshold(
		src=image,
		maxValue=255,
		adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
		thresholdType=cv.THRESH_BINARY_INV,
		blockSize=block_size,
		C=constant,
		dst=None
	)

def apply_morphological_opening(image: MatLike, kernel_size: int) -> MatLike:
	kernel = np.ones((kernel_size, kernel_size), np.uint8)

	return cv.morphologyEx(
		src=image,
		op=cv.MORPH_OPEN,
		kernel=kernel,
		dst=None,
		anchor=(-1, -1),
		iterations=1,
		borderType=cv.BORDER_CONSTANT,
		borderValue=(0, 0, 0)
	)

def apply_morphological_closing(image: MatLike, kernel_size: int) -> MatLike:
	kernel = np.ones((kernel_size, kernel_size), np.uint8)

	return cv.morphologyEx(
		src=image,
		op=cv.MORPH_CLOSE,
		kernel=kernel,
		dst=None,
		anchor=(-1, -1),
		iterations=1,
		borderType=cv.BORDER_CONSTANT,
		borderValue=(0, 0, 0)
	)

def apply_morphological_dilate(image: MatLike, kernel_size: int) -> MatLike:
	kernel = np.ones((kernel_size, kernel_size), np.uint8)

	return cv.dilate(
		src=image,
		kernel=kernel,
		dst=None,
		anchor=(-1, -1),
		iterations=1,
		borderType=cv.BORDER_CONSTANT,
		borderValue=(0, 0, 0)
	)

def apply_morphological_erode(image: MatLike, kernel_size: int) -> MatLike:
	kernel = np.ones((kernel_size, kernel_size), np.uint8)

	return cv.erode(
		src=image,
		kernel=kernel,
		dst=None,
		anchor=(-1, -1),
		iterations=1,
		borderType=cv.BORDER_CONSTANT,
		borderValue=(0, 0, 0)
	)

def get_contours(image: MatLike) -> tuple[Sequence[MatLike], MatLike]:
	return cv.findContours(
		image=image,
		mode=cv.RETR_EXTERNAL,
		method=cv.CHAIN_APPROX_NONE,
		contours=None,
		hierarchy=None,
		offset=(0, 0)
	)

def draw_contours(image: MatLike, contour: Sequence[MatLike], hierarchy: MatLike | None) -> MatLike:
	canvas = cast(MatLike, np.zeros(image.shape, np.uint8))
	return cv.drawContours(
		image=canvas,
		contours=contour,
		contourIdx=-1,
		color=(255, 255, 255),
		thickness=2,
		lineType=cv.FILLED,
		hierarchy=hierarchy,
		maxLevel=0,
		offset=(0, 0)
	)

def get_quadrilateral_contour(contour: MatLike, nsides: int) -> MatLike:
	return cv.approxPolyN(
		curve=contour,
		nsides=nsides,
		approxCurve=None,
		epsilon_percentage=-1,
		ensure_convex=True
	)

def get_quadrilateral_corners_tl_tr_br_bl(quad: MatLike) -> MatLike:
	# Get x and y values of the corners in separate lists
	x_list: list[float] = [corner[1] for corner in quad[0]]
	y_list: list[float] = [corner[0] for corner in quad[0]]

	# Get the min and max values of x and y values
	x_min: float = min(x_list)
	x_max: float = max(x_list)
	y_min: float = min(y_list)
	y_max: float = max(y_list)

	# Get non-rotated bounding box corners that encompass the quadrilateral
	# In the order of top-left, top-right, bottom-right, bottom-left
	bbox_corners: list[list[float]] = [
		[y_min, x_min],
		[y_min, x_max],
		[y_max, x_max],
		[y_max, x_min]
	]

	# Quadrilateral corners in the order of top-left, top-right, bottom-right, bottom-left
	quad_sorted_corners: list[list[float] | None] = [None, None, None, None]

	# Holds the status of whether a quadrilateral corner has been assigned to some value in quad_corners
	quad_assigned_corners: list[bool] = [False, False, False, False]
	
	for _ in range(0, 4):
		for j, quad_corner in enumerate(quad[0]):
			if quad_assigned_corners[j] is True:
				continue
			
			# Calculate the distances between the bounding box corners and the quadrilateral corner
			bbox_corner_distances: list[float] = [math.dist(quad_corner, bbox_corner) for bbox_corner in bbox_corners]
			bbox_corner_min_distance: float | None = None
			bbox_corner_min_index: int | None = None

			for k, bbox_corner_distance in enumerate(bbox_corner_distances):
				if quad_sorted_corners[k] is not None:
					continue
				
				# If no minimum distance has been found yet, or the current distance is smaller than the minimum distance
				if (bbox_corner_min_distance is None) or (bbox_corner_distance < bbox_corner_min_distance):
					bbox_corner_min_distance = bbox_corner_distance
					bbox_corner_min_index = k
				elif bbox_corner_distance == bbox_corner_min_distance:
					bbox_corner_min_index = None
			
			# If a minimum distance has been found, assign the quadrilateral corner to the bounding box corner
			if bbox_corner_min_index is not None:
				quad_sorted_corners[bbox_corner_min_index] = quad_corner
				quad_assigned_corners[j] = True

	# Final Enumeration to ensure all corners are assigned
	for j, quad_corner in enumerate(quad[0]):
		if quad_assigned_corners[j] is True:
			continue
		
		# Calculate the distances between the bounding box corners and the quadrilateral corner
		bbox_corner_distances: list[float] = [math.dist(quad_corner, bbox_corner) for bbox_corner in bbox_corners]
		bbox_corner_min_distance: float | None = None
		bbox_corner_min_index: int | None = None

		for k, bbox_corner_distance in enumerate(bbox_corner_distances):
			if quad_sorted_corners[k] is not None:
				continue
			
			if (bbox_corner_min_distance is None) or (bbox_corner_distance < bbox_corner_min_distance):
				bbox_corner_min_distance = bbox_corner_distance
				bbox_corner_min_index = k

		# If a minimum distance has been found, assign the quadrilateral corner to the bounding box corner
		if bbox_corner_min_index is not None:
			quad_sorted_corners[bbox_corner_min_index] = quad_corner
			quad_assigned_corners[j] = True
	
	return np.array(quad_sorted_corners, dtype=np.float32)

def apply_perspective_transform(image: MatLike, corners: MatLike) -> MatLike:
	height: int
	width: int
	height, width = image.shape[:2]

	transform: MatLike = cv.getPerspectiveTransform(
		src=np.array(corners, dtype=np.float32),
		dst=np.array([[0, 0], [0, width], [height, width], [height, 0]], dtype=np.float32),
		solveMethod=cv.DECOMP_LU
	)

	return cv.warpPerspective(
		src=image,
		M=transform,
		dsize=image.shape,
		flags=cv.INTER_AREA
	)

def get_harris_corners(image: MatLike) -> MatLike:
	corners = cv.cornerHarris(
		src=image,
		blockSize=2,
		ksize=3,
		k=0.04,
		dst=None,
		borderType=cv.BORDER_DEFAULT
	)

	return corners

def get_good_features_corner(image: MatLike, maxCorners: int, qualityLevel: float, minDistance: float) -> MatLike:
	return cv.goodFeaturesToTrack(
		image=image,
		maxCorners=maxCorners,
		qualityLevel=qualityLevel,
		minDistance=minDistance,
		corners=None,
		mask=None,
		blockSize=3,
		useHarrisDetector=True
	)