from collections import namedtuple
from collections.abc import Callable, Sequence
import math
import os.path
from typing import cast

import cv2 as cv
from cv2.typing import MatLike, Scalar
import numpy as np

# (y, x) format for points
CvImagePoint = namedtuple("CvImagePoint", ["x", "y"])
CvImageShape = namedtuple("CvImageShape", ["width", "height", "channels"])

class CvImage:    
    def __init__(self: "CvImage", image: MatLike, file_path: str | None = None) -> None:
        self.image: MatLike = image
        self.file_path: str | None = file_path

    @staticmethod
    def from_matrix(matrix: MatLike) -> "CvImage":
        return CvImage(matrix)
    
    @staticmethod
    def from_file_path(file_path: str) -> "CvImage":
        assert os.path.isfile(file_path), f'Error: File "{file_path}" does not exist.'
        image: MatLike | None = cv.imread(file_path)
        assert image is not None, f'Error: Could not read file "{file_path}".'

        return CvImage(image, file_path)

    def get_shape(self: "CvImage") -> CvImageShape:
        shape: tuple[int, int] | tuple[int, int, int] = self.image.shape
        return CvImageShape(self.image.shape[0], self.image.shape[1], self.image.shape[2] if len(shape) == 3 else 1)

    def clone(self: "CvImage") -> "CvImage":
        return CvImage(self.image.copy(), self.file_path)
    
    def grayscale(self: "CvImage") -> "CvImage":
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY, None, 0, cv.ALGO_HINT_DEFAULT)
        return self
    
    def invert(self: "CvImage") -> "CvImage":
        self.image = cv.invert(self.image, None)[1]
        return self
    
    def gaussian_blur(self: "CvImage", kernel_size: int, sigma_x: float = 0, sigma_y: float = 0) -> "CvImage":
        self.image = cv.GaussianBlur(self.image, (kernel_size, kernel_size), sigma_x, None, sigma_y, cv.BORDER_DEFAULT, cv.ALGO_HINT_DEFAULT)
        return self
    
    def otsu_threshold(self: "CvImage", threshold: float = 0, max_value = 255) -> "CvImage":
        self.image = cv.threshold(self.image, threshold, max_value, cv.THRESH_BINARY_INV + cv.THRESH_OTSU, None)[1]
        return self
    
    def erode(self: "CvImage", kernel_size: int, iterations: int = 1) -> "CvImage":
        kernel: MatLike = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size), [-1, -1])
        self.image = cv.erode(self.image, kernel, None, [-1, -1], iterations, cv.BORDER_DEFAULT, 0)
        return self
    
    def opening(self: "CvImage", kernel_size: int, iterations: int = 1) -> "CvImage":
        kernel: MatLike = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size), [-1, -1])
        self.image = cv.morphologyEx(self.image, cv.MORPH_OPEN, kernel, None, [-1, -1], iterations, cv.BORDER_DEFAULT, 0)
        return self
    
    def get_contours(self: "CvImage", mode: int = cv.RETR_TREE, method: int = cv.CHAIN_APPROX_NONE, contours: Sequence[MatLike] | None = None, hierarchy: MatLike | None = None) -> tuple[Sequence[MatLike], MatLike]:
        return cv.findContours(self.image, mode, method, contours, hierarchy, (0, 0))
    
    def get_contours_sorted(self: "CvImage", key: Callable[[MatLike], float], descending: bool = False, mode: int = cv.RETR_TREE, method: int = cv.CHAIN_APPROX_NONE, contours: Sequence[MatLike] | None = None, hierarchy: MatLike | None = None) -> tuple[Sequence[MatLike], MatLike]:
        new_contours: Sequence[MatLike]
        new_hierarchy: MatLike
        new_contours, new_hierarchy = self.get_contours(mode, method, contours, hierarchy)

        return sorted(new_contours, key=key, reverse=descending), new_hierarchy
    
    def get_contour_quadrilateral_approximation_corners(self: "CvImage", contour: MatLike, epsilon_percentage: float = -1) -> list[CvImagePoint]:
        quadrilateral_corners: list[tuple[float, float]] = cv.approxPolyN(contour, 4, None, epsilon_percentage, True)[0]

        # Get x and y values of the corners in separate lists
        x_list: list[float] = [corner[0] for corner in quadrilateral_corners]
        y_list: list[float] = [corner[1] for corner in quadrilateral_corners]
    
        # Get the min and max values of x and y values
        x_min: float = min(x_list)
        x_max: float = max(x_list)
        y_min: float = min(y_list)
        y_max: float = max(y_list)
    
        # Get non-rotated bounding box corners that encompass the quadrilateral
        # In the order of top-left, top-right, bottom-right, bottom-left
        bounding_box_corners: list[CvImagePoint] = [CvImagePoint(x_min, y_min), CvImagePoint(x_min, y_max), CvImagePoint(x_max, y_min), CvImagePoint(x_max, y_max)]

        # Quadrilateral corners in the order of top-left, top-right, bottom-right, bottom-left
        quadrilateral_corners_sorted: list[CvImagePoint | None] = [None, None, None, None]
        quadrilateral_corners_assigned: list[bool] = [False, False, False, False]
	
        for i in range(0, 4):
            for j, quadrilateral_corner in enumerate(quadrilateral_corners):
                if quadrilateral_corners_assigned[j] is True:
                    continue

                # Calculate the distances between the bounding box corners and the quadrilateral corner
                bounding_box_corners_distances: list[float] = [math.dist(quadrilateral_corner, bounding_box_corner) for bounding_box_corner in bounding_box_corners]
                bounding_box_corners_distances_min: float | None = None
                bounding_box_corners_distances_min_index: int | None = None

                for k, bbox_corner_distance in enumerate(bounding_box_corners_distances):
                    if quadrilateral_corners_sorted[k] is not None:
                        continue

                    # If no minimum distance has been found yet, or the current distance is smaller than the minimum distance
                    if (bounding_box_corners_distances_min is None) or (bbox_corner_distance < bounding_box_corners_distances_min):
                        bounding_box_corners_distances_min = bbox_corner_distance
                        bounding_box_corners_distances_min_index = k
                    elif (i < 4) and (bbox_corner_distance == bounding_box_corners_distances_min):
                        bounding_box_corners_distances_min_index = None

                # If a minimum distance has been found, assign the quadrilateral corner to the bounding box corner
                if bounding_box_corners_distances_min_index is not None:
                    quadrilateral_corners_sorted[bounding_box_corners_distances_min_index] = CvImagePoint(quadrilateral_corner[0], quadrilateral_corner[1])
                    quadrilateral_corners_assigned[j] = True
    	
        return cast(list[CvImagePoint], quadrilateral_corners_sorted)
    
    def get_contour_extreme_quadrilateral_corners(self: "CvImage", contour: MatLike) -> list[CvImagePoint]:
        contour_count: int = len(contour)

        # Get x and y values of the corners in separate lists
        x_list: list[float] = [contour[i][0][0] for i in range(0, contour_count)]
        y_list: list[float] = [contour[i][0][1] for i in range(0, contour_count)]
    
        # Get the min and max values of x and y values
        x_min: float = min(x_list)
        x_max: float = max(x_list)
        y_min: float = min(y_list)
        y_max: float = max(y_list)
    
        # Get non-rotated bounding box corners that encompass the quadrilateral
        # In the order of top-left, top-right, bottom-right, bottom-left
        bounding_box_corners: list[CvImagePoint] = [CvImagePoint(x_min, y_min), CvImagePoint(x_min, y_max), CvImagePoint(x_max, y_min), CvImagePoint(x_max, y_max)]

        # Quadrilateral corners in the order of top-left, top-right, bottom-right, bottom-left
        extreme_quadrilateral_corners: list[CvImagePoint] = bounding_box_corners.copy()
        extreme_quadrilateral_corners.reverse()
	
        for i, bounding_box_corner in enumerate(bounding_box_corners):
            bounding_box_corner_min_distance: float = math.dist(bounding_box_corner, extreme_quadrilateral_corners[i])

            for j in range(0, contour_count):
                corner_distance: float = math.dist(contour[j][0], bounding_box_corner)

                if corner_distance < bounding_box_corner_min_distance:
                    bounding_box_corner_min_distance = corner_distance
                    extreme_quadrilateral_corners[i] = CvImagePoint(contour[j][0][0], contour[j][0][1])

        return extreme_quadrilateral_corners
    
    def warp_perspective(self: "CvImage", quadrilateral_corners: list[CvImagePoint]) -> "CvImage":
        shape: CvImageShape = self.get_shape()
        
        transform_matrix: MatLike = cv.getPerspectiveTransform(np.array(quadrilateral_corners, np.float32), np.array([(0, 0), (0, shape.height), (shape.width, 0), (shape.width, shape.height)], np.float32))
        
        self.image = cv.warpPerspective(self.image, transform_matrix, shape[:2], None, cv.INTER_AREA, cv.BORDER_DEFAULT, 0)
        return self
    
    def resize_to(self: "CvImage", size_x: int, size_y: int) -> "CvImage":
        self.image = cv.resize(self.image, (size_x, size_y), None, 0, 0, cv.INTER_AREA)
        return self
    
    def resize_by(self: "CvImage", factor_x: float, factor_y: float) -> "CvImage":
        self.image = cv.resize(self.image, None, None, factor_x, factor_y, cv.INTER_AREA)
        return self
    
    def get_harris_corners(self: "CvImage", block_size: int, kernel_size: int, k: float, threshold_percentage: float) -> list[CvImagePoint]:
        corner_probability_matrix: MatLike = cv.cornerHarris(self.image, block_size, kernel_size, k, None, cv.BORDER_DEFAULT)
        corner_probability_matrix_shape: tuple[int, int] = corner_probability_matrix.shape

        corners: list[CvImagePoint] = []
        corner_probability_threshold: float = corner_probability_matrix.max() * threshold_percentage

        for x in range(0, corner_probability_matrix_shape[1]):
            for y in range(0, corner_probability_matrix_shape[0]):
                if corner_probability_matrix[y][x] >= corner_probability_threshold:
                    corners.append(CvImagePoint(x, y))
        
        return corners
    
    def get_harris_corners_predefined_groups(self: "CvImage", block_size: int, kernel_size: int, k: float, threshold_percentage: float, predefined_corners: list[CvImagePoint], predefined_corner_area_percentage: float) -> list[CvImagePoint]:
        harris_corners: list[CvImagePoint] = self.get_harris_corners(block_size, kernel_size, k, threshold_percentage)
        grouped_corners: list[CvImagePoint] = []

        image_shape: CvImageShape = self.get_shape()
        predefined_corner_radius: float = ((image_shape.height * image_shape.width * predefined_corner_area_percentage) / math.pi) ** 0.5

        for predefined_corner in predefined_corners:
            in_range_corners: list[CvImagePoint] = []

            for harris_corner in harris_corners:
                if math.dist(predefined_corner, harris_corner) <= predefined_corner_radius:
                    in_range_corners.append(harris_corner)
            
            in_range_corners_count: int = len(in_range_corners)
            if in_range_corners_count == 0:
                grouped_corners.append(predefined_corner)
            else:
                x_sum: float = sum(corner.x for corner in in_range_corners)
                y_sum: float = sum(corner.y for corner in in_range_corners)

                grouped_corners.append(CvImagePoint(x_sum / in_range_corners_count, y_sum / in_range_corners_count))

        return grouped_corners
    
    def draw_points(self: "CvImage", points: list[CvImagePoint], radius: int, color: Scalar, border_thickness: int, border_type: int = cv.LINE_AA) -> "CvImage":
        for point in points:
            cv.circle(self.image, (int(point.x), int(point.y)), int(radius), color, border_thickness, border_type, 0)

        return self

    def show(self: "CvImage", title: str, size_x: int | None = None, size_y: int | None = None) -> "CvImage":
        new_title: str = f'{title} ({self.file_path})' if self.file_path else title

        if (size_x is None) and (size_y is None):
            cv.imshow(new_title, self.image)
            cv.waitKey(0)
        elif (size_x is not None) and (size_y is not None):
            self.clone().resize_to(size_x, size_y).show(title)
        elif size_x is not None:
            factor: float = size_x / self.get_shape().width
            self.clone().resize_by(factor, factor).show(title)
        elif size_y is not None:
            factor: float = size_y / self.get_shape().height
            self.clone().resize_by(factor, factor).show(title)
        else:
            assert "How did we get here?"

        return self