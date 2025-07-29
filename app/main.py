import math
from typing import Sequence
import cv2 as cv
from cv2.typing import MatLike

from tkinter import Tk, filedialog

import numpy as np
import torch
import torchvision.transforms as transforms

from cv_image import CvImage, CvImagePoint, CvImageShape

from digit_model import DigitModel
from sudoku_board import SudokuBoard



# Contour biggest blob
# perspective tranform
# corner detection

if __name__ == "__main__":
	#Initilize tkinter for file dialog
	root: Tk = Tk()
	root.withdraw()

	# Open file dialog and make user select image
	image_path: str = filedialog.askopenfilename(
		title="Select Sudoku Image",
		filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
	)
	assert image_path != '', "Error: No image selected."

	image_show_size: int = 600
	
	# Original Image
	original_image: CvImage = CvImage.from_file_path(image_path).show("Original Image", None, image_show_size)

	# Original Image
	image: CvImage = original_image.clone()
	
	# Filtering, Thresholding, and Opening
	image.grayscale().gaussian_blur(3).otsu_threshold().show("Thresholded", None, 800).opening(3).show("Opening", None, image_show_size)

	# Contours, Corner Points & Warping Perspective
	largest_contour: MatLike = image.get_contours_sorted(cv.contourArea, True)[0][0]
	largest_contour_corners: list[CvImagePoint] = image.get_contour_quadrilateral_approximation_corners(largest_contour)
	original_image.clone().draw_points(largest_contour_corners, 3, [255, 0, 0], -1).show("Quadrilateral Corners", None, image_show_size)
	warped_image: CvImage = original_image.clone().warp_perspective(largest_contour_corners).show("Warped Perspective", None, image_show_size)
	warped_image.grayscale().gaussian_blur(3).otsu_threshold().show("Warped Thresholded", None, image_show_size)

	# Predefined Points and Harris Corners
	image_shape: CvImageShape = warped_image.get_shape()
	harris_raw_corners: list[CvImagePoint] = warped_image.get_harris_corners(5, 5, 0.08, 0.01)
	predefined_corners: list[CvImagePoint] = [CvImagePoint((x / 9) * (image_shape.height), (y / 9) * (image_shape.width)) for y in range(0, 10) for x in range(0, 10)]
	harris_grouped_corners: list[CvImagePoint] = warped_image.get_harris_corners_predefined_groups(7, 7, 0.05, 0.01, predefined_corners, 0.002)

	warped_original_image: CvImage = original_image.clone().warp_perspective(largest_contour_corners)
	harris_corner_image: CvImage = warped_original_image.clone()
	harris_corner_image.draw_points(harris_raw_corners, 3, [255, 0, 0], -1).show("Raw Harris Corners", None, image_show_size)
	harris_corner_image.draw_points(predefined_corners, 3, [0, 255, 0], -1).show("Predefined Corners", None, image_show_size)
	harris_corner_image.draw_points(harris_grouped_corners, 3, [0, 0, 255], -1).show("Grouped Harris Corners", None, image_show_size)

	model = DigitModel()
	model.load_state_dict(torch.load("./model/mnist_model.pt"))
	model.eval()

	sudoku_board: SudokuBoard = SudokuBoard()
	for i in range(0, 89):
		if (i + 1) % 10 == 0:
			continue
		print(i)
		corners: list[CvImagePoint] = [harris_grouped_corners[i], harris_grouped_corners[i + 10], harris_grouped_corners[i + 1], harris_grouped_corners[i + 11]]
		cell_image: CvImage = warped_original_image.clone().warp_perspective(corners).grayscale().gaussian_blur(3).otsu_threshold().show("Cell", None, image_show_size)
		cell_contours = cell_image.get_contours_sorted(cv.contourArea, True, cv.RETR_LIST)[0]
		for cell_contour in cell_contours:
			height, width = cell_image.get_shape()[:2]

			x, y, w, h = cv.boundingRect(cell_contour)
			square_side = max(w, h)

			contour_moments = cv.moments(cell_contour)
			contour_center = CvImagePoint(contour_moments['m10'] / contour_moments['m00'], contour_moments['m01'] / contour_moments['m00']) if contour_moments['m00'] != 0 else CvImagePoint(0, 0)

			row, col = i // 10, i % 10
			if w * h < height * width * 0.10:
				sudoku_board.set_value(row, col, 0)
			elif w * h > height * width * 0.70:
				sudoku_board.set_value(row, col, 0)
			elif math.dist(contour_center, [width / 2, height / 2]) > math.dist([0, 0], [width, height]) * 0.10:
				sudoku_board.set_value(row, col, 0)
			else:
				x = x - (square_side - w) // 2
				y = y - (square_side - h) // 2

				cell_image.warp_perspective([CvImagePoint(x, y), CvImagePoint(x, y + square_side), CvImagePoint(x + square_side, y), CvImagePoint(x + square_side, y + square_side)]).resize_to(20, 20)
				cell_image.image = cv.copyMakeBorder(cell_image.image, 4, 4, 4, 4, cv.BORDER_CONSTANT, value=0)
				cell_image.show("Cell Resized", None, image_show_size)
				digit = model(transforms.ToTensor()(cell_image.image).unsqueeze(0)).argmax()
				print(f"Predicted Digit: {digit.item()}")
				sudoku_board.set_value(row, col, digit.item())
				break
		
	sudoku_board.print_board()
		
		

	# To reload the saved model for inference
	#model = md.NeuralNet()
	#model.load_state_dict(torch.load("mnist_model.pt"))
	#model.eval()