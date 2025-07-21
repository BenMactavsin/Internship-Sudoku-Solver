import cv2 as cv
from cv2.typing import MatLike

import tkinter as tk
from tkinter import filedialog

#import numpy as np
import image_processing as ip



# Contour biggest blob
# perspective tranform
# corner detection



# Displays image in a seperate window
def show_image(title: str, image: MatLike) -> None:
	image = cv.resize(image, None, None, fx=800/max(image.shape), fy=800/max(image.shape), interpolation=cv.INTER_AREA)
	cv.imshow(title, image)
	cv.waitKey(0)

if __name__ == "__main__":
	#Initilize tkinter for file dialog
	root: tk.Tk = tk.Tk()
	root.withdraw()

	# Open file dialog and make user select image
	image_path: str = filedialog.askopenfilename(
		title="Select Sudoku Image",
		filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
	)
	assert image_path != '', "Error: No image selected."
	
	# Read the image
	original_image: MatLike | None = cv.imread(image_path)
	assert original_image is not None, "Error: Could not open image."

	# Original Image
	show_image(f"{image_path} (Original İmage)", original_image)
	
	# Filtering & Thresholding
	grayscale_image: MatLike = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
	modified_image = ip.apply_gaussian_blur(grayscale_image, 3)
	modified_image = ip.apply_otsu_threshold(modified_image)
	modified_image = ip.apply_morphological_opening(modified_image, 3)
	show_image(f"{image_path} (Thresholded)", modified_image)

	# Contours
	contours, hierarchy = ip.get_contours(modified_image)
	biggest_contour = sorted(contours, key=cv.contourArea, reverse=True)[0]
	polygon = ip.get_quadrilateral_contour(biggest_contour, 4)
	polygon_image = ip.draw_contours(modified_image, [polygon], hierarchy)
	show_image(f"{image_path} (Largest Contour)", polygon_image)

	quad_corners = ip.get_quadrilateral_corners_tl_tr_br_bl(polygon)
	transformed_image = ip.apply_perspective_transform(grayscale_image, quad_corners)
	show_image(f"{image_path} (Zoomed In)", transformed_image)

	# square_corners = ip.get_good_features_corner(transformed_image, 100, 0.005, (transformed_image.shape[0] / 9) * 0.90)
	# print(len(square_corners), "corners found.")

	for x in range(0, 10):
		for y in range(0, 10):
			cv.circle(transformed_image, (int(((transformed_image.shape[1] - 5) * 0.99) * (x/9) + 5), int(((transformed_image.shape[0] - 5) * 0.99) * (y/9) + 5)), 3, 255, -1)
	
	show_image(f"{image_path} (Corners)", transformed_image)
