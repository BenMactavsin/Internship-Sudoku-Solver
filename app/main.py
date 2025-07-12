import cv2




# Contour biggest blob
# perspective tranform
# corner detection






if __name__ == "__main__":
	image_path = "images\example_1.jpeg"
	sudoku_image = cv2.imread(image_path)

	cv2.imshow("Input Image", sudoku_image)
	cv2.waitKey(0)
