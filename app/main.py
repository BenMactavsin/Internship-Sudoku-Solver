from pathlib import Path
from digit_classifier import DigitClassifier
from sudoku_board import SudokuBoard
from sudoku_image_processor import SudokuImageProcessor
import cv2 as cv

root_path = Path(".")
image_path = root_path/"images"/"example_3.jpg"
model_path = root_path/"model"/"396.pth"

if __name__ == "__main__":
	original = SudokuImageProcessor.from_path(image_path)

	processor = original.clone()
	processor.to_grayscale()
	processor.filter()
	processor.threshold()
	processor.morph()
	processor.get_grid_corners()

	warped_original = original.clone()
	warped_original.warp_to(processor.grid_corners)

	processor = warped_original.clone()
	processor.to_grayscale()
	processor.filter()
	processor.threshold()
	processor.get_cell_corners()

	digit_classifier = DigitClassifier()
	digit_classifier.load_from(model_path)
	digit_classifier.eval()

	sudoku_board = SudokuBoard()

	for i in range(0, 81):
		x, y = i % 9, i // 9

		cell_processor = warped_original.clone()
		cell_processor.warp_to([processor.actual_corners[j] for j in [x+y*10, (x+1)+y*10, x+(y+1)*10, (x+1)+(y+1)*10]])
		cell_processor.to_grayscale()
		cell_processor.filter()
		cell_processor.threshold()
		cell_processor.get_cell_brect()

		if cell_processor.image is None:
			continue

		digit = digit_classifier.classify_digit(cell_processor.image)
		sudoku_board.set(x, y, digit)
	
	
	sudoku_board.print_board()
	print("\n")
	if not sudoku_board.solve_for_all():
		print("Could not solve sudoku!")
	else:
		sudoku_board.print_board()