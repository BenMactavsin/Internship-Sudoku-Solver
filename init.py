import numpy
import cv2 as opencv
import matplotlib.pyplot as plotlib
import os

## Image Paths
image_root = 'images/'
image_paths = [image_root + path for path in os.listdir(image_root)]

## Figure and Subfigure Definitions
fig_rows = 4
fig_columns = 1
plot_rows = 1
plot_columns = len(image_paths)

fig = plotlib.figure()
subfigs = fig.subfigures(nrows=fig_rows, ncols=fig_columns, squeeze=False)

subfigs[0, 0].suptitle("Original Image") # Figure 1
subfigs[1, 0].suptitle("Low-Pass Filtering") # Figure 2
subfigs[2, 0].suptitle("Adaptive Gaussian Thresholding") # Figure 3
subfigs[3, 0].suptitle("Morphological Gradient") # Figure 4

## Plot Definitions
for i in range(0, fig_rows):
	for j in range(0, fig_columns):
		ax = subfigs[i, j].subplots(nrows=plot_rows, ncols=plot_columns, squeeze=False)
		for k in range(0, plot_rows):
			for n in range(0, plot_columns):
				ax[k, n].set_xticks([])
				ax[k, n].set_yticks([])

## Sets image to
def set_plot_image(img, fig_row, fig_col, plot_row, plot_col):
	ax = subfigs[fig_row, fig_col].axes[(plot_columns * plot_row) + plot_col]
	ax.imshow(img, cmap = 'gray')

for i in range(0, plot_columns):
	# Opening Image
	img = opencv.imread(image_paths[i], opencv.IMREAD_GRAYSCALE)
	kernel = numpy.ones((3,3),numpy.uint8)/9

	# Figure 1
	set_plot_image(img, 0, 0, 0, i)

	# Figure 2 Low-Pass Filtering
	img = opencv.filter2D(img, -1, kernel)
	set_plot_image(img, 1, 0, 0, i)

	# Figure 3 Adaptive Gaussian Thresholding
	img = opencv.adaptiveThreshold(img, 255, opencv.ADAPTIVE_THRESH_GAUSSIAN_C, opencv.THRESH_BINARY_INV, 3, 1)
	set_plot_image(img, 2, 0, 0, i)

	# Figure 4 Morphological Gradient
	set_plot_image(img, 3, 0, 0, i)

# Export all results
fig.savefig("export.png", bbox_inches='tight', dpi=1000)