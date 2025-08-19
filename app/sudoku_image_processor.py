from pathlib import Path
import cv2 as cv
from cv2.typing import MatLike, Size
import numpy as np
from collections.abc import Sequence
from math import dist

class SudokuImageProcessor:    
    def __init__(self: "SudokuImageProcessor", image: MatLike) -> None:
        self.image: MatLike = image.copy()
        self.grid_contour: MatLike = None
        self.grid_corners: Sequence[tuple[int, int]] = None
        self.harris_corners: Sequence[tuple[int, int]] = None
        self.ideal_corners: Sequence[tuple[int, int]] = None
        self.actual_corners: Sequence[tuple[int, int]] = None

    @staticmethod
    def from_path(path: Path) -> "SudokuImageProcessor":
        absolute_path: str = str(path.resolve())
        return SudokuImageProcessor(cv.imread(absolute_path))

    def clone(self: "SudokuImageProcessor") -> "SudokuImageProcessor":
        copy = SudokuImageProcessor(self.image.copy())
        copy.grid_contour = self.grid_contour.copy() if self.grid_contour is not None else None
        copy.grid_corners = self.grid_corners.copy() if self.grid_corners is not None else None
        copy.harris_corners = self.harris_corners.copy() if self.harris_corners is not None else None
        copy.ideal_corners = self.ideal_corners.copy() if self.ideal_corners is not None else None
        copy.actual_corners = self.actual_corners.copy() if self.actual_corners is not None else None
        return SudokuImageProcessor(self.image.copy())
    
    def invert(self: "SudokuImageProcessor") -> None:
        self.image = cv.bitwise_not(self.image)
    
    def to_grayscale(self: "SudokuImageProcessor") -> None:
        self.image = cv.cvtColor(self.image, code=cv.COLOR_BGR2GRAY)
    
    def filter(self: "SudokuImageProcessor") -> None:
        self.image = cv.GaussianBlur(self.image, ksize=(3, 3), sigmaX=0)
    
    def threshold(self: "SudokuImageProcessor") -> None:
        self.image = cv.threshold(self.image, thresh=0, maxval=255, type=cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
    
    def morph(self: "SudokuImageProcessor") -> None:
        kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(3, 3))
        self.image = cv.morphologyEx(self.image, op=cv.MORPH_OPEN, kernel=kernel)
    
    def get_grid_corners(self: "SudokuImageProcessor") -> None:
        contours = cv.findContours(self.image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)[0]
        grid_contour = sorted(contours, key=cv.contourArea, reverse=True)[0]
        brx, bry, brw, brh = cv.boundingRect(grid_contour)

        brect_corners = [[brx, bry], [brx, bry + brh], [brx + brw, bry], [brx + brw, bry + brh]]
        quad_corners = cv.approxPolyN(grid_contour, nsides=4, epsilon_percentage=-1, ensure_convex=True)[0]
        sorted_corners = [sorted(quad_corners, key=lambda qc: dist(qc, bc), reverse=False)[0] for bc in brect_corners]

        self.grid_contour = grid_contour
        self.grid_corners = sorted_corners
    
    def warp_to(self: "SudokuImageProcessor", from_corners: Sequence[tuple[int, int]]) -> None:
        #dstx = max(dist(from_corners[0], from_corners[2]), dist(from_corners[1], from_corners[3]))
        #dsty = max(dist(from_corners[0], from_corners[1]), dist(from_corners[2], from_corners[3]))
        dstx = min(dist(from_corners[0], from_corners[2]), dist(from_corners[1], from_corners[3]))
        dsty = min(dist(from_corners[0], from_corners[1]), dist(from_corners[2], from_corners[3]))
        dst_corners = [[0, 0], [0, dsty], [dstx, 0], [dstx, dsty]]

        #inter_method = cv.INTER_CUBIC
        inter_method = cv.INTER_AREA

        transform_matrix: MatLike = cv.getPerspectiveTransform(src=np.array(from_corners, np.float32), dst=np.array(dst_corners, np.float32))
        self.image = cv.warpPerspective(self.image, M=transform_matrix, dsize=(int(dstx), int(dsty)), flags=inter_method)
    
    def resize_to(self: "SudokuImageProcessor", dest_size: Size) -> None:
        ih, iw = self.image.shape[:2]
        if ih <= 0 or iw <= 0:
            self.image = np.zeros((dest_size[1], dest_size[0]), dtype=np.uint8)
            return
        src_area, dest_area = ih * iw, dest_size[0] * dest_size[1]

        if src_area < dest_area:
            self.image = cv.resize(self.image, dsize=dest_size, interpolation=cv.INTER_AREA)
        else:
            self.image = cv.resize(self.image, dsize=dest_size, interpolation=cv.INTER_CUBIC)
    
    def get_cell_corners(self: "SudokuImageProcessor") -> None:
        corner_map = cv.cornerHarris(self.image, blockSize=5, ksize=5, k=0.08)
        cmh, cmw = corner_map.shape[:2]
        corner_threshold = corner_map.max() * 0.01
        dist_threshold = dist([0, 0], [cmw, cmh]) * 0.01
        
        harris_corners = [[x, y] for x in range(0, cmw) for y in range(0, cmh) if corner_map[y, x] >= corner_threshold]
        ideal_corners = [[(x / 9) * cmw, (y / 9) * cmh] for x in range(0, 10) for y in range(0, 10)]
        actual_corners = [np.array([hc for hc in harris_corners if dist(hc, ic) <= dist_threshold] or [ic], np.float32).mean(axis=0).tolist() for ic in ideal_corners]

        self.harris_corners = harris_corners
        self.ideal_corners = ideal_corners
        self.actual_corners = actual_corners
    
    def get_cell_brect(self: "SudokuImageProcessor") -> None:
        cell_contours = cv.findContours(self.image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]
        ih, iw = self.image.shape
        diagonal_dist_threshold = dist([0, 0], [iw, ih]) * 0.10
        image_center = [iw / 2, ih / 2]

        for contour in cell_contours:
            brx, bry, brw, brh = cv.boundingRect(contour)
            square_side = max(brw, brh)
    
            contour_moments = cv.moments(contour)
            contour_center = [contour_moments['m10'] / contour_moments['m00'], contour_moments['m01'] / contour_moments['m00']] if contour_moments['m00'] != 0 else [0, 0]
    
            brect_area =  brw * brh
            total_area = ih * iw
            if brect_area < total_area * 0.10: # Eğer alan %10'dan küçükse, atla
                continue
            elif brect_area > total_area * 0.75: # Eğer alan %75'den büyükse, atla
                continue
            elif dist(contour_center, image_center) > diagonal_dist_threshold: # Eğer resmin orta noktasında çok uzaksa, atla
                continue
            else:
                x = brx - (square_side - brw) // 2
                y = bry - (square_side - brh) // 2

                self.image = self.image[y:y+square_side, x:x+square_side]
                self.resize_to((20, 20))
                self.image = cv.copyMakeBorder(self.image, 4, 4, 4, 4, cv.BORDER_CONSTANT, value=0)
                return
        
        self.image = None