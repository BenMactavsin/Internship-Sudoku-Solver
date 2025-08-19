import numpy as np
from cv2.typing import MatLike
class SudokuBoard:
    def __init__(self: "SudokuBoard") -> None:
        self.matrix: MatLike = np.zeros((9, 9), dtype=np.integer)

    def clone(self: "SudokuBoard") -> "SudokuBoard":
        clone = SudokuBoard()
        clone.matrix = self.matrix.copy()
        return clone

    def get(self: "SudokuBoard", x: int, y: int) -> int:
        return self.matrix[y, x]
    
    def set(self: "SudokuBoard", x: int, y: int, value: int) -> None:
        self.matrix[y, x] = value
    
    def print_board(self: "SudokuBoard") -> None:
        for x in range(0, 9):
            if x % 3 == 0:
                print("+---+---+---+")

            for y in range(0, 9):
                if y % 3 == 0:
                    print("|", end="")

                print(self.get(x, y), end="")
            
            print("|")
        
        print("+---+---+---+")

    def digit_is_valid(self: "SudokuBoard", x: int, y: int, d: int) -> bool:
        if any([self.get(rx, y) == d for rx in range(0, 9)]):
            return False
        
        if any([self.get(x, cy) == d for cy in range(0, 9)]):
            return False

        for rx in range((x // 3) * 3, ((x // 3) + 1) * 3):
            for cy in range((y // 3) * 3, ((y // 3) + 1) * 3):
                if self.get(rx, cy) == d:
                    return False
        
        return True

    def solve_for(self: "SudokuBoard", x: int, y: int) -> bool:
        nx, ny = x + ((y + 1) // 9), (y + 1) % 9

        if self.get(x, y) != 0:
            if nx >= 9:
                return True
            else:
                return self.solve_for(nx, ny)

        for d in range(1, 10):
            if self.digit_is_valid(x, y, d):
                self.set(x, y, d)

                if nx >= 9:
                    return True
                
                if self.solve_for(nx, ny):
                    return True
                
                self.set(x, y, 0)
        
        return False      


    def solve_for_all(self: "SudokuBoard") -> bool:
        return self.solve_for(0, 0)