class SudokuBoard:
    def __init__(self) -> None:
        self.matrix: list[list[int]] = [[0 for _ in range(9)] for _ in range(9)]

    def set_value(self, row: int, col: int, value) -> None:
        self.matrix[row][col] = value

    def get_value(self, row, col):
        return self.matrix[row][col]
    
    def print_board(self) -> None:
        for row in self.matrix:
            for col in row:
                print(f"{col} ", end="")
            print("\n", end="")

    def solve(self) -> None:
        #TODO: Implement a solving algorithm after make the model read the digits correctly first
        return None