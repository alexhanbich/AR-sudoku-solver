import numpy as np

class SolveSudoku():

    def __init__(self, vals, coors):
        self.grid = self._build_grid(vals, coors)


    def _build_grid(self, vals, coors):
        sudoku_digits = np.zeros((9,9), dtype=int)
        for i in range(len(coors)):
            x_coor, y_coor = coors[i]
            sudoku_digits[x_coor][y_coor] = vals[i]
        return sudoku_digits.T


    def get_grid(self):
        return self.grid

    
    def is_valid_row(self, row, num):
        for col in range(9):
            if self.grid[row][col] == num:
                return False
        return True


    def is_valid_col(self, col, num):
            for row in range(9):
                if self.grid[row][col] == num:
                    return False
            return True


    def is_valid_box(self, row, col, num):
        s_row = (row//3)*3
        s_col = (col//3)*3
        for row in range(s_row, s_row+3):
            for col in range(s_col, s_col+3):
                if self.grid[row][col] == num:
                    return False
        return True
    

    def is_valid(self, row, col, num):
        return self.is_valid_row(row, num) and self.is_valid_col(col, num) and self.is_valid_box(row, col, num)
        

    def solve(self):
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == 0:
                    for num in range(1, 10):
                        if self.is_valid(row, col, num):
                            self.grid[row][col] = num
                            if self.solve():
                                return True
                            else:
                                self.grid[row][col] = 0
                    return False
        return True