# Class for sparse G-matrix

class Sparse_G_matrix:
    def __init__(self, G_dict):
        self.rows = 0
        self.cols = 0
        self.data = []

        for (row,col), value in G_dict.items():
            self.data.append(((row, col), value))
            self.rows = max(self.rows, row+1)
            self.cols = max(self.cols, col+1)
    
    def display(self):
        for val in self.data:
            print(val)