
    # Generate Random Colorings:
    #     Define a set of colors: {Red, Blue, Yellow, Green}.
    #     Implement a function to randomly select a color from the set.

    # Generate a 20x20 Grid:
    #     Initialize a 20x20 grid with all cells initially uncolored.
    #     Implement functions to randomly select rows and columns based on the provided logic.
    #     Apply the color to the selected rows and columns according to the instructions.

    # Encode the Grid:
    #     Use one-hot encoding to represent the colors. Assign a unique one-hot vector to each color.

    # Generate Labels:
    #     Based on your logic, determine whether the generated grid is safe or dangerous.
    #     Assign labels accordingly.

    # Repeat for Training Data:
    #     Repeat steps 2-4 to generate a sufficient amount of training data.
    #     Ensure a diverse set of grids with various colorings and patterns.
 
import random
import numpy as np


colorsForRow = ["R", "B", "Y", "G"]
colorsForCol = ["R", "B", "Y", "G"]
sizeForRow = list(range(0,20))
sizeForCol = list(range(0,20))

# Create a dictionary to map colors to one-hot vectors
color_to_one_hot = {
    "R": (0, 0, 0, 0, 1),
    "B": (0, 0, 0, 1, 0),
    "Y": (0, 0, 1, 0, 0),
    "G": (0, 1, 0, 0, 0),
    "U": (1, 0, 0, 0, 0)
}

def generate_grid(grid_size):
    grid = np.full((grid_size, grid_size), "U", dtype=str)
    return grid

def getRandomHotColor(random_color):
    # getting the random hot vector color
    random_hot_vector = color_to_one_hot[random_color]
    return random_hot_vector

def RowColoring(grid, colorsForRow):
    row_index = random.choice(sizeForRow)
    random_color = random.choice(colorsForRow)     
    colorsForRow.remove(random_color)
    sizeForRow.remove(row_index)
    grid[row_index, :] = random_color
    return grid

def ColumnColoring(grid, colorsForCol):
    col_index = random.choice(sizeForCol)
    random_color = random.choice(colorsForCol)     
    colorsForCol.remove(random_color)
    sizeForCol.remove(col_index)
    grid[:, col_index] = random_color
    return grid

def encode_grid(grid, color_to_one_hot):
    # Flatten the grid and encode each color using one-hot vectors
    encoded_vector = []

    for row in grid:
        for cell in row:
            encoded_vector.extend(color_to_one_hot[cell])

    return np.array(encoded_vector)

# generating the grid
grid = generate_grid(20)
# filling the first row
grid = RowColoring(grid, colorsForRow)
# filling first col
grid = ColumnColoring(grid,colorsForCol)
# filling second row
grid = RowColoring(grid, colorsForRow)
# filling second col
grid = ColumnColoring(grid, colorsForCol)

vector = encode_grid(grid, color_to_one_hot)
np.set_printoptions(threshold=np.inf)
print(vector)
