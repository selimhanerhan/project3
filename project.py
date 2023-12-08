
    # Generate a 20x20 Grid:
    
    # Encode the Grid:
    #     Use one-hot encoding to represent the colors. Assign a unique one-hot vector to each color.

    # Loss Function
    #   Use the binary cross entropy
    #   -y_i * ln(f(x_i)) - (1-y_i) * ln(1 - f(x_i))
    
    # Output space
    #  y -> 0 or 1 (safe or dangerous)
    
    # Model Space
    #   Linear regression
    #   f(x) = sigmoid( w.x ) w is a vector of size 1600
    
    # Training algorithm
    #   SGD
    #   x = (1, x_1, x_2, x_3)
    #   general linear function
    #   x_1 * w_1 + x_2 * w_2 + x_3 * w_3
    
    # roadmap
    #     Prepare Data:
    #     Flatten each 20x20 grid into a 1D array. Each flattened array becomes a sample in your dataset.
    #     Label each sample based on the known logic of whether the grid is safe or dangerous.

    # Initialize Weights:
    #     Initialize the weight vector (ww) with small random values. The size of the weight vector should match the length of your flattened input vector.

    # Define Model:
    #     Define the linear regression model, which computes predictions (fw(x)fw(x)) as the dot product of the weights and inputs.

    # Loss Function:
    #     Choose a loss function appropriate for your task. For binary classification with linear regression, mean squared error might be used, but other metrics like cross-entropy may also be considered.

    # Gradient Descent:
    #     Implement a gradient descent algorithm to iteratively update the weights in the direction that reduces the loss.

    # Training Loop:
    #     Iterate over your dataset, computing predictions, calculating the loss, and updating weights using gradient descent.
    
    
 
import random
import numpy as np



numSamples = 1000

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

allGrids = []
## when we create the data does every data need to be unique?

for i in range(numSamples):
    colorsForRow = ["R", "B", "Y", "G"]
    colorsForCol = ["R", "B", "Y", "G"]
    sizeForRow = list(range(0,20))
    sizeForCol = list(range(0,20))
    
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

    allGrids.append(grid)

np.set_printoptions(threshold=np.inf)
#print(allGrids[0])
redCount = 0
yellowCount = 0

## Labeling the data whether its safe or dangerous
### red count needs to be equal to 2
#### check left if left is R and right is R then its dangerous
#### check top if top is R and if the bottom is R then its dangerous
# Function to check if a red wire is laid before a yellow wire


# Function to label the grid as safe or dangerous
def label_grid(grid):
    if is_dangerous(grid):
        return "Dangerous"
    else:
        return "Safe"

gridLabels = {}
for grid in allGrids:
    label = label_grid(grid)
    gridLabels[str(grid)] = label



## MODEL SPACE

numFeatures = len(gridLabels)

weightVector = np.random.randn(numFeatures, 1)

# f(x_1,x_2,...,x_d) = sigmoid(w_0 + x_1 w_1 + x_2 w_2 + ... + w_d x_d)
def predict(inputVector, weightVector):
    result = np.dot(inputVector, weightVector)
    return sigmoid(result)

# it turns the result of dot product into a probability thats between 0 to 1.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# loss = -y_i * log(f(x_i)) - (1-y_i) * log(1 - f(x_i))
def binary_cross_entropy_loss(yTrue, yPredicted):
    # small constant to avoid log(0)
    epsilon = 1e-15 
    loss = -yTrue * np.log(np.clip(yPredicted, epsilon, 1 - epsilon) - (1 - yTrue) * np.log(np.clip(1 - yPredicted, epsilon, 1 - epsilon)))
    return loss


# not too sure about this but it should represent the gradient as below
# −y_i ​log(f(x_i​))−(1−y_i​)log(1−f(x_i​))
def gradient_binary_cross_entropy(yTrue, yPredicted, x_j):
    gradient = -yTrue * (1 - yPredicted) * x_j + (1 - yTrue) * yPredicted * x_j
    return gradient

