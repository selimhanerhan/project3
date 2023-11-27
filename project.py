import numpy as np
import random

# Function to generate a 20x20 wiring diagram with history and labels
def generate_wiring_diagram():
    # Initialize a 20x20 grid with placeholder values
    wiring_diagram = np.full((20, 20), '', dtype='<U6')

    # List of colors to choose from
    colors = ["R", "B", "Y", "G"]

    # Initialize previous values and labels
    prev_row = np.full((20, 1), '', dtype='<U6')
    prev_col = np.full((1, 20), '', dtype='<U6')
    label = ''

    # Iterate until all cells are filled
    while '' in wiring_diagram:
        # Step 1: Starting with rows
        row1 = random.randint(0, 19)
        color1 = random.choice(colors)
        wiring_diagram[row1, :] = color1
        label = 'Dangerous' if 'R' in prev_row[row1] and 'Y' in prev_row[row1] and prev_row[row1].tolist().index('R') < prev_row[row1].tolist().index('Y') else 'Safe'
        print(wiring_diagram)
        prev_row[row1][0] = color1

        # Step 2: Switching to columns
        col2 = random.randint(0, 19)
        remaining_colors = np.array([c for c in colors if c != color1])
        color2 = np.random.choice(remaining_colors)
        wiring_diagram[:, col2] = color2
        prev_col[0, col2] = color2

        # Step 3: Switching back to rows
        row3 = random.randint(0, 19)
        remaining_colors = np.array([c for c in colors if c not in [color1, color2]])
        color3 = np.random.choice(remaining_colors)
        wiring_diagram[row3, :] = color3
        prev_row[row3][0] = color3

        # Step 4: Switching back to columns
        col4 = random.randint(0, 19)
        remaining_colors = np.array([c for c in colors if c != color3])
        color4 = remaining_colors[0]  # Take the remaining color
        wiring_diagram[:, col4] = color4
        prev_col[0, col4] = color4

    return wiring_diagram, label

# Example usage
generated_diagram, label = generate_wiring_diagram()

# Display the generated wiring diagram and label
print("Wiring Diagram:")
print(generated_diagram)
print("\nLabel:", label)
