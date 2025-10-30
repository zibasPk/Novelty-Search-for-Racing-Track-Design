import json
import matplotlib as plt
import os

# Define the path to the JSON file
json_file_path = os.path.join(os.getcwd(), '1725.json')
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Extract points from the JSON dataSet
points = [(item['x'], item['y']) for item in data.get('dataSet', [])]

# Separate x and y coordinates
x_coords = [point[0] for point in points]
y_coords = [point[1] for point in points]

# Plot the points using matplotlib
plt.figure(figsize=(8, 8))
plt.scatter(x_coords, y_coords, c='blue', marker='o')
plt.title('Track Points Visualization')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.grid(True)
plt.axis('equal')
plt.show()