import json
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from PIL import Image
import os

# Print current working directory to check where the script is being run from
print("Current working directory:", os.getcwd())

# Load the JSON data from a file with correct path
with open('data/correspondance.json', 'r') as f:  # Ensure the file is in the 'data' folder
    json_data = json.load(f)

# Extract points from the loaded JSON
im1_points = np.array(json_data["im1Points"])
im2_points = np.array(json_data["im2Points"])

# Compute the midway shape (average of the two point sets)
midway_points = (im1_points + im2_points) / 2

# Compute Delaunay triangulation on the midway shape
tri = Delaunay(midway_points)

# Save the triangulation for morphing purposes
triangulation_data = {
    "midway_points": midway_points.tolist(),  # Convert to list for JSON compatibility
    "simplices": tri.simplices.tolist()  # Save the triangular mesh
}

# Save the data to a JSON file
with open('data/triangulation_morph.json', 'w') as f:
    json.dump(triangulation_data, f)

print("Triangulation data saved successfully for morphing.")

# Create a figure with two subplots for the two images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Load the two images
img1 = Image.open("images/img1-1.png")
img2 = Image.open("images/img2.png")

# Show first image with triangulation
ax1.imshow(img1)
ax1.set_title("Image 1 with Triangulation")
for simplex in tri.simplices:
    ax1.plot(im1_points[simplex, 0], im1_points[simplex, 1], 'r-', linewidth=1)

# Show second image with triangulation
ax2.imshow(img2)
ax2.set_title("Image 2 with Triangulation")
for simplex in tri.simplices:
    ax2.plot(im2_points[simplex, 0], im2_points[simplex, 1], 'r-', linewidth=1)

plt.tight_layout()

# Save the figure to a file before showing it
plt.savefig('images/triangles.png')

# Show the figure after saving it
plt.show()
