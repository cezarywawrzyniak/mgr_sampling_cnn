import sys
import numpy as np
import matplotlib.pyplot as plt
import os

if len(sys.argv) != 2:
    print("Usage: python script.py folder_directory")
    sys.exit(1)

# Get the folder directory from command-line argument
folder_path = sys.argv[1]

# Check if the folder exists
if not os.path.isdir(folder_path):
    print("Invalid folder directory.")
    sys.exit(1)

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Check if there are any files in the folder
if len(file_list) == 0:
    print("No files found in the folder.")
    sys.exit(1)

# Select a random file from the list
random_file = np.random.choice(file_list)

# Load the selected .npy file
file_path = os.path.join(folder_path, random_file)
occ_map = np.load(file_path)

# Downsample the occupancy map
downsample_factor = 4  # Adjust the factor as desired
downsampled_occ_map = occ_map[::downsample_factor, ::downsample_factor, ::downsample_factor]

# Create a mask for the occupied places
occupied_mask = downsampled_occ_map.astype(bool)

# Create a color array for the occupied places (set to black)
color = np.zeros(downsampled_occ_map.shape + (4,))
color[occupied_mask] = (0, 0, 0, 0.2)  # Set RGB values to (0, 0, 0) and alpha (transparency) to 0.5

# Visualize the 3D occupancy map with semi-transparent objects (without wireframe edges)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the occupied places as semi-transparent cuboids
ax.voxels(occupied_mask, facecolors=color, edgecolors=color)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Occupancy Map')
plt.show()
