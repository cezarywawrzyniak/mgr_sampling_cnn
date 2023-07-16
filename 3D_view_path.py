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

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
indices = np.nonzero(occ_map)
ax.scatter(indices[0], indices[1], indices[2], c='b', marker='o')
# Set plot limits based on the image dimensions
ax.set_xlim(0, occ_map.shape[0])
ax.set_ylim(0, occ_map.shape[1])
ax.set_zlim(0, occ_map.shape[2])

plt.show()
