import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import random


def get_random_file(directory_path):
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        raise ValueError("Invalid directory path provided.")

    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    if not files:
        return None  # The directory is empty

    random_file = random.choice(files)
    file_name_without_extension = os.path.splitext(random_file)[0]
    return file_name_without_extension


if len(sys.argv) != 2:
    print("Usage: python script.py file_name")
    sys.exit(1)

# Get the filename from command-line argument
# file_name = sys.argv[1]
file_name = get_random_file('/home/czarek/mgr/3D_eval_data/images/')
# file_name = 'map_4_path2_sx6_sy27_sz32_fx46_fy65_fz34'
print(file_name)

occ_path = f'/home/czarek/mgr/3D_eval_data/images/{file_name}.npy'
path_path = f'/home/czarek/mgr/3D_eval_data/masks/{file_name}.npy'

# Load the occupancy map and path
occ_map = np.load(occ_path)
path = np.load(path_path)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
visualized_image = np.array(occ_map)

# Obtain the indices of the path positions
indices = np.nonzero(visualized_image == 255)
ax.scatter(indices[0], indices[1], indices[2], c='r', marker='o')

path_indices = np.nonzero(path)
print(path_indices)
colors = path[path_indices]
print(colors)
ax.scatter(path_indices[0], path_indices[1], path_indices[2], c=colors, cmap='jet', marker='o')

# Set plot limits based on the image dimensions
ax.set_xlim(0, visualized_image.shape[0])
ax.set_ylim(0, visualized_image.shape[1])
ax.set_zlim(0, visualized_image.shape[2])

plt.show()
