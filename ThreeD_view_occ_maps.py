import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# folder_path = sys.argv[1]
folder_path = '/home/czarek/mgr/3D_data/train/images'

if not os.path.isdir(folder_path):
    print("Invalid folder directory.")
    sys.exit(1)

file_list = os.listdir(folder_path)

if len(file_list) == 0:
    print("No files found in the folder.")
    sys.exit(1)

random_file = np.random.choice(file_list)

file_path = os.path.join(folder_path, random_file)
occ_map = np.load(file_path)

# downsample the occupancy map
# downsample_factor = 4  # adjust the factor as desired
# downsampled_occ_map = occ_map[::downsample_factor, ::downsample_factor, ::downsample_factor]

# occupied_mask = downsampled_occ_map.astype(bool)
occupied_mask = occ_map.astype(bool)

# color = np.zeros(downsampled_occ_map.shape + (4,))
color = np.zeros(occ_map.shape + (4,))
color[occupied_mask] = (0, 0, 0, 0.2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
indices = np.nonzero(occ_map)
# plot the occupied places as semi-transparent
ax.voxels(occupied_mask, facecolors=color, edgecolors=color)

# ax.scatter(indices[0], indices[1], indices[2], c='k', marker='o')
# ax.set_xlim(0, occ_map.shape[0]/4)
# ax.set_ylim(0, occ_map.shape[1]/4)
# ax.set_zlim(0, occ_map.shape[2]/4)
ax.set_xlim(0, occ_map.shape[0])
ax.set_ylim(0, occ_map.shape[1])
ax.set_zlim(0, occ_map.shape[2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Occupancy Map')
plt.show()
