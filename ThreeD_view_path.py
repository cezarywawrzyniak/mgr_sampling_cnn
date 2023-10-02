import sys
import numpy as np
import matplotlib.pyplot as plt
import os

if len(sys.argv) != 2:
    print("python script.py folder_path")
    sys.exit(1)

folder_path = sys.argv[1]

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

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
indices = np.nonzero(occ_map)
print(indices)
ax.scatter(indices[0], indices[1], indices[2], c='b', marker='o')
# plot limits based on the image dimensions
ax.set_xlim(0, occ_map.shape[0])
ax.set_ylim(0, occ_map.shape[1])
ax.set_zlim(0, occ_map.shape[2])

plt.show()
