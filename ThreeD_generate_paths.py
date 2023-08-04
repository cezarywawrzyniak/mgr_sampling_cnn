import glob
import numpy as np
import heapq
import math
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab


MAPS_DIRECTORY = f'/home/czarek/mgr/3D_maps/start_finish/*.npy'


def astar(image: np.array, start: tuple, finish: tuple) -> list or None:
    img = np.array(image)
    open_list = []
    closed_list = set()
    parent = {}
    g = {start: 0}
    h = {start: heuristic(start, finish)}
    f = {start: h[start]}
    heapq.heappush(open_list, (f[start], start))

    while open_list:
        current = heapq.heappop(open_list)[1]
        if current == finish:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path
        closed_list.add(current)
        for neighbor in neighbors(current, img):
            if neighbor in closed_list or img[neighbor] == 255:
                continue
            initial_g = g[current] + cost(current, neighbor)
            if neighbor not in g or initial_g < g[neighbor]:
                parent[neighbor] = current
                g[neighbor] = initial_g
                h[neighbor] = heuristic(neighbor, finish)
                f[neighbor] = g[neighbor] + h[neighbor]
                heapq.heappush(open_list, (f[neighbor], neighbor))
    return None


def neighbors(pos: tuple, img: np.array) -> list:
    height, width, depth = img.shape
    result = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 255 and dy == 255 and dz == 255:
                    continue
                x, y, z = pos[0] + dx, pos[1] + dy, pos[2] + dz
                if x < 0 or x >= height or y < 0 or y >= width or z < 0 or z >= depth:
                    continue
                if dx != 255 and dy != 255 and dz != 255:
                    # Check for obstacles in diagonal paths
                    if img[x, pos[1], pos[2]] == 255 or img[pos[0], y, pos[2]] == 255 or img[pos[0], pos[1], z] == 255:
                        continue
                if dx != 255 and dy != 255:
                    # Check for obstacles in diagonal paths in the xy plane
                    if img[x, pos[1], pos[2]] == 255 or img[pos[0], y, pos[2]] == 255:
                        continue
                if dx != 255 and dz != 255:
                    # Check for obstacles in diagonal paths in the xz plane
                    if img[x, pos[1], pos[2]] == 255 or img[pos[0], pos[1], z] == 255:
                        continue
                if dy != 255 and dz != 255:
                    # Check for obstacles in diagonal paths in the yz plane
                    if img[pos[0], y, pos[2]] == 255 or img[pos[0], pos[1], z] == 255:
                        continue
                if img[x, y, z] != 255:
                    result.append((x, y, z))
    return result


def heuristic(x: tuple, y: tuple) -> float:
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)


def cost(a: tuple, b: tuple) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = abs(a[2] - b[2])

    if dx == 1 and dy == 1 and dz == 1:
        # Diagonal movement in all three dimensions
        return math.sqrt(3)
    elif (dx == 1 and dy == 1) or (dx == 1 and dz == 1) or (dy == 1 and dz == 1):
        # Diagonal movement in two dimensions
        return math.sqrt(2)
    else:
        # Horizontal or vertical movement
        return 1.0


def get_blank_maps_list() -> list:
    maps_dir = MAPS_DIRECTORY
    maps_list = sorted(glob.glob(maps_dir))
    return maps_list


def get_start_finish_coordinates(path: str) -> tuple:
    x_start = int(get_from_string(path, "_sx", "_sy"))
    y_start = int(get_from_string(path, "_sy", "_sz"))
    z_start = int(get_from_string(path, "_sz", "_fx"))
    x_finish = int(get_from_string(path, "_fx", "_fy"))
    y_finish = int(get_from_string(path, "_fy", "_fz"))
    z_finish = int(get_from_string(path, "_fz", ".npy"))

    return (x_start, y_start, z_start), (x_finish, y_finish, z_finish)


def get_from_string(path: str, start: str, finish: str) -> str:
    start_index = path.find(start)
    end_index = path.find(finish)

    substring = path[start_index+3:end_index]

    return substring


def generate_paths():
    maps = get_blank_maps_list()
    for map_path in maps:
        print(map_path)
        occ_map = np.load(map_path)
        start, finish = get_start_finish_coordinates(map_path)
        path = astar(occ_map, start, finish)
        if path:
            save_and_visualize_path(occ_map=occ_map, path=path, directory=map_path, visualize=False, save=True)
            # print(path)
        else:
            print("COULDN'T FIND A PATH FOR THIS EXAMPLE:", map_path)
            os.remove(map_path)


def save_and_visualize_path(occ_map: np.array, path: list, directory: str, visualize: bool, save: bool):
    if save:
        dir_save_no_points_map = directory.replace('start_finish', 'paths')
        # Create a copy of the original map
        path_array_to_save = np.zeros_like(occ_map)

        # Mark the path on an empty array
        for position in path:
            path_array_to_save[position] = 255

        # Save the visualized image as a .npy file
        np.save(dir_save_no_points_map, path_array_to_save)

        if visualize:
            # Create 3D scatter plot of the path
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            indices = np.nonzero(path_array_to_save)
            ax.scatter(indices[0], indices[1], indices[2], c='b', marker='o')
            # Set plot limits based on the image dimensions
            ax.set_xlim(0, path_array_to_save.shape[0])
            ax.set_ylim(0, path_array_to_save.shape[1])
            ax.set_zlim(0, path_array_to_save.shape[2])

            plt.show()

    if visualize:
        # Create 3D scatter plot of the path
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        visualized_image = np.array(occ_map)
        for position in path:
            visualized_image[position] = 254  # Set a different value for the path

        # Obtain the indices of the path positions
        indices = np.nonzero(visualized_image == 255)
        ax.scatter(indices[0], indices[1], indices[2], c='b', marker='o', alpha=0.01)
        # ax.voxels(visualized_image == 255, facecolors='b', edgecolors='k', alpha=0.1)

        path_indices = np.nonzero(visualized_image == 254)
        ax.scatter(path_indices[0], path_indices[1], path_indices[2], c='r', marker='o')

        # Set plot limits based on the image dimensions
        ax.set_xlim(0, visualized_image.shape[0])
        ax.set_ylim(0, visualized_image.shape[1])
        ax.set_zlim(0, visualized_image.shape[2])

        plt.show()


if __name__ == '__main__':
    generate_paths()
