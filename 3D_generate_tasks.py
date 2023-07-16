import random
import glob
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

TASK_NUMBER_FOR_SINGLE_MAP = 10
MAPS_DIRECTORY = f'/home/czarek/mgr/3D_maps/blanks/*.npy'


def get_blank_maps_list() -> list:
    maps_dir = MAPS_DIRECTORY
    maps_list = sorted(glob.glob(maps_dir))
    return maps_list


def iterate_over_all_maps(blank_maps: list):
    for map_path in blank_maps:
        occ_map = np.load(map_path)
        for i in range(TASK_NUMBER_FOR_SINGLE_MAP):
            map_path_to_save = draw_start_and_finish(occ_map, map_path, i)
            np.save(map_path_to_save, occ_map)


def draw_start_and_finish(occ_map: np.array, path: str, number: int) -> str:
    dimensions = occ_map.shape

    x_start, y_start, z_start = 0, 0, 0
    while occ_map[y_start, x_start, z_start] == 255:
        x_start = random.randint(0, dimensions[0] - 1)
        y_start = random.randint(0, dimensions[1] - 1)
        z_start = random.randint(0, dimensions[2] - 1)
        if occ_map[y_start, x_start, z_start] != 255:
            break

    x_finish, y_finish, z_finish = 0, 0, 0
    dst = 0.0
    while occ_map[y_finish, x_finish, z_finish] == 255 or dst < 10.0:
        x_finish = random.randint(0, dimensions[0] - 1)
        y_finish = random.randint(0, dimensions[1] - 1)
        z_finish = random.randint(0, dimensions[2] - 1)
        dst = distance.euclidean((x_start, y_start, z_start), (x_finish, y_finish, z_finish))
        if occ_map[y_start, x_start, z_start] != 255 and dst >= 10.0:
            break

    new_path = add_to_string(path, 'sx', str(x_start))
    new_path = add_to_string(new_path, 'sy', str(y_start))
    new_path = add_to_string(new_path, 'sz', str(z_start))
    new_path = add_to_string(new_path, 'fx', str(x_finish))
    new_path = add_to_string(new_path, 'fy', str(y_finish))
    new_path = add_to_string(new_path, 'fz', str(z_finish))

    new_path = add_to_string(new_path, 'path', str(number))
    new_path = new_path.replace('blanks', 'start_finish')

    # visualize_start_finish(occ_map, new_path, x_start, y_start, z_start, x_finish, y_finish, z_finish)

    return new_path


def visualize_start_finish(occ_map: np.array, path: str, x_start: int, y_start: int, z_start: int, x_finish: int, y_finish: int, z_finish: int):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a grid of coordinates
    xx, yy, zz = np.meshgrid(np.arange(occ_map.shape[1]+1), np.arange(occ_map.shape[0]+1), np.arange(occ_map.shape[2]+1))

    # Set the color values based on occupancy
    colors = np.zeros(occ_map.shape + (3,))
    colors[occ_map == 0] = [1, 1, 1]  # Set unoccupied places to white
    colors[occ_map == 255] = [0, 0, 0]  # Set occupied places to black

    # Plot the 3D occupancy map
    ax.voxels(xx, yy, zz, occ_map.astype(bool), facecolors=colors)

    # Plot the start and finish positions
    ax.scatter(x_start, y_start, z_start, c='g', marker='o', s=30, label='Start')
    ax.scatter(x_finish, y_finish, z_finish, c='r', marker='o', s=30, label='Finish')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Occupancy Map')
    ax.legend()
    plt.show()


def add_to_string(path: str, where: str, to_add: str) -> str:
    start_index = path.find(where)
    end_index = start_index + len(where)

    new_path = path[:end_index] + to_add + path[end_index:]

    return new_path


def generate_tasks():
    maps = get_blank_maps_list()
    iterate_over_all_maps(maps)


if __name__ == '__main__':
    generate_tasks()
