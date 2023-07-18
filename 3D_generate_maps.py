import random
import numpy as np


def create_map(width: int, height: int, depth: int) -> np.array:
    # Create a white map
    occ_map = np.zeros((height, width, depth), dtype=float)

    # Fill with obstacles, for map size 128: 15 40 75
    for _ in range(random.randint(1, 2)):
        occ_map = add_obstacle(occ_map, 10, 47, 10, 10, 10, 10)  # Long in x

    for _ in range(random.randint(1, 2)):
        occ_map = add_obstacle(occ_map, 10, 10, 10, 47, 10, 10)  # Long in y

    for _ in range(random.randint(1, 2)):
        occ_map = add_obstacle(occ_map, 10, 10, 10, 10, 10, 47)  # Long in z

    for _ in range(random.randint(1, 3)):
        occ_map = add_obstacle(occ_map, 25, 25, 25, 25, 25, 25)  # Symmetrical big

    for _ in range(random.randint(10, 20)):
        occ_map = add_obstacle(occ_map, 10, 10, 10, 10, 10, 10)  # Symmetrical small

    return occ_map


def add_obstacle(occ_map: np.array, x_min: int, x_max: int, y_min: int, y_max: int, z_min: int, z_max: int) -> np.array:
    # Get random placement
    x_start = random.randint(0, occ_map.shape[0]-1)
    y_start = random.randint(0, occ_map.shape[1]-1)
    z_start = random.randint(0, occ_map.shape[2]-1)
    # Get random size
    x_end = random.randint(x_start + x_min, x_start + x_max)
    y_end = random.randint(y_start + y_min, y_start + y_max)
    z_end = random.randint(z_start + z_min, z_start + z_max)

    # Add obstacle
    occ_map[x_start:x_end, y_start:y_end, z_start:z_end] = 255

    return occ_map


def generate_maps():
    for map_no in range(10):  # CHANGE NUMBER OF GENERATED MAPS
        save_path = f'/home/czarek/mgr/3D_maps/blanks/map_{map_no}_path_sx_sy_sz_fx_fy_fz.npy'
        occ_map = create_map(80, 80, 80)  # Update dimensions for width, height, and depth
        np.save(save_path, occ_map)


if __name__ == '__main__':
    generate_maps()
