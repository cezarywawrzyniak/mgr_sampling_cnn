import numpy as np
import glob

LAYERS = 5
MAPS_DIRECTORY = f'/home/czarek/mgr/3D_maps/one_example/paths/*.npy'


def is_point_valid(point, occ_map):
    if 0 <= point[0] < occ_map.shape[0] and 0 <= point[1] < occ_map.shape[1] and 0 <= point[2] < occ_map.shape[2]:
        if is_point_in_free_space(point, occ_map):
            return True
        else:
            return False
    else:
        return False


def is_point_in_free_space(point, occ_map):
    if occ_map[point] == 255:
        return False
    else:
        return True


def enlarge_the_path(occ_path, path_path):
    # Load the occupancy map and path
    occ_map = np.load(occ_path)
    path = np.load(path_path)

    path_indices = np.nonzero(path == 255)

    path_xs = path_indices[0]
    path_ys = path_indices[1]
    path_zs = path_indices[2]

    new_path = np.array(path)

    for path_point in range(0, len(path_xs)):
        og_x = path_xs[path_point]
        og_y = path_ys[path_point]
        og_z = path_zs[path_point]
        # print(og_x, og_y, og_z)
        # new_points = []

        path_value = 255
        for i in range(LAYERS):
            # Generate points in the current layer
            # print(i)
            path_value -= 255/LAYERS-2
            # print(path_value)
            layer_points = []
            for x in range(-i, i + 1):
                for y in range(-i, i + 1):
                    for z in range(-i, i + 1):
                        # Skip the points in the inner layers
                        if abs(x) < i and abs(y) < i and abs(z) < i:
                            continue
                        layer_points.append(tuple(np.array([og_x, og_y, og_z]) + np.array([x, y, z])))
            for new_point in layer_points:
                if is_point_valid(new_point, occ_map):
                    # print(new_path[new_point], path_value)
                    if path_value > new_path[new_point]:
                        new_path[new_point] = path_value

    np.save(path_path, new_path)


def get_blank_maps_list() -> list:
    maps_dir = MAPS_DIRECTORY
    maps_list = sorted(glob.glob(maps_dir))
    return maps_list


def generate_paths():
    maps = get_blank_maps_list()
    for path_path in maps:
        occ_path = path_path.replace('paths', 'start_finish')
        # print(occ_path)
        # print(path_path)
        enlarge_the_path(occ_path, path_path)


if __name__ == '__main__':
    generate_paths()
