import numpy as np

LAYERS = 5


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


file_name = 'map_0_path0_sx0_sy0_sz0_fx50_fy42_fz103'

occ_path = f'/home/czarek/mgr/3D_maps/one_example/start_finish/{file_name}.npy'
path_path = f'/home/czarek/mgr/3D_maps/one_example/paths/{file_name}.npy'

# Load the occupancy map and path
occ_map = np.load(occ_path)
path = np.load(path_path)

path_indices = np.nonzero(path == 255)

path_xs = path_indices[0]
path_ys = path_indices[1]
path_zs = path_indices[2]

obstacles_indices = np.nonzero(occ_map == 255)
obstacles_xs = obstacles_indices[0]
obstacles_ys = obstacles_indices[1]
obstacles_zs = obstacles_indices[2]

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
        print(i)
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
                print(new_path[new_point], path_value)
                if path_value > new_path[new_point]:
                    new_path[new_point] = path_value

np.save(path_path, new_path)

