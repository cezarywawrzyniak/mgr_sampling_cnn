import glob
import cv2
import numpy as np

from astar import astar


def load_coords_from_path():
    map_path = '/home/czarek/mgr/maps/start_finish/map_0_path0_sx192_sy148_fx6_fy308.png'
    x_start = 192
    y_start = 148
    start = (x_start, y_start)
    x_finish = 6
    y_finish = 308
    finish = (x_finish, y_finish)

    occ_map = cv2.imread(map_path, 0)
    neg_map = 255 - occ_map
    print(occ_map)
    path = astar(neg_map, start, finish)
    print(path)
    visualize_path(occ_map=occ_map, path=path)


def visualize_path(occ_map: np.array, path: list):
    map_with_path = cv2.cvtColor(occ_map, cv2.COLOR_GRAY2BGR)
    for point in path:
        map_with_path[point[1], point[0]] = (255, 0, 0)

    cv2.imshow("Occupancy Map", map_with_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    load_coords_from_path()
