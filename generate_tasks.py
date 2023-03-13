import random

import glob
import cv2
import numpy as np
from scipy.spatial import distance

TASK_NUMBER_FOR_SINGLE_MAP = 10
MAPS_DIRECTORY = f'/home/czarek/mgr/maps/blanks/*.png'


def get_blank_maps_list() -> list:
    maps_dir = MAPS_DIRECTORY
    maps_list = sorted(glob.glob(maps_dir))
    return maps_list


def iterate_over_all_maps(blank_maps: list):
    for map_path in blank_maps:
        occ_map = cv2.imread(map_path, 0)
        for i in range(TASK_NUMBER_FOR_SINGLE_MAP):
            map_path_to_save = draw_start_and_finish(occ_map, map_path, i)
            # print(map_path_to_save)
            cv2.imwrite(map_path_to_save, occ_map)


def draw_start_and_finish(occ_map: np.array, path: str, number: int) -> str:
    dimensions = occ_map.shape

    x_start = 0
    y_start = 0
    while occ_map[y_start, x_start] == 0:
        x_start = random.randint(0, dimensions[0] - 1)
        y_start = random.randint(0, dimensions[1] - 1)
        if occ_map[y_start, x_start] != 0:
            break

    x_finish = 0
    y_finish = 0
    dst = 0.0
    while occ_map[y_finish, x_finish] == 0 or dst < 75.0:
        x_finish = random.randint(0, dimensions[0] - 1)
        y_finish = random.randint(0, dimensions[1] - 1)
        dst = distance.euclidean((x_start, y_start), (x_finish, y_finish))
        if occ_map[y_finish, x_finish] != 0 and dst >= 75.0:
            break

    new_path = add_to_string(path, 'sx', str(x_start))
    new_path = add_to_string(new_path, 'sy', str(y_start))
    new_path = add_to_string(new_path, 'fx', str(x_finish))
    new_path = add_to_string(new_path, 'fy', str(y_finish))

    new_path = add_to_string(new_path, 'path', str(number))
    new_path = new_path.replace('blanks', 'start_finish')

    visualize_start_finish(occ_map, new_path, x_start, y_start, x_finish, y_finish)

    return new_path


def visualize_start_finish(occ_map: np.array, path: str, x_start: int, y_start: int, x_finish: int, y_finish: int):
    color_path = path.replace('start_finish', 'start_finish_visualized')

    start_coordinates = (x_start, y_start)
    finish_coordinates = (x_finish, y_finish)
    color_map = cv2.cvtColor(occ_map, cv2.COLOR_GRAY2BGR)

    color_map = cv2.circle(color_map, start_coordinates, 5, (0, 255, 0), -1)
    color_map = cv2.circle(color_map, finish_coordinates, 5, (0, 0, 255), -1)
    # cv2.imshow("Occupancy Map", color_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(color_path, color_map)


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
