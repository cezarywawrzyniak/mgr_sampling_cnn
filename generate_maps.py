import random

import cv2
import numpy as np


def create_map(width: int, height: int) -> np.array:
    # create white image
    occ_map = np.zeros((height, width, 1), np.uint8)
    occ_map.fill(255)
    # make black borders
    occ_map[0, :] = 0
    occ_map[:, 0] = 0
    occ_map[-1, :] = 0
    occ_map[:, -1] = 0

    # fill with obstacles
    for i in range(0, random.randint(2, 7)):
        occ_map = add_obstacle(occ_map, 75, 350, 25, 75)  # long in x

    for i in range(0, random.randint(2, 7)):
        occ_map = add_obstacle(occ_map, 25, 75, 75, 350)  # long in y

    for i in range(0, random.randint(1, 3)):
        occ_map = add_obstacle(occ_map, 75, 175, 75, 175)  # symmetrical big

    for i in range(0, random.randint(5, 15)):
        occ_map = add_obstacle(occ_map, 25, 125, 25, 125)  # symmetrical small

    return occ_map


def add_obstacle(occ_map: np.array, x_min: int, x_max: int, y_min: int, y_max: int) -> np.array:
    # get random placement
    x_start = random.randint(0, 512)
    y_start = random.randint(0, 512)
    # get random size
    x_end = random.randint(x_start + x_min, x_start + x_max)
    y_end = random.randint(y_start + y_min, y_start + y_max)

    # add obstacle
    cv2.rectangle(occ_map, pt1=(x_start, y_start), pt2=(x_end, y_end), color=0, thickness=-1)

    return occ_map


def generate_maps():
    for map_no in range(0, 10):  # CHANGE NUMBER OF GENERATED MAPS
        save_path = f'/home/czarek/mgr/maps/blanks/map_{map_no}_path_sx_sy_fx_fy.png'
        occ_map = create_map(512, 512)
        show_map(occ_map)  # SHOW
        cv2.imwrite(save_path, occ_map)


def show_map(occ_map: np.array):
    cv2.imshow("Occupancy Map", occ_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    generate_maps()
