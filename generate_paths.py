import glob
import cv2
import numpy as np
import heapq
import math
import os

MAPS_DIRECTORY = f'/home/czarek/mgr/maps/start_finish/*.png'


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
            if neighbor in closed_list or img[neighbor] == 0:
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
    height, width = img.shape
    result = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            x, y = pos[0] + dx, pos[1] + dy
            if x < 0 or x >= height or y < 0 or y >= width:
                continue
            if dx != 0 and dy != 0:
                # Check for obstacles in diagonal paths
                if img[x, pos[1]] == 0 or img[pos[0], y] == 0:
                    continue
            if img[x, y] != 0:
                result.append((x, y))
    return result


def heuristic(x: tuple, y: tuple) -> float:
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def cost(a: tuple, b: tuple) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if dx == 1 and dy == 1:
        return math.sqrt(2)
    else:
        return 1.0


def get_blank_maps_list() -> list:
    maps_dir = MAPS_DIRECTORY
    maps_list = sorted(glob.glob(maps_dir))
    return maps_list


def get_start_finish_coordinates(path: str) -> tuple:
    x_start = int(get_from_string(path, "_sx", "_sy"))
    y_start = int(get_from_string(path, "_sy", "_fx"))
    x_finish = int(get_from_string(path, "_fx", "_fy"))
    y_finish = int(get_from_string(path, "_fy", ".png"))

    return (y_start, x_start), (y_finish, x_finish)


def get_from_string(path: str, start: str, finish: str) -> str:
    start_index = path.find(start)
    end_index = path.find(finish)

    substring = path[start_index+3:end_index]

    return substring


def generate_paths():
    maps = get_blank_maps_list()
    # finished = False
    for map_path in maps:
        occ_map = cv2.imread(map_path, 0)
        start, finish = get_start_finish_coordinates(map_path)
        path = astar(occ_map, start, finish)
        if path:
            visualize_path(occ_map=occ_map, path=path, directory=map_path)
            # finished = True
        else:
            print("COULDN'T FIND A PATH FOR THIS EXAMPLE:", map_path)
            os.remove(map_path)
        # if finished:
        #     break


def visualize_path(occ_map: np.array, path: list, directory: str):
    dir_points_map = directory.replace('start_finish', 'start_finish_visualized')
    no_points_map = cv2.cvtColor(occ_map, cv2.COLOR_GRAY2BGR)
    points_map = cv2.imread(dir_points_map)
    path_image = np.zeros_like(points_map)  # VISUAL
    path_image.fill(255)  # VISUAL
    path_white = np.zeros_like(points_map)
    path_white = cv2.cvtColor(path_white, cv2.COLOR_BGR2GRAY)

    for point in path:
        points_map[point[0], point[1]] = (255, 0, 0)  # VISUAL
        path_image = cv2.circle(path_image, (point[1], point[0]), 10, (255, 0, 0), -1)  # VISUAL
        path_white = cv2.circle(path_white, (point[1], point[0]), 10, 255, -1)

    path_image = cv2.GaussianBlur(path_image, (33, 33), cv2.BORDER_WRAP)  # VISUAL
    path_white = cv2.GaussianBlur(path_white, (33, 33), cv2.BORDER_WRAP)

    for point in path:
        path_white[point[0], point[1]] = 255

    hsv = cv2.cvtColor(path_image, cv2.COLOR_BGR2HSV)  # VISUAL
    lower_blue = np.array([110, 0, 0])  # VISUAL
    upper_blue = np.array([130, 255, 255])  # VISUAL

    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # VISUAL

    gray_points = cv2.cvtColor(points_map, cv2.COLOR_BGR2GRAY)  # VISUAL
    gray_no_points = cv2.cvtColor(no_points_map, cv2.COLOR_BGR2GRAY)

    thresh_points = cv2.threshold(gray_points, 240, 255, cv2.THRESH_BINARY)[1]  # VISUAL
    thresh_no_points = cv2.threshold(gray_no_points, 240, 255, cv2.THRESH_BINARY)[1]

    not_mask_points = cv2.bitwise_not(thresh_points)  # VISUAL

    mask2_points = mask - not_mask_points  # VISUAL

    img_points_path_masked = cv2.bitwise_and(path_image, path_image, mask=thresh_points)  # VISUAL
    path_brightness = cv2.bitwise_and(path_white, path_white, mask=thresh_no_points)

    mask2_points_inv = cv2.bitwise_not(mask2_points)  # VISUAL

    img_points_masked = cv2.bitwise_and(points_map, points_map, mask=mask2_points_inv)  # VISUAL

    result_points = cv2.add(img_points_masked, img_points_path_masked)  # VISUAL

    dir_save_no_points_map = directory.replace('start_finish', 'paths')
    dir_save_points_map = dir_save_no_points_map.replace('paths', 'paths_with_points')  # VISUAL

    cv2.imwrite(dir_save_no_points_map, path_brightness)
    print(dir_save_no_points_map)
    cv2.imwrite(dir_save_points_map, result_points)  # VISUAL
    print(dir_save_points_map)  # VISUAL

    # cv2.imshow("result_points", result_points)  # SHOW
    # cv2.imshow("result_no_points", path_brightness)  # SHOW
    # cv2.waitKey(0)  # SHOW
    # cv2.destroyAllWindows()  # SHOW


if __name__ == '__main__':
    generate_paths()
