import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import distance
from pathlib import Path
from time import perf_counter

from train import UNet_cooler, MapsDataModule
from generate_paths import astar
from rrt_star import RRTStar
from neural_rrt import RRTStar as NeuralRRTStar

BASE_PATH = Path('/home/czarek/mgr/eval_data/test/one_shot')
MODEL_PATH = "/home/czarek/mgr/models/sampling_cnn_vol3_32.pth"
MAX_ITERATIONS = 5000
GOAL_THRESHOLD = 5.0


def load_data():
    data_module = MapsDataModule(main_path=BASE_PATH, batch_size=1, test_size=0.01, num_workers=16)
    data_module.setup("test")

    batch_size = 1

    dataloader = torch.utils.data.DataLoader(
        data_module.train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_module._num_workers
    )
    return dataloader


def get_occmap_and_coordinates(image, coords):
    occ_map = image.data.detach().cpu().numpy()
    occ_map = occ_map.transpose((0, 2, 3, 1))
    occ_map = occ_map[0]

    x_start = coords.data.tolist()[0][0][0][0]
    y_start = coords.data.tolist()[0][0][0][1]
    x_finish = coords.data.tolist()[0][0][1][0]
    y_finish = coords.data.tolist()[0][0][1][1]
    start = (int(y_start), int(x_start))
    finish = (int(y_finish), int(x_finish))

    return occ_map, start, finish


def calculate_length(path):
    path_length = 0.0
    for i in range(len(path) - 1):
        path_length += distance.euclidean(path[i], path[i+1])
    return path_length


def astar_pathfinding(occ_map, start, finish):
    occ_map_gray = cv2.cvtColor(occ_map, cv2.COLOR_BGR2GRAY)
    timer_start = perf_counter()
    path = astar(occ_map_gray, start, finish)
    timer_finish = perf_counter()
    calculate_time = timer_finish - timer_start
    path_length = calculate_length(path)
    # print("A*")
    # print("CALCULATE_TIME:", calculate_time)
    # print("PATH LENGTH:", path_length)

    # visualize_astar_path(occ_map_gray, path, start, finish)
    return calculate_time, path_length


def visualize_astar_path(occ_map: np.array, path: list, start, finish):
    print(path)
    print("START:", start)
    print("FINISH:", finish)
    points_map = cv2.cvtColor(occ_map, cv2.COLOR_GRAY2BGR)

    for point in path:
        points_map = cv2.circle(points_map, (point[1], point[0]), 10, (255, 0, 0), -1)

    points_map = cv2.circle(points_map, (start[1], start[0]), 10, (0, 255, 0), -1)
    points_map = cv2.circle(points_map, (finish[1], finish[0]), 10, (0, 0, 255), -1)

    cv2.imshow("ASTAR", points_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rrt_star_pathfinding(occ_map, start, finish):
    occ_map_gray = cv2.cvtColor(occ_map, cv2.COLOR_BGR2GRAY)
    rrt = RRTStar(occ_map=occ_map_gray, start=start, goal=finish, max_iterations=MAX_ITERATIONS,
                  goal_threshold=GOAL_THRESHOLD)
    timer_start = perf_counter()
    path, iterations = rrt.rrt_star()
    timer_finish = perf_counter()
    calculate_time = timer_finish - timer_start
    path_length = calculate_length(path)
    # print("RRT*")
    # print("CALCULATE_TIME:", calculate_time)
    # print("PATH LENGTH:", path_length)

    # rrt.visualize_path(path)
    return calculate_time, path_length, iterations


def neural_rrt_star_pathfinding(model, image, mask, coords, occ_map, start, finish):
    timer_start = perf_counter()
    with torch.no_grad():
        output = model(image, coords)
        clipped = torch.clamp(output, min=-3, max=1)

    clipped = clipped.detach().cpu().numpy()
    clipped = clipped.transpose((0, 2, 3, 1))

    rrt_neural = NeuralRRTStar(occ_map=occ_map, heat_map=clipped, start=start, goal=finish,
                               max_iterations=MAX_ITERATIONS, goal_threshold=GOAL_THRESHOLD, neural_bias=0.75)
    path, iterations = rrt_neural.rrt_star()
    timer_finish = perf_counter()
    calculate_time = timer_finish - timer_start
    path_length = calculate_length(path)
    # print("NeuralRRT*")
    # print("CALCULATE_TIME:", calculate_time)
    # print("PATH LENGTH:", path_length)

    # ideal_mask = mask.data.detach().cpu().numpy()
    # ideal_mask = ideal_mask.transpose((0, 2, 3, 1))
    # ideal_mask = ideal_mask[0]
    # rrt_neural.visualize_path(path, ideal_mask)
    return calculate_time, path_length, iterations


def main():
    dataloader = load_data()
    # columns = ['A* Time', 'A* Length', 'RRT* Time', 'RRT* Length', 'NeuralRRT* Time', 'NeuralRRT* Length']
    columns = pd.MultiIndex.from_product([['A*', 'RRT*', 'Neural+RRT*'], ['Time', 'Length', 'Iterations']])
    df = pd.DataFrame(columns=columns)

    model = UNet_cooler()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    i = 0
    timer_start = perf_counter()
    for batch in dataloader:
        i += 1
        for j in range(3):
            image, mask, coords = batch
            occ_map, start, finish = get_occmap_and_coordinates(image, coords)
            astar_time, astar_length = astar_pathfinding(occ_map, start, finish)
            rrt_star_time, rrt_star_length, rrt_star_iterations = rrt_star_pathfinding(occ_map, start, finish)

            neural_rrt_star_time, neural_rrt_star_length, neural_rrt_star_iterations = (
                neural_rrt_star_pathfinding(model, image, mask, coords, occ_map, start, finish))

            new_row = [astar_time, astar_length, '-', rrt_star_time, rrt_star_length, rrt_star_iterations,
                       neural_rrt_star_time, neural_rrt_star_length, neural_rrt_star_iterations]
            df.loc[len(df)] = new_row
            print("ROW NO:", j)
        break
        if i >= 1000:
            break

    print(df)
    timer_finish = perf_counter()
    calculate_time = timer_finish - timer_start
    print("CALCULATE_TIME:", calculate_time)
    df.to_excel('results_one_problem.xlsx', index=True)


if __name__ == '__main__':
    main()
