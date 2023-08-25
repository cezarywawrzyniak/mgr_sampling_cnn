import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import distance
from pathlib import Path
from time import perf_counter

from ThreeD_train import ThreeD_UNet_cooler, MapsDataModule
from ThreeD_generate_paths import astar
from ThreeD_rrt_star import RRTStar
from ThreeD_neural_rrt import RRTStar as NeuralRRTStar

BASE_PATH = Path('/home/czarek/mgr/3D_eval_data')
MODEL_PATH = "/home/czarek/mgr/models/3D_sampling_cnn_vol2_47.pth"
MAX_ITERATIONS = 5000
GOAL_THRESHOLD = 3.0


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
    occ_map = np.array(image[0].detach().cpu().numpy())
    occ_map = occ_map.transpose((1, 2, 0))

    x_start = coords.data.tolist()[0][0][0][0]
    y_start = coords.data.tolist()[0][0][0][1]
    z_start = coords.data.tolist()[0][0][0][2]
    x_finish = coords.data.tolist()[0][0][1][0]
    y_finish = coords.data.tolist()[0][0][1][1]
    z_finish = coords.data.tolist()[0][0][1][2]
    start = (int(x_start), int(y_start), int(z_start))
    finish = (int(x_finish), int(y_finish), int(z_finish))

    # start_ = tuple(coords.data.tolist()[0][0][0])
    # finish_ = tuple(coords.data.tolist()[0][0][1])

    return occ_map, start, finish


def calculate_length(path):
    path_length = 0.0
    for i in range(len(path) - 1):
        path_length += distance.euclidean(path[i], path[i+1])
    return path_length


def astar_pathfinding(occ_map, start, finish):
    timer_start = perf_counter()
    path = astar(occ_map, start, finish)
    timer_finish = perf_counter()
    calculate_time = timer_finish - timer_start
    path_length = calculate_length(path)
    # print("A*")
    # print("CALCULATE_TIME:", calculate_time)
    # print("PATH LENGTH:", path_length)

    # visualize_astar_path(occ_map_gray, path, start, finish)
    return calculate_time, path_length


def rrt_star_pathfinding(occ_map, start, finish):
    rrt = RRTStar(occ_map=occ_map, start=start, goal=finish, max_iterations=MAX_ITERATIONS,
                  goal_threshold=GOAL_THRESHOLD)
    timer_start = perf_counter()
    path, iterations = rrt.rrt_star()
    timer_finish = perf_counter()
    calculate_time = timer_finish - timer_start
    path_length = calculate_length(path)
    # print("RRT*")
    # print("CALCULATE_TIME:", calculate_time)
    # print("PATH LENGTH:", path_length)
    # print("ITERATIONS:", iterations)

    # rrt.visualize_path(path)
    return calculate_time, path_length, iterations


def neural_rrt_star_pathfinding(model, image, mask, coords, occ_map, start, finish):
    timer_start = perf_counter()
    with torch.no_grad():
        output = model(image, coords)

    visualized_output = np.array(output[0, 0].detach().cpu().numpy())
    visualized_output = ((visualized_output - visualized_output.min()) / (
            visualized_output.max() - visualized_output.min())) * 1.0
    # visualized_output = visualized_output.transpose((1, 2, 0))
    threshold_output = 0.96
    visualized_output_binary = (visualized_output > threshold_output)
    visualized_output_masked = visualized_output.copy()
    visualized_output_masked[~visualized_output_binary] = 0
    # output_indices = np.nonzero(visualized_output_masked)
    # colors = visualized_output_masked[output_indices]

    rrt_neural = NeuralRRTStar(occ_map=occ_map, heat_map=visualized_output_masked, start=start, goal=finish,
                               max_iterations=MAX_ITERATIONS, goal_threshold=GOAL_THRESHOLD, neural_bias=0.75)
    path, iterations = rrt_neural.rrt_star()
    timer_finish = perf_counter()
    calculate_time = timer_finish - timer_start
    path_length = calculate_length(path)
    # print("NeuralRRT*")
    # print("CALCULATE_TIME:", calculate_time)
    # print("PATH LENGTH:", path_length)
    # print("ITERATIONS:", iterations)

    # rrt_neural.visualize_path(path)
    #
    # output_indices = np.nonzero(visualized_output_masked)
    # output_colors = visualized_output_masked[output_indices]
    #
    # visualized_mask = np.array(mask[0, 0].detach().cpu().numpy())
    # visualized_mask = ((visualized_mask - visualized_mask.min()) / (
    #         visualized_mask.max() - visualized_mask.min())) * 255
    # mask_indices = np.nonzero(visualized_mask)
    # mask_colors = visualized_mask[mask_indices]
    #
    # occ_map_indices = np.nonzero(occ_map)
    #
    # # Set of 3 plots
    # fig = plt.figure(figsize=(15, 5))
    #
    # # Create four subplots with 3D projections
    # ax1 = fig.add_subplot(131, projection='3d')
    # ax2 = fig.add_subplot(132, projection='3d')
    # ax3 = fig.add_subplot(133, projection='3d')
    #
    # # Scatter plots
    # ax1.scatter(occ_map_indices[0], occ_map_indices[1], occ_map_indices[2], c='k', marker='o')
    # # ax1.voxels(voxels_mask, facecolors=voxels_color, edgecolors=voxels_color)
    # ax2.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c=mask_colors, cmap='jet', marker='o')
    # ax3.scatter(output_indices[0], output_indices[1], output_indices[2], c=colors, cmap='jet', marker='o')
    #
    # # Set axis limits
    # for ax in [ax1, ax2, ax3]:
    #     ax.set_xlim(0, visualized_output_masked.shape[0])
    #     ax.set_ylim(0, visualized_output_masked.shape[1])
    #     ax.set_zlim(0, visualized_output_masked.shape[2])
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #
    # plt.tight_layout()
    # plt.show()

    return calculate_time, path_length, iterations


def main():
    dataloader = load_data()
    columns = pd.MultiIndex.from_product([['3D_A*', '3D_RRT*', '3D_NeuralRRT*'], ['Time', 'Length', 'Iterations']])
    df = pd.DataFrame(columns=columns)

    model = ThreeD_UNet_cooler()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    i = 0
    timer_start = perf_counter()
    for batch in dataloader:
        i += 1
        # for j in range(100):
        image, mask, coords = batch
        occ_map, start, finish = get_occmap_and_coordinates(image, coords)
        astar_time, astar_length = astar_pathfinding(occ_map, start, finish)
        rrt_star_time, rrt_star_length, rrt_star_iterations = rrt_star_pathfinding(occ_map, start, finish)
        neural_rrt_star_time, neural_rrt_star_length, neural_rrt_star_iterations = neural_rrt_star_pathfinding(
            model, image, mask, coords, occ_map, start, finish)
        new_row = [astar_time, astar_length, '-', rrt_star_time, rrt_star_length, rrt_star_iterations,
                   neural_rrt_star_time, neural_rrt_star_length, neural_rrt_star_iterations]
        df.loc[len(df)] = new_row
        print("ROW NO:", i)
        # break
        if i >= 1000:
            break

    print(df)
    timer_finish = perf_counter()
    calculate_time = timer_finish - timer_start
    print("CALCULATE_TIME:", calculate_time)
    df.to_excel('3D_results_vol3_47.xlsx', index=True)


if __name__ == '__main__':
    main()
