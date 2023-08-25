from typing import Tuple, List

import torch
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from pathlib import Path
from time import perf_counter

from ThreeD_train import ThreeD_UNet_cooler, MapsDataModule

BASE_PATH = Path('/home/czarek/mgr/3D_eval_data')
MODEL_PATH = "/home/czarek/mgr/models/3D_sampling_cnn_vol2_47.pth"
MAX_ITERATIONS = 5000
GOAL_THRESHOLD = 50.0


def get_blank_maps_list() -> list:
    maps_list = [str(image_path) for image_path in sorted((BASE_PATH / 'images').iterdir())]
    return maps_list


def get_start_finish_coordinates(path: str) -> tuple:
    x_start = int(get_from_string(path, "_sx", "_sy"))
    y_start = int(get_from_string(path, "_sy", "_sz"))
    z_start = int(get_from_string(path, "_sz", "_fx"))
    x_finish = int(get_from_string(path, "_fx", "_fy"))
    y_finish = int(get_from_string(path, "_fy", "_fz"))
    z_finish = int(get_from_string(path, "_fz", ".npy"))

    return (x_start, y_start, z_start), (x_finish, y_finish, z_finish)


def get_from_string(path: str, start: str, finish: str) -> str:
    start_index = path.find(start)
    end_index = path.find(finish)

    substring = path[start_index+3:end_index]

    return substring


class Node:
    def __init__(self, position: tuple[int, int, int], cost: float = 0.0):
        self.position = position
        self.parent = None
        self.children = []
        self.cost = cost


class RRTStar:
    def __init__(self, occ_map: np.array, heat_map: np.array, start: tuple[int, int, int], goal: tuple[int, int, int],
                 max_iterations: int, goal_threshold: float, neural_bias: float):
        self.start_node = Node(start)
        self.goal = goal
        self.max_iterations = max_iterations
        self.iteration_no = None
        self.search_radius = None
        self.goal_threshold = goal_threshold
        self.nodes = [self.start_node]
        self.occ_map = occ_map  # TODO
        self.map_height, self.map_width, self.map_depth = occ_map.shape
        self.best_distance = float('inf')
        self.best_node = None
        self.neural_bias = neural_bias
        self.heat_map = heat_map - 250

    def generate_random_sample(self) -> tuple[int, int, int]:
        while True:
            x = random.randint(0, self.map_width - 1)
            y = random.randint(0, self.map_height - 1)
            z = random.randint(0, self.map_depth - 1)
            if self.occ_map[x, y, z] == 0:
                return x, y, z
            # else:
            #     print(self.occ_map[x, y, z])

    def generate_neural_sample(self) -> tuple[int, int, int]:
        while True:
            # Flatten the heatmap to a 1D array
            flat_heatmap = self.heat_map.flatten()

            # Add a constant to shift the values to be non-negative
            # shifted_heatmap = flat_heatmap - np.min(flat_heatmap) + 1e-6
            shifted_heatmap = flat_heatmap - np.min(flat_heatmap)

            # Calculate the weights by taking the exponential of the shifted heatmap
            # weights = np.exp(shifted_heatmap)
            weights = shifted_heatmap

            # Normalize the weights to sum up to 1
            normalized_weights = weights / np.sum(weights)

            # Generate a random value between 0 and 1
            random_value = random.uniform(0, 1)

            # Calculate the cumulative weights
            cumulative_weights = np.cumsum(normalized_weights)

            # Find the index where the random value falls in the cumulative weights
            index = np.searchsorted(cumulative_weights, random_value)

            # Convert the index back to 2D coordinates
            height, width, depth = self.heat_map.shape
            heat_map_shape = height, width, depth

            # Ensure the index is within bounds
            index = min(max(index, 0), np.prod(heat_map_shape) - 1)

            # x, y, z = np.unravel_index(index-1, heat_map_shape)  # TODO
            x, y, z = np.unravel_index(index, heat_map_shape)
            if self.occ_map[x, y, z] == 0:
                return x, y, z  # TODO
            # else:
            #     print(self.occ_map[x, y, z])

    def find_nearest_neighbor(self, sample) -> Node:
        nearest_node = None
        min_dist = float('inf')

        for node in self.nodes:
            dist = distance.euclidean(node.position, sample)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def steer(self, from_node: Node, to_point: tuple[int, int, int]) -> Node:
        # vector to new node
        direction = (to_point[0] - from_node.position[0], to_point[1] - from_node.position[1],
                     to_point[2] - from_node.position[2])
        dist = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)

        # scaling down the vector if it exceeds max_step_size
        if dist > self.search_radius:
            direction = (direction[0] * self.search_radius / dist, direction[1] * self.search_radius / dist,
                         direction[2] * self.search_radius / dist)

        # recalculate the distance
        dist = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
        new_cost = from_node.cost + dist  # Calculate the new cost

        new_node = Node((from_node.position[0] + direction[0], from_node.position[1] + direction[1],
                         from_node.position[2] + direction[2]), new_cost)
        new_node.parent = from_node

        return new_node

    def can_connect_nodes(self, from_node: Node, to_node: Node) -> bool:
        if self.is_collision_free(from_node.position, to_node.position):
            from_node.children.append(to_node)
            to_node.parent = from_node
            return True
        else:
            return False

    def is_collision_free(self, point1: tuple[int, int, int], point2: tuple[int, int, int]) -> bool:
        # get distance between points and 100 point between them within that distance
        dist = np.linalg.norm(np.array(point1) - np.array(point2))
        to_check = np.linspace(0, dist, num=100)

        if point1 == point2:
            return False

        # check every calculated point between point1 and point2 for obstacle
        for dis_int in to_check:
            x = int(point1[0] - ((dis_int * (point1[0] - point2[0])) / dist))
            y = int(point1[1] - ((dis_int * (point1[1] - point2[1])) / dist))
            z = int(point1[2] - ((dis_int * (point1[2] - point2[2])) / dist))
            if self.occ_map[x, y, z] != 0:
                # print(self.occ_map[x, y, z])
                return False

        return True

    def rewire_tree(self, new_node: Node):
        # get list of nearby nodes
        nearby_nodes = self.find_nearby_nodes(new_node)

        # check if there is a better path to the new node from the nodes in the list, replace when lower cost
        for nearby_node in nearby_nodes:
            new_cost = nearby_node.cost + distance.euclidean(nearby_node.position, new_node.position)
            if new_cost < new_node.cost:
                if self.is_collision_free(nearby_node.position, new_node.position):
                    new_node.parent.children.remove(new_node)
                    new_node.parent = nearby_node
                    nearby_node.children.append(new_node)
                    new_node.cost = new_cost

        # check if there is a better bath to one of the nodes in the list from the new node, replace when lower cost
        for node in nearby_nodes:
            redone_cost = new_node.cost + distance.euclidean(new_node.position, node.position)
            if redone_cost < node.cost:
                if self.is_collision_free(new_node.position, node.position):
                    node.parent.children.remove(node)
                    node.parent = new_node
                    new_node.children.append(node)
                    node.cost = redone_cost

    def find_nearby_nodes(self, node: Node) -> list[Node]:
        nearby_nodes = []
        for other_node in self.nodes:
            if distance.euclidean(node.position, other_node.position) <= self.search_radius:
                nearby_nodes.append(other_node)
        return nearby_nodes

    def goal_reached(self, node: Node, goal: tuple[int, int, int]) -> bool:
        dist = distance.euclidean(node.position, goal)
        if dist < self.best_distance:
            self.best_distance = dist
            self.best_node = node
        return dist <= self.goal_threshold

    def find_path(self, goal: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        path = []
        current_node = goal

        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent

        path.reverse()  # Reverse the path to start from the start node
        return path

    def lebesgue_measure(self, dim: int) -> float:
        return math.pow(math.pi, dim / 2.0) / math.gamma((dim / 2.0) + 1)

    def search_space_volume(self) -> float:
        return self.map_width * self.map_height * self.map_width

    def compute_search_radius(self, dim: int) -> float:
        return math.pow(2 * (1 + 1.0 / dim) * (self.search_space_volume() / self.lebesgue_measure(dim)) * (
                    math.log(self.iteration_no) / self.iteration_no), 1.0 / dim)

    def rrt_star(self) -> tuple[list[tuple[int, int, int]], int]:
        goal_node = None

        for i in range(self.max_iterations):
            self.iteration_no = i + 1
            self.search_radius = self.compute_search_radius(dim=3)
            # print("ITERATION:", self.iteration_no)
            # print("BEST DISTANCE:", self.best_distance)
            # print("SEARCH RADIUS:", self.search_radius)

            if random.random() < self.neural_bias:
                random_sample = self.generate_neural_sample()
            else:
                random_sample = self.generate_random_sample()

            nearest_neighbor = self.find_nearest_neighbor(random_sample)
            new_node = self.steer(nearest_neighbor, random_sample)

            if self.can_connect_nodes(nearest_neighbor, new_node):
                self.rewire_tree(new_node)

                if self.goal_reached(new_node, self.goal):
                    goal_node = new_node
                    goal_node.position = self.goal
                    # Break for now, if tuned better it can iterate for longer to find better path?
                    break
                self.nodes.append(new_node)

        if goal_node is None:  # Goal not reached
            goal_node = self.best_node  # Take the closest node to goal TODO should be checked for obstacle
            # return None

        # Find the best path from the goal to the start
        path = self.find_path(goal_node)
        return path, self.iteration_no

    def visualize_path(self, path: list[tuple[int, int, int]]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot occupancy map
        x_occ, y_occ, z_occ = np.where(self.occ_map == 1.0)
        occupied_mask = self.occ_map.astype(bool)
        color = np.zeros(self.occ_map.shape + (4,))
        color[occupied_mask] = (0, 0, 0, 0.2)
        # ax.scatter(x_occ, y_occ, z_occ, c='k', marker='o', label='Obstacles')
        ax.voxels(occupied_mask, facecolors=color, edgecolors=color)

        # Plot path
        z_values = [position[2] for position in path]
        y_values = [position[1] for position in path]
        x_values = [position[0] for position in path]
        ax.plot3D(x_values, y_values, z_values, 'r-', linewidth=8, label='Path')

        # Plot nodes and connections
        for node in self.nodes:
            for child in node.children:
                z_values = [node.position[2], child.position[2]]
                y_values = [node.position[1], child.position[1]]
                x_values = [node.position[0], child.position[0]]
                ax.plot3D(x_values, y_values, z_values, 'b-', alpha=0.2)

        # Set start and goal markers if available
        if self.start_node.position is not None:
            ax.scatter(self.start_node.position[0], self.start_node.position[1], self.start_node.position[2], c='g', marker='o', s=100, label='Start')
        if self.goal is not None:
            ax.scatter(self.goal[0], self.goal[1], self.goal[2], c='r', marker='o', s=100, label='Goal')

        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('RRT* Path Visualization in 3D')
        plt.show()


def generate_paths():
    model = ThreeD_UNet_cooler()
    model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()

    data_module = MapsDataModule(main_path=BASE_PATH)
    data_module.setup("test")

    batch_size = 1
    sampler = torch.utils.data.RandomSampler(data_module.train_dataset)

    dataloader = torch.utils.data.DataLoader(
        data_module.train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=data_module._num_workers
    )

    batch = next(iter(dataloader))
    image, mask, coords = batch
    start = tuple(coords.data.tolist()[0][0][0])
    finish = tuple(coords.data.tolist()[0][0][1])

    occ_map = np.array(image[0].detach().cpu().numpy())
    occ_map = occ_map.transpose((1, 2, 0))
    occ_map_indices = np.nonzero(occ_map)

    timer_neural_start = perf_counter()
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

    rrt_neural = RRTStar(occ_map=occ_map, heat_map=visualized_output_masked, start=start, goal=finish,
                         max_iterations=MAX_ITERATIONS, goal_threshold=GOAL_THRESHOLD, neural_bias=0.75)

    path, iterations = rrt_neural.rrt_star()
    timer_neural_stop = perf_counter()

    output_indices = np.nonzero(visualized_output_masked)
    output_colors = visualized_output_masked[output_indices]

    visualized_mask = np.array(mask[0, 0].detach().cpu().numpy())
    visualized_mask = ((visualized_mask - visualized_mask.min()) / (
                visualized_mask.max() - visualized_mask.min())) * 255
    mask_indices = np.nonzero(visualized_mask)
    mask_colors = visualized_mask[mask_indices]

    if path:
        print(path)
        # rrt_neural.visualize_path(path)
    else:
        print("COULDN'T FIND A PATH FOR THIS EXAMPLE:", start, finish)

    print(f'Calculation time of neural RRT*: {timer_neural_stop - timer_neural_start}')
    print(f'Start X:{start[0]}, Y:{start[1]}, Z:{start[2]}')
    print(f'Finish X:{finish[0]}, Y:{finish[1]}, Z:{finish[2]}')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(occ_map_indices[0], occ_map_indices[1], occ_map_indices[2], c='k', marker='o')
    occupied_mask = occ_map.astype(bool)
    color = np.zeros(occ_map.shape + (4,))
    color[occupied_mask] = (0, 0, 0, 0.2)
    ax.voxels(occupied_mask, facecolors=color, edgecolors=color)
    ax.scatter(start[0], start[1], start[2], c='g', marker='o', s=300, label='Start')
    ax.scatter(finish[0], finish[1], finish[2], c='r', marker='o', s=300, label='Finish')
    # ax.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c=mask_colors, cmap='jet', marker='o')
    # ax.scatter(output_indices[0], output_indices[1], output_indices[2], c=output_colors, cmap='jet', marker='o')
    # ax.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c='b', marker='o')
    # ax.scatter(output_indices[0], output_indices[1], output_indices[2], c='r', marker='o')
    ax.set_xlim(0, visualized_mask.shape[0])
    ax.set_ylim(0, visualized_mask.shape[1])
    ax.set_zlim(0, visualized_mask.shape[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(occ_map_indices[0], occ_map_indices[1], occ_map_indices[2], c='k', marker='o')
    # ax.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c=mask_colors, cmap='jet', marker='o')
    ax.scatter(output_indices[0], output_indices[1], output_indices[2], c=output_colors, cmap='jet', marker='o')
    # ax.scatter(mask_indices[0], mask_indices[1], mask_indices[2], c='b', marker='o')
    # ax.scatter(output_indices[0], output_indices[1], output_indices[2], c='r', marker='o')
    ax.set_xlim(0, visualized_mask.shape[0])
    ax.set_ylim(0, visualized_mask.shape[1])
    ax.set_zlim(0, visualized_mask.shape[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == '__main__':
    generate_paths()
