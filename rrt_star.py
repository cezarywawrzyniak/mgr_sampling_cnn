import glob
from typing import Tuple, List

import cv2
import random
import math
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

MAPS_DIRECTORY = f'/home/czarek/mgr/maps/start_finish/*.png'
MAX_ITERATIONS = 5000
GOAL_THRESHOLD = 5.0


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


class Node:
    def __init__(self, position: tuple[int, int], cost: float = 0.0):
        self.position = position
        self.parent = None
        self.children = []
        self.cost = cost


class RRTStar:
    def __init__(self, occ_map: np.array, start: tuple[int, int], goal: tuple[int, int], max_iterations: int,
                 goal_threshold: float):
        self.start_node = Node(start)
        self.goal = goal
        self.max_iterations = max_iterations
        self.iteration_no = None
        self.search_radius = None
        self.goal_threshold = goal_threshold
        self.nodes = [self.start_node]
        self.occ_map = occ_map
        self.map_height, self.map_width = occ_map.shape
        self.best_distance = float('inf')
        self.best_node = None

    def generate_random_sample(self) -> tuple[int, int]:
        while True:
            x = random.randint(0, self.map_width - 1)
            y = random.randint(0, self.map_height - 1)
            if self.occ_map[y, x] != 0:
                return y, x

    def find_nearest_neighbor(self, sample) -> Node:
        nearest_node = None
        min_dist = float('inf')

        for node in self.nodes:
            dist = distance.euclidean(node.position, sample)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def steer(self, from_node: Node, to_point: tuple[int, int]) -> Node:
        # vector to new node
        direction = (to_point[0] - from_node.position[0], to_point[1] - from_node.position[1])
        dist = math.sqrt(direction[0] ** 2 + direction[1] ** 2)

        # scaling down the vector if it exceeds max_step_size
        if dist > self.search_radius:
            direction = (direction[0] * self.search_radius / dist, direction[1] * self.search_radius / dist)

        # recalculate the distance
        dist = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        new_cost = from_node.cost + dist  # Calculate the new cost

        new_node = Node((from_node.position[0] + direction[0], from_node.position[1] + direction[1]), new_cost)
        new_node.parent = from_node

        return new_node

    def can_connect_nodes(self, from_node: Node, to_node: Node) -> bool:
        if self.is_collision_free(from_node.position, to_node.position):
            from_node.children.append(to_node)
            to_node.parent = from_node
            return True
        else:
            return False

    def is_collision_free(self, point1: tuple[int, int], point2: tuple[int, int]) -> bool:
        # get distance between points and 100 point between them within that distance
        dist = np.linalg.norm(np.array(point1) - np.array(point2))
        to_check = np.linspace(0, dist, num=100)

        if point1 == point2:
            return False

        # check every calculated point between point1 and point2 for obstacle
        for dis_int in to_check:
            y = int(point1[0] - ((dis_int * (point1[0] - point2[0])) / dist))
            x = int(point1[1] - ((dis_int * (point1[1] - point2[1])) / dist))
            if self.occ_map[y, x] == 0:
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

    def goal_reached(self, node: Node, goal: tuple[int, int]) -> bool:
        dist = distance.euclidean(node.position, goal)
        if dist < self.best_distance:
            self.best_distance = dist
            self.best_node = node
        return dist <= self.goal_threshold

    def find_path(self, goal: tuple[int, int]) -> list[tuple[int, int]]:
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
        return self.map_width * self.map_height

    def compute_search_radius(self, dim: int) -> float:
        return math.pow(2 * (1 + 1.0 / dim) * (self.search_space_volume() / self.lebesgue_measure(dim)) * (
                    math.log(self.iteration_no) / self.iteration_no), 1.0 / dim)

    def rrt_star(self) -> tuple[list[tuple[int, int]], int]:
        goal_node = None

        for i in range(self.max_iterations):
            self.iteration_no = i + 1
            self.search_radius = self.compute_search_radius(dim=2)
            print("ITERATION:", self.iteration_no)
            # print("BEST DISTANCE:", self.best_distance)
            # print("SEARCH RADIUS:", self.search_radius)

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

    def visualize_tree(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # Plot obstacles or occupancy map if available
        if self.occ_map is not None:
            ax.imshow(self.occ_map, cmap='gray', origin='lower')

        # Plot nodes and connections
        for node in self.nodes:
            for child in node.children:
                y_values = [node.position[0], child.position[0]]
                x_values = [node.position[1], child.position[1]]
                ax.plot(x_values, y_values, 'b-')

        # Set start and goal markers if available
        if self.start_node.position is not None:
            ax.plot(self.start_node.position[1], self.start_node.position[0], 'go', markersize=8, label='Start')
        if self.goal is not None:
            ax.plot(self.goal[1], self.goal[0], 'ro', markersize=8, label='Goal')

        ax.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('RRT* Tree Visualization')
        plt.show()

    def visualize_path(self, path: list[tuple[int, int]]):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # Plot obstacles or occupancy map if available
        if self.occ_map is not None:
            ax.imshow(self.occ_map, cmap='gray', origin='lower')

        # Plot path
        y_values = [position[0] for position in path]
        x_values = [position[1] for position in path]
        ax.plot(x_values, y_values, 'r-', linewidth=2, label='Path')

        # Plot nodes and connections
        for node in self.nodes:
            for child in node.children:
                y_values = [node.position[0], child.position[0]]
                x_values = [node.position[1], child.position[1]]
                ax.plot(x_values, y_values, 'b-', alpha=0.2)

        # Set start and goal markers if available
        if self.start_node.position is not None:
            ax.plot(self.start_node.position[1], self.start_node.position[0], 'go', markersize=8, label='Start')
        if self.goal is not None:
            ax.plot(self.goal[1], self.goal[0], 'ro', markersize=8, label='Goal')

        ax.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('RRT* Path Visualization')
        plt.show()


def generate_paths():
    maps = get_blank_maps_list()
    for map_path in maps:
        print(map_path)
        occ_map = cv2.imread(map_path, 0)
        start, finish = get_start_finish_coordinates(map_path)

        rrt = RRTStar(occ_map=occ_map, start=start, goal=finish, max_iterations=MAX_ITERATIONS,
                      goal_threshold=GOAL_THRESHOLD)
        path = rrt.rrt_star()

        finished = True
        if path:
            print(path)
            rrt.visualize_tree()
            rrt.visualize_path(path)
        else:
            print("COULDN'T FIND A PATH FOR THIS EXAMPLE:", map_path)
        if finished:
            break


if __name__ == '__main__':
    generate_paths()
