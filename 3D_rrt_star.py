import glob
import cv2
import random
import math
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

MAPS_DIRECTORY = f'/home/czarek/mgr/3D_maps/start_finish/*.npy'
MAX_ITERATIONS = 5000
GOAL_THRESHOLD = 5.0


def get_blank_maps_list() -> list:
    maps_dir = MAPS_DIRECTORY
    maps_list = sorted(glob.glob(maps_dir))
    return maps_list


def get_start_finish_coordinates(path: str) -> tuple:
    x_start = int(get_from_string(path, "_sx", "_sy"))
    y_start = int(get_from_string(path, "_sy", "_sz"))
    z_start = int(get_from_string(path, "_sz", "_fx"))
    x_finish = int(get_from_string(path, "_fx", "_fy"))
    y_finish = int(get_from_string(path, "_fy", "_fz"))
    z_finish = int(get_from_string(path, "_fz", ".npy"))

    return (z_start, y_start, x_start), (z_finish, y_finish, x_finish)


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
    def __init__(self, occ_map: np.array, start: tuple[int, int, int], goal: tuple[int, int, int], max_iterations: int,
                 goal_threshold: float):
        self.start_node = Node(start)
        self.goal = goal
        self.max_iterations = max_iterations
        self.iteration_no = None
        self.search_radius = None
        self.goal_threshold = goal_threshold
        self.nodes = [self.start_node]
        self.occ_map = occ_map
        self.map_height, self.map_width, self.map_depth = occ_map.shape
        self.best_distance = float('inf')
        self.best_node = None

    def generate_random_sample(self) -> tuple[int, int, int]:
        while True:
            x = random.randint(0, self.map_width - 1)
            y = random.randint(0, self.map_height - 1)
            z = random.randint(0, self.map_depth - 1)
            if self.occ_map[x, y, z] != 255:
                return x, y, z

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
            if self.occ_map[x, y, z] == 255:
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

    def rrt_star(self) -> list[tuple[int, int, int]]:
        goal_node = None

        for i in range(self.max_iterations):
            self.iteration_no = i + 1
            self.search_radius = self.compute_search_radius(dim=3)
            print("ITERATION:", self.iteration_no)
            print("BEST DISTANCE:", self.best_distance)
            print("SEARCH RADIUS:", self.search_radius)

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
        return path

    def visualize_path(self, path: list[tuple[int, int, int]]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot occupancy map
        x_occ, y_occ, z_occ = np.where(self.occ_map == 255)
        ax.scatter(x_occ, y_occ, z_occ, c='k', marker='s', label='Obstacles', s=50)

        # Plot path
        z_values = [position[2] for position in path]
        y_values = [position[1] for position in path]
        x_values = [position[0] for position in path]
        ax.plot3D(x_values, y_values, z_values, 'r-', linewidth=2, label='Path')

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
    maps = get_blank_maps_list()
    for map_path in maps:
        print(map_path)
        occ_map = np.load(map_path)
        start, finish = get_start_finish_coordinates(map_path)

        rrt = RRTStar(occ_map=occ_map, start=start, goal=finish, max_iterations=MAX_ITERATIONS,
                      goal_threshold=GOAL_THRESHOLD)
        path = rrt.rrt_star()

        finished = True
        if path:
            print(path)
            # rrt.visualize_tree()
            rrt.visualize_path(path)
        else:
            print("COULDN'T FIND A PATH FOR THIS EXAMPLE:", map_path)
        if finished:
            break


if __name__ == '__main__':
    generate_paths()
