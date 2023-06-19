import glob
import cv2
import random
import math
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

MAPS_DIRECTORY = f'/home/czarek/mgr/maps/start_finish/*.png'


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
    def __init__(self, position, cost=0):
        self.position = position
        self.parent = None
        self.children = []
        self.cost = cost


class RRTStar:
    def __init__(self, occ_map, start, goal, max_iterations, max_step_size, nearby_nodes_radius, goal_threshold):
        self.start_node = Node(start)
        self.goal = goal
        self.max_iterations = max_iterations
        self.max_step_size = max_step_size
        self.radius = nearby_nodes_radius
        self.goal_threshold = goal_threshold
        self.nodes = [self.start_node]
        self.occ_map = occ_map
        self.map_height, self.map_width = occ_map.shape
        self.best_distance = float('inf')
        self.best_node = None

    def generate_random_sample(self):
        while True:
            x = random.randint(0, self.map_width - 1)
            y = random.randint(0, self.map_height - 1)
            if self.occ_map[y, x] != 0:
                return x, y

    def find_nearest_neighbor(self, sample):
        nearest_node = None
        min_dist = float('inf')

        for node in self.nodes:
            dist = distance.euclidean(node.position, sample)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def steer(self, from_node, to_point):
        # vector to new node
        direction = (to_point[0] - from_node.position[0], to_point[1] - from_node.position[1])
        dist = math.sqrt(direction[0] ** 2 + direction[1] ** 2)

        # scaling down the vector if it exceeds max_step_size
        if dist > self.max_step_size:
            direction = (direction[0] * self.max_step_size / dist, direction[1] * self.max_step_size / dist)

        # recalculate the distance
        dist = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        new_cost = from_node.cost + dist  # Calculate the new cost

        new_node = Node((from_node.position[0] + direction[0], from_node.position[1] + direction[1]), new_cost)
        new_node.parent = from_node

        return new_node

    def connect_nodes(self, from_node, to_node):
        if self.is_collision_free(from_node.position, to_node.position):
            from_node.children.append(to_node)
            to_node.parent = from_node
            return True
        else:
            return False

    def is_collision_free(self, point1, point2):
        dist = np.linalg.norm(np.array(point1) - np.array(point2))
        to_check = np.linspace(0, dist, num=100)
        # print(point1)
        # print(point2)
        # print(to_check)
        if point1 == point2:
            return False

        for dis_int in to_check:
            y = int(point1[0] - ((dis_int * (point1[0] - point2[0])) / dist))
            x = int(point1[1] - ((dis_int * (point1[1] - point2[1])) / dist))
            # chat suggestion
            # y = int(point1[1] + ((dis_int * (point2[1] - point1[1])) / dist))
            # x = int(point1[0] + ((dis_int * (point2[0] - point1[0])) / dist))
            # print(y, x)
            if self.occ_map[y, x] == 0:
                return False

        return True

    def rewire_tree(self, new_node):
        nearby_nodes = self.find_nearby_nodes(new_node, self.radius)

        for node in nearby_nodes:
            new_cost = new_node.cost + distance.euclidean(node.position, new_node.position)
            if new_cost < node.cost:
                if self.is_collision_free(node.position, new_node.position):
                    node.parent.children.remove(node)
                    node.parent = new_node
                    new_node.children.append(node)
                    node.cost = new_cost

    def find_nearby_nodes(self, node, radius):
        nearby_nodes = []
        for other_node in self.nodes:
            distance.euclidean(node.position, other_node.position)
            if distance.euclidean(node.position, other_node.position) <= radius:
                nearby_nodes.append(other_node)
        return nearby_nodes

    def goal_reached(self, node, goal):
        dist = distance.euclidean(node.position, goal)
        # print(dist)
        if dist < self.best_distance:
            self.best_distance = dist
            self.best_node = node
        return dist <= self.goal_threshold

    def find_path(self, goal):
        path = []
        current_node = goal

        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent

        path.reverse()  # Reverse the path to start from the start node
        return path

    def rrt_star(self):
        goal_node = None

        for _ in range(self.max_iterations):
            print("ITERATION:", _)
            print("BEST DISTANCE:", self.best_distance)
            random_sample = self.generate_random_sample()
            nearest_neighbor = self.find_nearest_neighbor(random_sample)
            # print("NEAREST_NEIGBOR", nearest_neighbor.position)
            # print("NEAREST_NEIGBOR", nearest_neighbor.cost)
            new_node = self.steer(nearest_neighbor, random_sample)
            # print("NEW_NODE", new_node.position)
            # print("NEW_NODE", new_node.cost)

            if self.connect_nodes(nearest_neighbor, new_node):
                self.rewire_tree(new_node)

                if self.goal_reached(new_node, self.goal):
                    goal_node = new_node
                    # TODO
                    # SHOULD IT REALLY BREAK HERE? I DON'T THINK SO BECAUSE IT CAN STILL FIND A BETTER PATH.
                    # I GUESS IT DEPENDS ON WHETHER WE WANT TO FIND A PATH OR THE BEST PATH POSSIBLE WITHIN MAX_ITER
                    break
                self.nodes.append(new_node)

        if goal_node is None:
            goal_node = self.best_node
            # return None  # Goal not reached

        # Find the best path from the goal to the start
        path = self.find_path(goal_node)
        return path

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

    def visualize_path(self, path):
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
    MAX_ITERATIONS = 5000
    MAX_STEP_SIZE = 100.0
    NEARBY_NODES_RADIUS = 80.0
    GOAL_THRESHOLD = 20.0
    maps = get_blank_maps_list()
    finished = False
    for map_path in maps:
        print(map_path)
        occ_map = cv2.imread(map_path, 0)
        start, finish = get_start_finish_coordinates(map_path)

        rrt = RRTStar(occ_map=occ_map, start=start, goal=finish, max_iterations=MAX_ITERATIONS,
                      max_step_size=MAX_STEP_SIZE, nearby_nodes_radius=NEARBY_NODES_RADIUS, goal_threshold=GOAL_THRESHOLD)
        path = rrt.rrt_star()

        finished = True
        if path:
            # visualize_path(occ_map=occ_map, path=path, directory=map_path)
            print(path)
            rrt.visualize_tree()
            rrt.visualize_path(path)
        else:
            print("COULDN'T FIND A PATH FOR THIS EXAMPLE:", map_path)
        if finished:
            break


if __name__ == '__main__':
    generate_paths()
