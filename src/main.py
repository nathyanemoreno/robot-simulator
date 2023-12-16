import time
from tkinter import NO
from traceback import print_tb
from matplotlib.image import NEAREST
import numpy as np
from selenium.webdriver.common.keys import Keys
from shapely.geometry import Polygon, Point, geo
from djikstra import get_points_in_radius, make_dijkstar_path, plot_map_and_route
from environiment import Environment
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from robot import Robot


def calculate_angle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


if __name__ == "__main__":
    game_url = 'https://files.crazygames.com/ruffle/worldshardestgame.html'

    # Replace 'your_div_xpath' with the XPath of the specific div to record
    div_xpath = "//*[@id='game-iframe']"

    firefox_binary_path = '/usr/bin/firefox'

    # Create an instance of the environment class
    environment = Environment()

    environment.init_browser(
        game_url, div_xpath)

    is_ready = False
    while not is_ready:
        is_ready = input("Ready? ")
        is_ready = is_ready == ""

    robot = Robot(environment, 3, (600, 300))

    robot_center, _, _ = robot.get_surrounds()

    start_point = robot_center
    main_index = 0
    target_index = 0

    path, graph = robot.get_map()

    target = None

    POS_DELTA_THRESHOLD = 7
    NEAREST_OBSTACLE_THRESHOLD = 20
    thresh = 50

    can_move = True
    while True:
        try:
            robot_center, final_spot, obstacles = robot.get_surrounds()
            # print(f"Obstacles centers: {obstacles}")

            if (robot_center != ()):  # If robot is present on the map

                # will_collide = False
                if (robot.size):
                    nearest_obstacle = min(obstacles, key=lambda obs: np.linalg.norm(
                        np.array(robot_center) - np.array(obs)))

                    obstacle_radius = 35
                    # for i, obs in enumerate(obstacles):
                    sum_sq = np.sum(
                        np.square(np.array(robot_center) - np.array(nearest_obstacle)))
                    # print(f"Robot center: {robot_center}, {obs}")

                    distance = np.sqrt(sum_sq)

                    is_near = distance < thresh

                    if (is_near):
                        print("-> Obstacle near")
                        # main_index = target_index
                        target_index = 1
                        target = None
                        # main_path = path
                        print(obstacles, nearest_obstacle)
                        new_obstacles = []
                        for i in obstacles:
                            new_obstacles.append(
                                *get_points_in_radius(graph, i, obstacle_radius))

                        try:
                            path = robot.get_path(
                                robot_center, (600, 300), new_obstacles)
                        except:
                            environment.release_key(Keys.ARROW_LEFT)
                            environment.release_key(Keys.ARROW_RIGHT)
                            environment.release_key(Keys.ARROW_UP)
                            environment.release_key(Keys.ARROW_DOWN)
                            # time.sleep(0.1)
                            continue

                        # thresh = 10
                    # elif(thresh < 40):
                    #     print("-> Obstacle away")
                    #     thresh = 30
                    #     angle = abs(calculate_angle(robot_center, obs))
                    #     print(
                    #         f" --> Angle : {round(angle)} --> Distance: {round(distance)} --> Index: {i}")
                    # can_move = distance > thresh

                    # print(f"Path from {robot_center} to {target}",path)
                if target is None:

                    target = path[target_index]
                    print(f"Setting target: {target} ")

                dx = robot_center[0] - target[0]
                dy = robot_center[1] - target[1]

                if abs(dx) < POS_DELTA_THRESHOLD:
                    dx = 0
                    # print("Stop dx")
                    environment.release_key(Keys.ARROW_LEFT)
                    environment.release_key(Keys.ARROW_RIGHT)

                if abs(dy) < POS_DELTA_THRESHOLD:
                    dy = 0
                    # print("Stop dy")
                    environment.release_key(Keys.ARROW_UP)
                    environment.release_key(Keys.ARROW_DOWN)

                if (dx == 0 and dy == 0):
                    print(
                        f"> Reached target#{target_index} {target}, position {robot_center}")

                    target = None
                    target_index += 1
                    continue

                if (dx != 0):
                    # print("Moving horizontally")
                    environment.release_key(Keys.ARROW_LEFT)
                    environment.release_key(Keys.ARROW_RIGHT)
                    # if (dx < 0):
                    #     environment.press_arrow_key(
                    #         Keys.ARROW_RIGHT)
                    environment.press_arrow_key(
                        Keys.ARROW_RIGHT if dx < 0 else Keys.ARROW_LEFT)

                if (dy != 0):
                    # print("Moving vertically")
                    environment.release_key(Keys.ARROW_UP)
                    environment.release_key(Keys.ARROW_DOWN)
                    environment.press_arrow_key(
                        Keys.ARROW_DOWN if dy < 0 else Keys.ARROW_UP)

            # time.sleep(0.01)
        except:
            time.sleep(1)
            print("Could not find robot")
            target = None
            environment.release_key(Keys.ARROW_LEFT)
            environment.release_key(Keys.ARROW_RIGHT)
            environment.release_key(Keys.ARROW_UP)
            environment.release_key(Keys.ARROW_DOWN)
            continue
