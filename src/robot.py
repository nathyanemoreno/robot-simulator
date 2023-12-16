import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from djikstra import make_dijkstar_path
from environiment import Environment
from shapely.geometry import Polygon


class Robot:
    def __init__(self, environment: Environment, vision_radius, initial_target=None):
        self.vision_radius = vision_radius * 30  # Square Grids
        # self.vision_radius = vision_radius * environment.grid_square_size# Square Grids
        self.environment = environment

        self.target = initial_target

    def get_surrounds(self, source="environment"):
        if (source == "path"):
            image = np.array(Image.open("recorded.png"))
            if (self.environment.driver):
                self.environment.driver.quit()

        if (source == "environment"):
            image = self.environment.read_screen()

        self.environment.state = image

        env = self.environment.get_environment(image)

        _, self.start_point, self.size = self.environment.find_red_square(
            self.environment.state)

        return env

    def get_map(self):
        erosion_mean = int(np.ceil((self.size[0] + self.size[1] + 8) / 4))
        print(f"Erosion: {erosion_mean}")

        map_area, mask = self.environment.get_map_contours(
            self.environment.state, erosion_mean)

        _, _, obstacles = self.environment.get_environment(
            self.environment.state)

        map_polygon = Polygon([tuple(coord[0]) for coord in map_area])

        _, path, graph = make_dijkstar_path(
            map_polygon, self.start_point, self.target, obstacles, 20)

        return path, graph

    def get_path(self, start_point, end_point, obstacles):
        erosion_mean = int(np.ceil((self.size[0] + self.size[1] + 8) / 4))
        print(f"Erosion: {erosion_mean}")

        map_area, mask = self.environment.get_map_contours(
            self.environment.state, erosion_mean)

        map_polygon = Polygon([tuple(coord[0]) for coord in map_area])

        _, path, _ = make_dijkstar_path(
            map_polygon, start_point, end_point, obstacles, 20)

        return path
