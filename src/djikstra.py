from math import radians
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, geo
from dijkstar import Graph, find_path
import networkx as nx
from shapely.geometry import Point

import networkx as nx
from shapely import affinity


def create_graph_from_polygon(polygon, grid_resolution=10):
    graph = nx.Graph()

    # Create a grid/maze representation within the polygon
    min_x, min_y, max_x, max_y = polygon.bounds
    
    step_x = (max_x - min_x) / grid_resolution
    step_y = (max_y - min_y) / grid_resolution

    # Add nodes and edges between all adjacent vertices within the grid
    for i in range(grid_resolution + 1):
        for j in range(grid_resolution + 1):
            p = (min_x + i * step_x, min_y + j * step_y)
            if polygon.contains(Point(p)):
                graph.add_node(p)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            neighbor = (min_x + (i + dx) * step_x,
                                        min_y + (j + dy) * step_y)
                            if polygon.contains(Point(neighbor)) and not crosses_boundary(p, neighbor, polygon):
                                cost = Point(p).distance(Point(neighbor))
                                graph.add_edge(p, neighbor, weight=cost)

    return graph


def crosses_boundary(point1, point2, polygon):
    line = geo.LineString([point1, point2])
    return line.crosses(polygon.boundary)


def make_dijkstar_path(polygon, start_point, end_point, avoid_points=None, grid_resolution=10):
    graph = create_graph_from_polygon(polygon, grid_resolution)

    # Round start and end points to the nearest grid points
    rounded_start = (round(start_point[0]), round(start_point[1]))
    rounded_end = (round(end_point[0]), round(end_point[1]))

    # Find the closest vertices to the rounded start and end points
    start_vertex = min(graph, key=lambda vertex: Point(
        vertex).distance(Point(rounded_start)))
    end_vertex = min(graph, key=lambda vertex: Point(
        vertex).distance(Point(rounded_end)))

    # Remove nodes (and edges) connected to avoid points
    if avoid_points:
        nodes_to_remove = []
        for point in avoid_points:
            for node in graph.nodes():
                # Adjust the distance threshold as needed
                if Point(node).distance(Point(point)) < 2.0:
                    nodes_to_remove.append(node)

        # Remove the nodes after the iteration is complete
        for node in nodes_to_remove:
            graph.remove_node(node)

    # Calculate the shortest path
    result = nx.shortest_path(
        graph, source=start_vertex, target=end_vertex, weight='weight')

    total_cost = sum(graph[u][v]['weight'] for u, v in zip(result, result[1:]))

    return total_cost, result, graph


def plot_map_and_route(polygon, start_point, end_point, path, avoid_points=None, grid_resolution=10):
    # Extract coordinates for plotting
    x_polygon, y_polygon = polygon.exterior.xy
    x_path, y_path = zip(*path)

    # Plot the map
    plt.figure(figsize=(8, 8))
    plt.plot(x_polygon, y_polygon, '-o', label='Polygon')

    # Highlight start and end points
    plt.scatter(*start_point, color='green', label='Start Point')
    plt.scatter(*end_point, color='red', label='End Point')

    # Highlight avoided points
    if avoid_points:
        x_avoid, y_avoid = zip(*avoid_points)
        plt.scatter(x_avoid, y_avoid, color='orange',
                    label='Avoided Points', marker='x')

    # Plot the grid/maze
    min_x, min_y, max_x, max_y = polygon.bounds
    step_x = (max_x - min_x) / grid_resolution
    step_y = (max_y - min_y) / grid_resolution

    for i in range(grid_resolution + 1):
        plt.plot([min_x + i * step_x, min_x + i * step_x],
                 [min_y, max_y], color='gray', linestyle='--', alpha=0.5)
        plt.plot([min_x, max_x], [min_y + i * step_y, min_y + i *
                 step_y], color='gray', linestyle='--', alpha=0.5)

    # Plot the grid points
    for i in range(grid_resolution + 1):
        for j in range(grid_resolution + 1):
            p = (min_x + i * step_x, min_y + j * step_y)
            if polygon.contains(Point(p)):
                plt.scatter(*p, color='black', marker='.')

    # Plot the route
    plt.plot(x_path, y_path, '-o', color='blue', label='Shortest Path')

    plt.title('Map with Shortest Path Inside Polygon')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage with your provided polygon
polygon_array = np.array(
    [
        [[90, 111]],
        [[89, 112]],
        [[88, 112]],
        [[88, 308]],
        [[253, 308]],
        [[253, 277]],
        [[254, 276]],
        [[542, 276]],
        [[542, 149]],
        [[544, 147]],
        [[569, 147]],
        [[571, 149]],
        [[571, 308]],
        [[671, 308]],
        [[671, 112]],
        [[670, 112]],
        [[669, 111]],
        [[508, 111]],
        [[507, 112]],
        [[506, 112]],
        [[506, 143]],
        [[505, 144]],
        [[217, 144]],
        [[217, 145]],
        [[216, 146]],
        [[216, 271]],
        [[215, 272]],
        [[189, 272]],
        [[188, 271]],
        [[188, 112]],
        [[187, 112]],
        [[186, 111]]
    ]
)

def get_points_in_radius(graph, center_point, radius):
    points_in_radius = []

    for node in graph.nodes():
        if Point(node).distance(Point(center_point)) <= radius:
            points_in_radius.append(node)

    return points_in_radius


your_polygon = Polygon([tuple(coord[0]) for coord in polygon_array])

start_point = (138, 210)
end_point = (600, 300)

obstacle_center = (300, 180)
obstacle_radius = 10

# result = get_points_in_radius(obstacle_center, obstacle_radius, 20)

avoid_points = [obstacle_center, (320, 230), (495, 160)]

# total_cost1, path1 = make_dijkstar_path(
#     your_polygon, start_point, end_point, None, 50)
total_cost2, path2, graph = make_dijkstar_path(
    your_polygon, start_point, end_point, avoid_points, 50)


result = get_points_in_radius(graph, obstacle_center, obstacle_radius)


avoid_points = [*avoid_points, *result]

total_cost_final, path_final, _a = make_dijkstar_path(
    your_polygon, start_point, end_point, avoid_points, 50)


# print("Total Cost 1:", total_cost2)
print("Total Cost 2:", total_cost2)
# print("Shortest path 1:", path1)
print("Shortest path 2:", path2)

plot_map_and_route(your_polygon, start_point,
                   end_point, path_final, avoid_points, 50)
