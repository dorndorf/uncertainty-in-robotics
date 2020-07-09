#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Till Hofmann <hofmann@kbsg.rwth-aachen.de>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Library General Public License for more details.
#
# Read the full text in the LICENSE.GPL file in the doc directory.
#
"""
Uncertainty in Robotics Assignment 5.1: Path Planning
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import euclidean


class MapDrawer:
    """ Draws a map and paths with pyplot. """
    def __init__(self):
        """ Initialize the drawer. """
        self.fig, self.ax = plt.subplots()
        plt.tight_layout()
        self.path_artist = None

    def clear(self):
        """ Clear the plot. """
        self.ax.clear()

    def draw_path(self, path, display_time=0.01):
        """ Draw a path and display it for some amount of time.

        Args:
            path: The path to display as list of (x, y) pairs
            display_time: The amount of time in seconds to display the plot
        """
        if self.path_artist:
            self.path_artist.remove()
        codes = [Path.MOVETO] + [Path.LINETO] * (len(path) - 1)
        self.path_artist = self.ax.add_patch(
            PathPatch(Path(path, codes),
                      facecolor='None',
                      edgecolor='red',
                      linewidth=5))

        plt.pause(display_time)

    def draw_cost_map(self, cost_map, display_time=0.01):
        """ Draw the given cost map.

        Args:
            cost_map: An ndarray in the shape of the map
            display_time: The amount of time in seconds to display the plot
        """
        cost_map = np.swapaxes(cost_map, 0, 1)
        self.ax.set_xticks(np.arange(cost_map.shape[1]))
        self.ax.set_yticks(np.arange(cost_map.shape[0]))
        plt.grid(True, which='both')
        self.ax.imshow(-cost_map, alpha=0.5, cmap='inferno', origin='lower')

    def draw_open_queue(self, open_queue):
        """ Draw the A* open queue.

        Args:
            open_queue: The open queue as list of (dist, pos) pairs, where pos is in the form (x, y)
        """

        for dist, (x, y) in open_queue:
            self.ax.text(x,
                         y,
                         "{:.0f}".format(dist),
                         ha="center",
                         va="center",
                         color="black")
        plt.pause(0.01)


def init_costmap(map):
    """ Initialize the cost map such that each cell has a higher initial cost than any possible path to that cell. """
    ### START: 1a
    cost_map = np.full_like(map, np.inf)
    return cost_map
    ### END: 1a


def get_adjacent_nodes(pos, limits):
    """ Compute the neighbor nodes of the given position.

    Args:
        pos: The position as (x, y) pair
        limits: The limits of the map as (xmax, ymax) pair

    Returns:
        A list of positions adjacent to the given position
    """
    ### START: 1b
    adj_nodes = []
    for shift in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        if 0 <= pos[0] + shift[0] < limits[0] and 0 <= pos[1] + shift[1] < limits[1]:
            adj_nodes.append((pos[0] + shift[0], pos[1] + shift[1]))
    return adj_nodes
    ### END: 1b


class ValueIterator:
    """ A heuristic using deterministic value iteration. """
    def __init__(self, map, goal):
        """ Initialize the value iterator by computing its cost map.

        Args:
            map: An ndarray where value of 1 means the cell is occupied; the cost map will have the same shape
            goal: The goal position as (x, y) pair
        """
        ### START: 1e
        self.cost_map = init_costmap(map)
        self.cost_map[goal] = 0
        visited = map.copy()
        queue = [(self.cost_map[goal], goal)]
        while queue:
            queue.sort()
            current = queue.pop(0)[1]
            if visited[current]:
                continue
            visited[current] = 1.0
            adj_nodes = get_adjacent_nodes(current, map.shape)
            for an in adj_nodes:
                if self.cost_map[current] + 1 < self.cost_map[an] and not map[an]:
                    self.cost_map[an] = self.cost_map[current] + 1
                    queue.append((self.cost_map[an], an))
        ### END: 1e

    def distance(self, start, goal):
        """ Compute the distance between the given start and goal, using the value iterator's costmap.

        Args:
            start: The start position as (x, y) pair
            goal: The goal position as (x, y) pair; must be the same as the position given to the constructor

        Returns:
            The estimated distance between start and goal
        """
        ### START: 1e
        return self.cost_map[start]
        ### END: 1e


def bfs(drawer, map, start, goal):
    """ Determine path from start to goal in the given map using BFS.
        Also draw the path fragment in each iteration.

    Args:
        drawer: A map drawer to use for drawing path fragments
        map: The map to operate on as ndarray; a value of 0 is a free cell, a value of 1 is an occupied cell
        start: The start position as (x, y) pair
        goal: The goal position as (x, y) pair
    Returns:
        A path from start to goal as list of pairs
    """
    ### START: 1c
    visited = map.copy()
    visited[start] = 1
    new_paths = [[start]]
    end_reached = False
    while not end_reached:
        paths = new_paths
        new_paths = []
        for path in paths:
            adj_nodes = get_adjacent_nodes(path[-1], map.shape)
            for node in adj_nodes:
                if visited[node]:
                    continue
                visited[node] = 1
                new_paths.append(path + [node])
                drawer.draw_path(path + [node])
                if node == goal:
                    final_path = new_paths[-1]
                    end_reached = True
                    break
    return final_path
    ### END: 1c


def reconstruct_path(came_from, current):
    """ Helper function for A* search to compute a path.

    Args:
        came_from: a map that defines the previous position on the path for each position on the path
        current: The end position
    Returns:
        A list of positions as (x, y) pairs
    """
    ### START: 1d
    path = []
    while (current[0] >= 0):
        path.append(current)
        current = tuple(came_from[current])

    return path[::-1]
    ### END: 1d


def astar(drawer, map, start, goal, heuristic):
    """ Do A* search to determine a path from start to goal on the given map.

    Args:
        drawer: A map drawer to use for the cost map and open queue in each iteration
        map: The map to operate on as ndarray; a value of 0 is a free cell, a value of 1 is an occupied cell
        start: The start position as (x, y) pair
        goal: The goal position as (x, y) pair
        heuristic: The heuristic function to use
    """
    ### START: 1d

    visited = map.copy()
    visited[start] = 1
    cost_map = init_costmap(map)
    cost_map[start] = 0.0
    queue = [(cost_map[start], start)]
    came_from = np.full(map.shape + (2,), -1, dtype=int)

    while queue:
        queue.sort()
        current = queue.pop(0)[1]
        visited[current] = 1
        if current == goal:
            return reconstruct_path(came_from, current)
        adj_nodes = get_adjacent_nodes(current, map.shape)
        for an in adj_nodes:
            if visited[an]:
                continue
            if cost_map[an] <= cost_map[current] + 1:
                continue
            cost_map[an] = cost_map[current] + 1
            came_from[an] = current
            queue.append((cost_map[an] + heuristic(an, goal), an))
        drawer.draw_cost_map(cost_map)
        drawer.draw_open_queue(queue)

    ### END: 1d


def init_maps():
    """ Initialize some artificial maps.

    Returns:
        A list of tuples (map, start, goal), where map is an ndarray and start and goal are positions in the form (x, y)
    """
    maps = []
    map = np.zeros((10, 10))
    map[4, 5] = 1
    map[4, 6] = 1
    map[5, 5] = 1
    map[5, 6] = 1
    map[6, 5] = 1
    map[6, 6] = 1
    map[7, 5] = 1
    map[7, 6] = 1
    map[8, 5] = 1
    map[8, 6] = 1
    maps.append((map, (6, 2), (6, 8)))
    map = np.zeros((50, 50))
    for (x, y), _ in np.ndenumerate(map):
        if x >= 10 and x < 20 and y >= 10 and y < 30:
            map[x, y] = 1
    maps.append((map, (5, 5), (5, 25)))
    maps.append((map, (15, 5), (15, 35)))
    map = np.zeros((50, 50))
    for (x, y), _ in np.ndenumerate(map):
        if x >= 10 and x < 20 and y >= 10 and y < 40:
            map[x, y] = 1
        elif x >= 30 and x < 40 and y >= 10 and y < 40:
            map[x, y] = 1
        elif x >= 10 and x < 40 and y == 40:
            map[x, y] = 1
    maps.append((map, (25, 15), (25, 45)))
    return maps


def main():
    maps = init_maps()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m',
                           '--map',
                           help='The map to use',
                           default=0,
                           type=int,
                           choices=range(len(maps)))
    argparser.add_argument('-s',
                           '--search',
                           choices=['bfs', 'astar'],
                           default='astar')
    argparser.add_argument('--heuristic',
                           default='iteration',
                           choices=['cityblock', 'euclidean', 'iteration'])
    args = argparser.parse_args()
    map, start, goal = maps[args.map]
    drawer = MapDrawer()
    drawer.draw_cost_map(map)
    plt.pause(2)
    if args.search == 'bfs':
        path = bfs(drawer, map, start, goal)
    elif args.search == 'astar':
        if args.heuristic == 'cityblock':
            heuristic = cityblock
        elif args.heuristic == 'euclidean':
            heuristic = euclidean
        elif args.heuristic == 'iteration':
            value_iterator = ValueIterator(map, goal)
            drawer.draw_cost_map(value_iterator.cost_map)
            plt.pause(5)
            heuristic = value_iterator.distance

        path = astar(drawer, map, start, goal, heuristic)
    if path:
        drawer.draw_path(path, 3)


if __name__ == '__main__':
    main()
