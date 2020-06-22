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
Assignment 4, Exercise 2
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def update_map(map, pos, reading):
    # START: 2b
    log_odd_map = np.log(map / (1 - map))
    log_odd_map[pos:pos + reading] += np.log(0.3 / 0.7)
    if (pos + reading) < len(map):
        log_odd_map[pos + reading] += np.log(0.6 / 0.4)
    log_odd_map[pos + reading + 1:pos + reading + 4] += np.log(0.8 / 0.2)
    map = 1 - (1 / (1 + np.exp(log_odd_map)))
    return map
    # END: 2b


def update_plot(map):
    """ Update the map, displaying the probability of each grid cell being occupie.

    Args:
        The one-dimensional map to plot, as a list of log odds.
    """
    # START: 2c
    x = np.arange(0, len(map))

    fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
    ax2.set_ylim(0.0, 1.0)

    ax.imshow(map[np.newaxis, :], cmap="binary", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks([])

    ax2.plot(x, map)
    plt.tight_layout()
    plt.pause(3)
    # END: 2c


def initialize_map():
    """ Initialize the map with the prior belief.

    Returns:
        A list in the size of the map, each entry containing the respective belief that the cell is occupied.
    """
    # START: 2a
    return np.ones(100) * 0.5
    # END: 2a


def initialize_runs():
    """ Initialize pre-defined runs.

    Returns:
        A list containing pre-defined runs. Each entry in the list is a list of
        pairs, each pair defining the robot's pose and the relative distance
        measurement.
    """

    runs = []
    runs.append([(10, 50), (10, 48), (10, 49)])
    runs.append([(0, 80), (10, 70), (50, 30)])
    r = []
    for i in range(10):
        r.append((i, 20 + 2 * i))
    runs.append(r)
    runs.append([(50, 30), (50, 29), (50, 50)])
    return runs


def main():
    runs = initialize_runs()
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',
                        '--run',
                        help='The pre-defined run to execute',
                        type=int,
                        choices=range(len(runs)),
                        default=0)
    args = parser.parse_args()
    map = initialize_map()
    update_plot(map)
    for pos, reading in runs[args.run]:
        map = update_map(map, pos, reading)
        update_plot(map)


if __name__ == '__main__':
    main()
