#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Till Hofmann <hofmann@kbsg.rwth-aachen.de>
"""
Assignment 2 Exercise 2
"""

import argparse
import numpy as np
import random
import math

import matplotlib.pyplot as pyplot

from enum import Enum
from scipy.spatial import distance


class Action(Enum):
    STAY = 0
    LEFT = 1
    UP = 2
    RIGHT = 3
    DOWN = 4


def initialize_belief(grid_shape):
    """ Compute the initial belief.

    Args:
        grid_shape: A pair (x,y) that defines the extensions of the grid.

    Returns:
        An ndarray with shape (x,y) containing the initial belief distribution.
    """
    # START: 2a
    return np.ones(grid_shape) / (grid_shape[0]*grid_shape[1])
    # END: 2a


def predict(previous_belief, action):
    """ Predict the next belief state given the next action.

    Args:
        previous_belief: The previous belief as ndarray
        action: The next action as instance of the enum class Action

    Returns:
        The predicted next belief state as ndarray.
    """
    # START: 2b
    succ_prob = 0.8
    stay_prob = 0.2
    if action == Action.STAY:
        return previous_belief
    stay_belief = previous_belief*stay_prob
    if action == Action.LEFT:
        move_belief = np.zeros(previous_belief.shape)
        move_belief[0, :] = previous_belief[0, :]*succ_prob
        move_belief[:-1, :] = np.add(move_belief[:-1, :], previous_belief[1:, :]*succ_prob)
    elif action == Action.RIGHT:
        move_belief = np.zeros(previous_belief.shape)
        move_belief[-1, :] = previous_belief[-1, :]*succ_prob
        move_belief[1:, :] = np.add(move_belief[1:, :], previous_belief[:-1, :]*succ_prob)
    elif action == Action.DOWN:
        move_belief = np.zeros(previous_belief.shape)
        move_belief[:, 0] = previous_belief[:, 0]*succ_prob
        move_belief[:, :-1] = np.add(move_belief[:, :-1], previous_belief[:, 1:]*succ_prob)
    elif action == Action.UP:
        move_belief = np.zeros(previous_belief.shape)
        move_belief[:, -1] = previous_belief[:, -1]*succ_prob
        move_belief[:, 1:] = np.add(move_belief[:, 1:], previous_belief[:, :-1]*succ_prob)

    return stay_belief + move_belief
    # END: 2b


def update(prediction, measurement):
    """ Update the prediction given the measurement.

    Args:
        prediction: A numpy.ndarray with the current prediction
        measurement: The measurement of the robot's position as tuple (x,y) 

    Returns:
        The updated belief state as ndarray.
    """
    # START 2c
    meas_probs = [0.2, 0.1, 0.05]
    new_pred = np.array(prediction)
    for pos, p in np.ndenumerate(prediction):
        dist = distance.cityblock(pos, measurement)
        new_pred[pos] = p *  meas_probs[dist] if dist < len(meas_probs) else 0.0
    return new_pred / new_pred.sum()
    # END 2c


def initialize_plot(iterations):
    """ Prepare a matplotlib.pyplot.

    Args:
        iterations: The number of plots that will be shown

    Returns:
        A pair (fig, axs) containing the pyplot.subplots
    """
    fig, axs = pyplot.subplots(2, math.floor(iterations / 2) + 1)
    fig.tight_layout()
    return (fig, axs)


def update_plot(ax, belief, action, measurement):
    """ Add the given data to the plot.

    Args:
        ax: The Axes object to plot the data into
        belief: The belief state to plot
        action: The action belonging to the belief state, as string
        measurement: The measurement belonging to the belief state
    """
    plot_data = np.swapaxes(belief, 0, 1)
    ax.set_title("{}, {}".format(action, measurement))
    im = ax.imshow(plot_data, origin='lower')
    ax.set_xticks(np.arange(plot_data.shape[1]))
    ax.set_yticks(np.arange(plot_data.shape[0]))
    for pos, bel in np.ndenumerate(plot_data):
        ax.text(pos[1],
                pos[0],
                "{:1.2f}".format(bel),
                ha="center",
                va="center",
                color="w")


def iterative_bayes_filter(grid_shape, actions, measurements):
    """ Iteratively apply the bayes filter and display plots at the end.

    Args:
        grid_shape: The shape (xs, ys) of the grid the robot is operating on
        actions: A list of actions to take
        measurements: A list of measurements taken after the actions
    """
    assert (len(actions) == len(measurements))
    belief = initialize_belief(grid_shape)
    fig, axs = initialize_plot(len(actions))
    update_plot(axs.flat[0], belief, "INIT", None)
    for i in range(len(actions)):
        prediction = predict(belief, actions[i])
        belief = update(prediction, measurements[i])
        assert (math.isclose(belief.sum(), 1))
        update_plot(axs.flat[i + 1], belief, actions[i].name, measurements[i])
    pyplot.show()


def initialize_runs():
    """ Initialize pre-defined runs.

    Returns:
        A list containing pre-defined runs. Each entry in the list is a pair of
        lists, defining actions and sensor measurements.
    """
    runs = []
    runs.append(([
        Action.LEFT,
        Action.UP,
        Action.RIGHT,
        Action.RIGHT,
        Action.DOWN,
        Action.DOWN,
        Action.LEFT,
    ], [
        (1, 2),
        (1, 3),
        (2, 3),
        (3, 3),
        (3, 2),
        (3, 1),
        (2, 1),
    ]))
    runs.append(([
        Action.LEFT,
        Action.LEFT,
        Action.UP,
        Action.RIGHT,
        Action.DOWN,
        Action.RIGHT,
        Action.DOWN,
        Action.DOWN,
        Action.DOWN,
    ], [
        (2, 2),
        (1, 2),
        (1, 2),
        (2, 3),
        (2, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
    ]))
    runs.append(([
        Action.LEFT,
        Action.LEFT,
        Action.LEFT,
        Action.LEFT,
        Action.LEFT,
    ], [
        (3, 2),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 2),
    ]))
    runs.append(([
        Action.LEFT,
        Action.UP,
        Action.RIGHT,
        Action.RIGHT,
        Action.DOWN,
        Action.DOWN,
        Action.LEFT,
    ], [
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
    ]))
    return runs


def main():
    runs = initialize_runs()
    parser = argparse.ArgumentParser()
    parser.add_argument('-x',
                        type=int,
                        default=5,
                        help="Size of the grid in x direction")
    parser.add_argument('-y',
                        type=int,
                        default=5,
                        help="Size of the grid in y direction")
    parser.add_argument('-r',
                        '--run',
                        help='The pre-defined run to execute',
                        type=int,
                        choices=range(len(runs)),
                        default=1)
    args = parser.parse_args()
    iterative_bayes_filter((args.x, args.y), *runs[args.run])


if __name__ == '__main__':
    main()
