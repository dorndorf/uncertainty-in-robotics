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
Uncertainty in Robotics 2020 Assignment 3.1
"""

import argparse

import matplotlib.pyplot as pyplot
import numpy as np
from scipy.spatial import distance
from scipy.stats import norm


class MonteCarloLocalizer:
    def __init__(self,
                 map_limits,
                 landmarks,
                 num_particles=100,
                 rng=np.random.default_rng()):
        """ Initialize the localizer.

        Args:
            map_limits: The limits of the map as tuple (xmin, xmax, ymin, ymax)
            landmarks: A dictionary of known landmarks
            num_particles: The number of particles to use
            rng: A random number generator
        """
        self.map_limits = map_limits
        self.num_particles = num_particles
        self.landmarks = landmarks
        self.rng = rng
        self.particles = None

    def initialize_particles(self):
        """ Initialize the particles with a uniform distribution within the limits of the map.
        
        Returns:
            A list of particles of the form (x, y, theta), where x and y are within the map limits, and theta is from [-pi, pi].
        """
        # START: 1a
        x_values = np.random.uniform(self.map_limits[0], self.map_limits[1], self.num_particles)
        y_values = np.random.uniform(self.map_limits[2], self.map_limits[3], self.num_particles)
        theta_values = np.random.uniform(-np.pi, np.pi, self.num_particles)
        return np.stack((x_values, y_values, theta_values), axis=-1)
        # END: 1a

    def initialize_plot(self):
        """ Initialize the matplotlib plot. """
        pyplot.axis(self.map_limits)
        pyplot.ion()
        pyplot.show()

    def update_plot(self):
        """ Update the plot with the latest particles. """
        pyplot.clf()
        pyplot.axis(self.map_limits)
        xs = [p[0] for p in self.particles]
        ys = [p[1] for p in self.particles]
        pyplot.scatter(xs, ys, marker='.')
        for p in self.particles:
            pyplot.scatter([p[0]], [p[1]],
                           c='orange',
                           marker=[(0, 0), (np.cos(p[2]), np.sin(p[2]))])
        pyplot.pause(1)

    def sample_odometry_model(self, pos, odometry):
        """ Compute a sample following the odometry model.

        Args:
            pos: The last position as tuple (x, y, theta)
            odometry: The odometry information as a pair (start, end) of two (x, y, theta) positions
        
        Returns:
            The sample as tuple (x, y, theta)
        """
        # START: 1d
        a1 = 0.1
        a2 = 0.1
        a3 = 0.05
        a4 = 0.05
        o_start, o_end = odometry
        rot1 = np.arctan2(o_end[1] - o_start[1], o_end[0] - o_start[0]) - o_start[2]
        trans = distance.euclidean(o_start[:-1], o_end[:-1])
        rot2 = o_end[2] - o_start[2] - rot1

        rot1 -= self.rng.normal(np.sqrt(a1 * np.square(rot1) + a2 * np.square(trans)))
        trans -= self.rng.normal(np.sqrt(a3 * np.square(trans) + a4 * (np.square(rot1) + np.square(rot2))))
        rot2 -= self.rng.normal(np.sqrt(a1 * np.square(rot2) + a2 * np.square(trans)))

        x_new = pos[0] + trans * np.cos(pos[2] + rot1)
        y_new = pos[1] + trans * np.sin(pos[2] + rot1)
        theta_new = pos[2] + rot1 + rot2
        return [x_new, y_new, theta_new]
        # END: 1d

    def landmark_model(self, pos, landmark_measurements):
        """ Compute the probability of the given position given landmark readings.

        Args:
            pos: A position as tuple (x, y, theta)
            landmark_measurements: A dictionary {name: reading} for named landmarks,
                each reading as pair (r, phi)

        Returns:
            The probability of the reading given the position
        """
        # START: 1c
        meas_distance = landmark_measurements['center'][0]
        meas_angle = landmark_measurements['center'][1]
        exp_distance = distance.euclidean(pos[:-1], self.landmarks['center'])
        exp_angle = np.arctan2(self.landmarks['center'][1] - pos[1], self.landmarks['center'][0] - pos[0]) - pos[2]
        prob = norm.pdf(exp_distance, loc=meas_distance, scale=0.2) * norm.pdf(exp_angle, loc=meas_angle,
                                                                               scale=np.pi / 16)

        return np.abs(prob)
        # END: 1c

    def resample(self, particles, weights):
        """ Resample from the given particles according to the given weights.

        Args:
            particles: A list of particles
            weights: The weights of the particles

        Returns:
            A list of particles with the same size as the input,
            resampled according to the weights
        """
        # START: 1b
        weighted_samples = np.random.choice(np.arange(self.num_particles), self.num_particles, p=weights)
        return np.take(particles, weighted_samples, axis=0)
        # END: 1b

    def mcl(self, odometry, landmark_measurement):
        """ Monte Carlo Localization. Updates the object's particles following the MCL algorithm.

        Args:
            odometry: An odometry reading as a pair (start, end) of two (x, y, theta) positions
            landmark_measurement: A dictionary {name: reading} for named landmarks,
                each reading as pair (r, phi)

        """
        # START: 1e
        particles = []
        weights = np.zeros(self.num_particles)
        for i, pos in enumerate(self.particles):
            new_pos = self.sample_odometry_model(pos, odometry)
            weights[i] = self.landmark_model(new_pos, landmark_measurement)
            particles.append(new_pos)

        weights = np.array(weights)/np.sum(weights)
        self.particles = self.resample(particles, weights)
        # END: 1e

    def run(self, odometry, landmark_measurements):
        """ Iteratively run Monte Carlo Localization.

        Args:
            odometry: A list of odometry readings
            landmark_measurements: A list of landmark measurements
        """
        assert len(odometry) == len(landmark_measurements)
        self.initialize_plot()
        self.particles = self.initialize_particles()
        self.update_plot()
        for i in range(len(odometry)):
            self.mcl(odometry[i], landmark_measurements[i])
            self.update_plot()


def init_runs():
    runs = []
    runs.append(([
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (0, 0, 0)),
    ], [
        {
            'center': [4.1, 0]
        },
        {
            'center': [3.1, 0]
        },
        {
            'center': [2.2, 0]
        },
        {
            'center': [1, 0]
        },
        {
            'center': [1, 0]
        },
    ]))
    runs.append(([
        ((0, 0, 0), (0, 0, np.pi / 4)),
        ((0, 0, 0), (0, 0, np.pi / 4)),
        ((0, 0, 0), (0, 0, np.pi / 4)),
        ((0, 0, 0), (0, 0, np.pi / 4)),
        ((0, 0, 0), (0, 0, np.pi / 4)),
        ((0, 0, 0), (0, 0, np.pi / 4)),
        ((0, 0, 0), (0, 0, np.pi / 4)),
        ((0, 0, 0), (0, 0, np.pi / 4)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
    ], [
        {
            'center': [1, np.pi / 4]
        },
        {
            'center': [1, 2 * np.pi / 4]
        },
        {
            'center': [1, 3 * np.pi / 4]
        },
        {
            'center': [1, 4 * np.pi / 4]
        },
        {
            'center': [1, 5 * np.pi / 4]
        },
        {
            'center': [1, 6 * np.pi / 4]
        },
        {
            'center': [1, 7 * np.pi / 4]
        },
        {
            'center': [1, 8 * np.pi / 4]
        },
        {
            'center': [2, 8 * np.pi / 4]
        },
        {
            'center': [3, 8 * np.pi / 4]
        },
        {
            'center': [4, 8 * np.pi / 4]
        },
    ]))
    runs.append(([
        ((0, 0, 0), (0, -0.5, 0)),
        ((0, 0, 0), (0, -0.5, 0)),
        ((0, 0, 0), (0, -0.5, 0)),
        ((0, 0, 0), (0, -0.5, 0)),
        ((0, 0, 0), (0, -0.5, 0)),
        ((0, 0, 0), (0, -0.5, 0)),
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 0)),
    ], [
        {
            'center': [0.5, np.pi / 2]
        },
        {
            'center': [1.0, np.pi / 2]
        },
        {
            'center': [1.5, np.pi / 2]
        },
        {
            'center': [2.0, np.pi / 2]
        },
        {
            'center': [2.5, np.pi / 2]
        },
        {
            'center': [3.0, np.pi / 2]
        },
        {
            'center': [3.0, np.pi / 2]
        },
        {
            'center': [3.0, np.pi / 2]
        },
    ]))
    runs.append(([
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 0)),
    ], [
        {
            'center': [2.0, 2.0]
        },
        {
            'center': [2.2, 2.0]
        },
        {
            'center': [2.2, 2.5]
        },
        {
            'center': [2.0, 2.0]
        },
        {
            'center': [1.8, 2.0]
        },
    ]))

    return runs


def main():
    runs = init_runs()
    parser = argparse.ArgumentParser('Monte Carlo Localizer')
    parser.add_argument('-n',
                        '--num-particles',
                        help='The number of particles',
                        type=int,
                        default=1000)
    parser.add_argument('-r',
                        '--run',
                        help='The pre-defined run to execute',
                        type=int,
                        choices=range(len(runs)),
                        default=0)

    args = parser.parse_args()
    landmarks = {'center': [5, 5]}
    mcl = MonteCarloLocalizer([0, 10, 0, 10], landmarks, args.num_particles)
    odometry = runs[args.run][0]
    landmark_measurements = runs[args.run][1]
    mcl.run(odometry, landmark_measurements)


if __name__ == '__main__':
    main()
