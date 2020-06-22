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
from numpy.random import binomial
from scipy.stats import norm


class MonteCarloLocalizer:
    def __init__(self,
                 map_limits,
                 landmarks,
                 num_particles=100,
                 alpha_slow=0.1,
                 alpha_fast=0.5,
                 rng=np.random.default_rng()):
        """ Initialize the localizer.

        Args:
            map_limits: The limits of the map as tuple (xmin, ymin, xmax, ymax)
            landmarks: A dictionary of known landmarks
            num_particles: The number of particles to use
            rng: A random number generator
        """
        self.map_limits = map_limits
        self.num_particles = num_particles
        self.landmarks = landmarks
        self.rng = rng
        self.w_slow = 1 / self.num_particles
        self.w_fast = 1 / self.num_particles
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast

    def initialize_particles(self, num_particles):
        """ Initialize the particles with a uniform distribution within the limits of the map.
        
        Returns:
            A list of particles of the form (x, y, theta), where x and y are
            within the map limits, and theta is from [-pi, pi].
        """
        return self.rng.uniform(
            (self.map_limits[0], self.map_limits[2], -np.pi),
            (self.map_limits[1], self.map_limits[3], np.pi),
            (num_particles, 3))

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
        start = odometry[0]
        end = odometry[1]
        rot1 = np.arctan2(end[1] - start[1], end[0] - start[0]) - start[2]
        trans = np.sqrt(
            np.square(end[0] - start[0]) + np.square(end[1] - start[1]))
        rot2 = end[2] - start[2] - rot1

        alpha1 = 0.1
        alpha2 = 0.1
        alpha3 = 0.05
        alpha4 = 0.05

        noisy_rot1 = rot1 - self.rng.normal(
            scale=np.sqrt(alpha1 * np.square(rot1) +
                          alpha2 * np.square(trans)))
        noisy_trans = trans - self.rng.normal(
            scale=np.sqrt(alpha3 * np.square(trans) +
                          alpha4 * np.square(rot1) + alpha4 * np.square(rot2)))
        noisy_rot2 = rot2 - self.rng.normal(
            scale=np.sqrt(alpha1 * np.square(rot2) +
                          alpha2 * np.square(trans)))

        x = pos[0] + noisy_trans * np.cos(pos[2] + noisy_rot1)
        y = pos[1] + noisy_trans * np.sin(pos[2] + noisy_rot1)
        theta = pos[2] + noisy_rot1 + noisy_rot2

        return (x, y, theta)

    def landmark_model(self, pos, landmark_measurements):
        """ Compute the probability of the given position given landmark readings.

        Args:
            pos: A position as tuple (x, y, theta)
            landmark_measurements: A dictionary {name: reading} for named landmarks,
                each reading as pair (r, phi)

        Returns:
            The probability of the reading given the position
        """
        prob = 1.0
        for landmark, measurement in landmark_measurements.items():
            dx = self.landmarks[landmark][0] - pos[0]
            dy = self.landmarks[landmark][1] - pos[1]
            r = np.sqrt(np.square(dx) + np.square(dy))
            phi = np.arctan2(dy, dx) - pos[2]
            sigma_r = 0.2
            sigma_phi = np.pi / 16
            prob *= norm(0, sigma_r).pdf(r - measurement[0])
            prob *= norm(0, sigma_phi).pdf(phi - measurement[1])
        return prob

    def resample(self, particles, weights):
        """ Resample from the given particles according to the given weights.

        Args:
            particles: A list of particles
            weights: The weights of the particles

        Returns:
            A list of particles with the same size as the input,
            resampled according to the weights
        """
        # START: 1
        w_norm = weights / np.sum(weights)
        w_avg = np.sum(weights) / self.num_particles
        self.w_slow += self.alpha_slow * (w_avg - self.w_slow)
        self.w_fast += self.alpha_fast * (w_avg - self.w_fast)
        num_reinit_samples = np.random.binomial(self.num_particles, max(0.0, 1 - (self.w_fast / self.w_slow)))
        weighted_samples = np.random.choice(np.arange(self.num_particles), self.num_particles - num_reinit_samples,
                                            p=w_norm)
        chosen_particles = np.take(particles, weighted_samples, axis=0)
        reinit_particles = self.initialize_particles(num_reinit_samples)
        particles = np.concatenate((chosen_particles, reinit_particles))
        print("Run:")
        print("Number of Random Samples: {}".format(num_reinit_samples))
        print("W avg: {}".format(w_avg))
        print("w slow: {}".format(self.w_slow))
        print("w fast: {}".format(self.w_fast))

        return particles
        # END: 1

    def mcl(self, odometry, landmark_measurement):
        """ Monte Carlo Localization. Updates the object's particles following the MCL algorithm.

        Args:
            odometry: An odometry reading as a pair (start, end) of two (x, y, theta) positions
            landmark_measurement: A dictionary {name: reading} for named landmarks,
                each reading as pair (r, phi)

        """
        weights = []
        prediction = []
        for particle in self.particles:
            p = self.sample_odometry_model(particle, odometry)
            prediction.append(p)
            w = self.landmark_model(p, landmark_measurement)
            weights.append(w)
        self.particles = self.resample(prediction, weights)

    def run(self, odometry, landmark_measurements):
        """ Iteratively run Monte Carlo Localization.

        Args:
            odometry: A list of odometry readings
            landmark_measurements: A list of landmark measurements
        """
        assert len(odometry) == len(landmark_measurements)
        self.initialize_plot()
        self.particles = self.initialize_particles(self.num_particles)
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
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0)),
    ], [
        {
            'center': [4.1, 0],
            'left': [1, np.pi],
        },
        {
            'center': [3.1, 0],
            'left': [2, np.pi],
        },
        {
            'center': [2.2, 0],
            'left': [3, np.pi],
        },
        {
            'center': [1, 0],
            'left': [4, np.pi],
        },
        {
            'center': [0.1, 0],
            'left': [5, np.pi],
        },
        {
            'center': [1, np.pi],
            'left': [5.5, np.pi],
        },
        {
            'center': [4.1, 0],
            'left': [1, np.pi],
        },
        {
            'center': [3.1, 0],
            'left': [2, np.pi],
        },
        {
            'center': [2.2, 0],
            'left': [3, np.pi],
        },
        {
            'center': [1, 0],
            'left': [4, np.pi],
        },
        {
            'center': [0.1, 0],
            'left': [5, np.pi],
        },
        {
            'center': [1, np.pi],
            'left': [6, np.pi],
        },
        {
            'center': [2.05, np.pi],
            'left': [7, np.pi],
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
                        default=500)
    parser.add_argument('-r',
                        '--run',
                        help='The pre-defined run to execute',
                        type=int,
                        choices=range(len(runs)),
                        default=0)

    args = parser.parse_args()
    landmarks = {'center': [5, 5], 'left': [0, 5]}
    mcl = MonteCarloLocalizer([0, 10, 0, 10], landmarks, args.num_particles)
    odometry = runs[args.run][0]
    landmark_measurements = runs[args.run][1]
    mcl.run(odometry, landmark_measurements)


if __name__ == '__main__':
    main()
