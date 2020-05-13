#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
"""
Uncertainty in Robotics, Exercise 1.1
"""

import argparse
import random
import numpy as np

from enum import Enum

weather_prob = [[0.8, 0.2, 0.0],
                [0.4, 0.4, 0.2],
                [0.2, 0.6, 0.2]]

class Choices(Enum):
    SUNNY = 0
    CLOUDY = 1
    RAINY = 2

def num_to_weather(w_i):
    if w_i == 0:
        return 'SUNNY'
    elif w_i == 1:
        return 'CLOUDY'
    elif w_i == 2:
        return 'RAINY'
    else:
        raise (ValueError, 'Undefined weather number.')


def generate_sequence(initial, k):
    """ Generate a sequence of k days given the initial day. """
    # START: 1c
    sequence = np.ones(k, int)*initial.value
    for i, w in enumerate(sequence):
        if i > 0:
            sequence[i] = np.random.choice([0, 1, 2], p=weather_prob[sequence[i-1]])
    sequence = [Choices[num_to_weather(s)] for s in sequence]
    sequence = np.insert(sequence, 0, initial)
    return sequence
    # END: 1c


def compute_distribution(sequence):
    """ Compute the stationary distribution from the given sequence."""
    # START: 1d
    distribution = {}
    for c in Choices:
        distribution[c] = np.count_nonzero(sequence == c) / sequence.size
    return distribution
    # END: 1d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('initial',
                        help='initial value',
                        choices=[c.name for c in Choices])
    parser.add_argument('-s',
                        '--sequence',
                        action='store_true',
                        help='Print the generated sequence')
    parser.add_argument('-d',
                        '--distribution',
                        action='store_true',
                        help='Print the stationary distribution')
    parser.add_argument('count', help='number of days', type=int)
    args = parser.parse_args()
    sequence = generate_sequence(Choices[args.initial], args.count)
    if args.sequence:
        print('Generated sequence: {}'.format([c.name for c in sequence]))
    if args.distribution:
        distribution = compute_distribution(sequence)
        print('Distribution:\n  SUNNY: {}\n  CLOUDY: {}\n  RAINY: {}'.format(
            distribution[Choices.SUNNY], distribution[Choices.CLOUDY],
            distribution[Choices.RAINY]))


if __name__ == '__main__':
    main()
